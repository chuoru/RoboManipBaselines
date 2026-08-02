"""Replay a UMI-collected demonstration (RealUMIDemoEnv, a robot-less handheld
rig -- see envs/real/umi/RealUMIEnvBase.py) onto the FR5 arm
(RealFairino5DemoEnv), by retargeting the recorded EEF trajectory and driving
env.step() with the resulting joint commands.

Unlike bin/Teleop.py's built-in --replay_log (which replays a log recorded by
the SAME env, using its own DOF/frame conventions unchanged -- see
teleop/TeleopBase.py's ReplayPhase), this script retargets across envs: the
UMI rig's recorded EEF pose has no calibrated common frame with the FR5 base
(see misc/CheckUmiFairino5Reachability.py's docstring), so it cannot be
replayed as-is. The same naive delta-pose retarget verified there (translation
delta added onto the FR5's init-pose EEF position; rotation delta rotated via
--vive_config's vive_to_eef_frame_rotation) is reused here, but IK is now done
through the env's own ArmManager (one damped-least-squares Newton step per
frame, warm-started from the previous frame -- exactly what real teleop does)
instead of iterating to convergence offline.

SAFETY: defaults to dry_run (no hardware connection; env prints the ServoJ
command it would have sent). Pass --real only once you have verified the
printed commands look sane (e.g. via CheckUmiFairino5Reachability.py and a
dry-run read-through of this script's own printed output) and are ready to
move the physical arm.
"""

import argparse
import time

import gymnasium as gym
import numpy as np
import pinocchio as pin
import yaml

from robo_manip_baselines.common import ArmManager, DataKey, RmbData

EEF_JOINT_ID = 6  # matches ik_eef_joint_id in RealFairino5EnvBase's ArmConfig


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("rmb_filename", type=str, help="UMI demo .rmb/.hdf5 file")
    parser.add_argument(
        "--real",
        action="store_true",
        help="actually connect to and move the FR5 (DANGEROUS -- omit this to "
        "stay in dry_run, which only prints the commands)",
    )
    parser.add_argument("--robot_ip", type=str, default="192.168.57.2")
    parser.add_argument("--gripper_hand_type", type=str, default="right")
    parser.add_argument("--gripper_modbus_port", type=str, default="/dev/ttyUSB0")
    parser.add_argument(
        "--gripper_type",
        type=str,
        default="tool_do",
        choices=["linker_hand", "tool_do"],
    )
    parser.add_argument("--gripper_do_id", type=int, default=1)
    parser.add_argument("--gripper_do_close_status", type=int, default=0)
    parser.add_argument(
        "--pos_scale",
        type=float,
        default=1.0,
        help="scale applied to the UMI translation delta before retargeting",
    )
    parser.add_argument(
        "--vive_config",
        type=str,
        default=None,
        help="path to a teleop Vive config yaml (e.g. teleop/configs/ViveUMI.yaml) "
        "to source vive_to_eef_frame_rotation from, instead of identity",
    )
    parser.add_argument(
        "--max_linear_vel",
        type=float,
        default=2.0,
        help="[m/s] glitch-filter threshold, see filter_glitches()",
    )
    parser.add_argument(
        "--max_angular_vel",
        type=float,
        default=720.0,
        help="[deg/s] glitch-filter threshold, see filter_glitches()",
    )
    parser.add_argument(
        "--time_scale",
        type=float,
        default=1.0,
        help="slow down replay by this factor (e.g. 3.0 = 3x slower / takes 3x "
        "longer) by inserting interpolated intermediate poses between recorded "
        "UMI frames. The original human-speed motion can require joint speeds "
        "well above joint_vel_limit (e.g. after lowering it to match the real "
        "arm's safe speed); replaying at 1x then relies entirely on "
        "overwrite_command_for_safety's per-step velocity clamp to catch up, "
        "which makes the arm lag further and further behind the demonstrated "
        "path instead of tracking it. Slowing down here (so demanded joint "
        "speed drops below the limit) keeps the arm actually tracking the "
        "path, rather than just being clamped to a safe speed while drifting "
        "off of it.",
    )
    return parser.parse_args()


def interpolate_se3(se3_a, se3_b, alpha):
    """Screw-motion interpolation between two poses (alpha=0 -> se3_a, alpha=1
    -> se3_b), via the same pin.log6/pin.integrate machinery ArmManager's IK
    uses elsewhere in this codebase."""
    return se3_a * pin.exp6(alpha * pin.log6(se3_a.inverse() * se3_b))


def stretch_trajectory(se3_list, gripper_list, time_scale):
    """Insert (time_scale - 1) interpolated poses between each consecutive
    pair of keyframes, so the same env.dt-paced step-by-step replay takes
    time_scale times as long and moves a proportionally smaller distance per
    step (see --time_scale)."""
    if time_scale <= 1.0:
        return se3_list, gripper_list
    n_sub = max(1, round(time_scale))
    stretched_se3_list = [se3_list[0]]
    stretched_gripper_list = [gripper_list[0]]
    for i in range(1, len(se3_list)):
        for k in range(1, n_sub + 1):
            alpha = k / n_sub
            stretched_se3_list.append(
                interpolate_se3(se3_list[i - 1], se3_list[i], alpha)
            )
            stretched_gripper_list.append(
                (1 - alpha) * gripper_list[i - 1] + alpha * gripper_list[i]
            )
    return stretched_se3_list, stretched_gripper_list


def filter_glitches(se3_list, time_seq, max_linear_vel, max_angular_vel):
    """Same glitch filter as misc/CheckUmiFairino5Reachability.py -- see that
    module for why this is needed (Vive lighthouse-occlusion pose jumps)."""
    max_angular_vel_rad = np.deg2rad(max_angular_vel)
    filtered_se3_list = [se3_list[0]]
    skipped_mask = np.zeros(len(se3_list), dtype=bool)
    last_valid_se3 = se3_list[0]
    last_valid_time = time_seq[0]
    for t in range(1, len(se3_list)):
        dt = max(time_seq[t] - last_valid_time, 1e-6)
        linear_vel = (
            np.linalg.norm(se3_list[t].translation - last_valid_se3.translation) / dt
        )
        angular_vel = (
            np.linalg.norm(
                pin.log3(last_valid_se3.rotation.T @ se3_list[t].rotation)
            )
            / dt
        )
        if linear_vel > max_linear_vel or angular_vel > max_angular_vel_rad:
            skipped_mask[t] = True
            filtered_se3_list.append(last_valid_se3)
        else:
            filtered_se3_list.append(se3_list[t])
            last_valid_se3 = se3_list[t]
            last_valid_time = time_seq[t]
    return filtered_se3_list, skipped_mask


def main():
    args = parse_argument()

    if args.vive_config is None:
        vive_to_eef_frame_rotation = np.eye(3)
    else:
        with open(args.vive_config, "r") as f:
            vive_config = yaml.safe_load(f)
        vive_to_eef_frame_rotation = np.array(
            vive_config["vive_to_eef_frame_rotation"], dtype=np.float64
        )

    print(f"[ReplayUmiOnFairino5] Load {args.rmb_filename}")
    with RmbData(args.rmb_filename) as rmb_data:
        umi_pose = rmb_data[DataKey.MEASURED_EEF_POSE][:]  # (T,7): tx,ty,tz,qw,qx,qy,qz
        umi_gripper = rmb_data[DataKey.COMMAND_GRIPPER_JOINT_POS][:]  # (T,1) percent closed
        time_seq = rmb_data[DataKey.TIME][:]

    n_steps = umi_pose.shape[0]
    umi_se3_list = [
        pin.SE3(pin.Quaternion(*umi_pose[t, 3:7]), umi_pose[t, 0:3])
        for t in range(n_steps)
    ]
    umi_se3_list, skipped_mask = filter_glitches(
        umi_se3_list, time_seq, args.max_linear_vel, args.max_angular_vel
    )
    print(
        f"[ReplayUmiOnFairino5] Skipped {int(skipped_mask.sum())}/{n_steps} glitch "
        f"frames (> {args.max_linear_vel} m/s or {args.max_angular_vel} deg/s)"
    )
    umi_se3_0 = umi_se3_list[0]

    dry_run = not args.real
    print(
        f"[ReplayUmiOnFairino5] {'DRY RUN (no hardware will move)' if dry_run else '*** REAL HARDWARE -- the FR5 WILL move ***'}"
    )
    if not dry_run:
        input(
            "[ReplayUmiOnFairino5] Press Enter to confirm you want to move the "
            "real FR5, or Ctrl+C to abort..."
        )

    env = gym.make(
        "robo_manip_baselines/RealFairino5DemoEnv-v0",
        robot_ip=args.robot_ip,
        camera_ids=None,
        gelsight_ids=None,
        pointcloud_camera_ids=None,
        gripper_hand_type=args.gripper_hand_type,
        gripper_modbus_port=args.gripper_modbus_port,
        gripper_type=args.gripper_type,
        gripper_do_id=args.gripper_do_id,
        gripper_do_close_status=args.gripper_do_close_status,
        dry_run=dry_run,
    )
    env.reset()
    env.unwrapped.move_to_init_pose()

    arm_manager = ArmManager(env.unwrapped, env.unwrapped.body_config_list[0])
    init_se3 = arm_manager.current_se3.copy()

    # Same naive delta-pose retarget as CheckUmiFairino5Reachability.py -- see
    # that module's docstring for why this is not yet a real calibrated
    # retargeting transform.
    target_se3_list = []
    for umi_se3_t in umi_se3_list:
        target_translation = init_se3.translation + args.pos_scale * (
            umi_se3_t.translation - umi_se3_0.translation
        )
        delta_umi_rotation = umi_se3_0.rotation.T @ umi_se3_t.rotation
        adjusted_rotation_delta = (
            vive_to_eef_frame_rotation
            @ delta_umi_rotation
            @ vive_to_eef_frame_rotation.T
        )
        target_rotation = init_se3.rotation @ adjusted_rotation_delta
        target_se3_list.append(pin.SE3(target_rotation, target_translation))

    target_se3_list, gripper_list = stretch_trajectory(
        target_se3_list, list(umi_gripper), args.time_scale
    )
    n_replay_steps = len(target_se3_list)

    print(
        f"[ReplayUmiOnFairino5] Replaying {n_replay_steps} steps "
        f"(time_scale={args.time_scale}) over an estimated "
        f"{n_replay_steps * env.unwrapped.dt:.1f}s..."
    )
    try:
        for t in range(n_replay_steps):
            # One damped-least-squares Newton step, warm-started from the
            # previous frame -- the same per-frame IK behavior as real
            # teleop's ArmManager.set_command_eef_pose(), not iterated to
            # convergence like the offline reachability check.
            arm_manager.set_command_eef_pose(target_se3_list[t])
            arm_manager.set_command_gripper_joint_pos(gripper_list[t])

            action = np.concatenate(
                [arm_manager.arm_joint_pos, arm_manager.gripper_joint_pos]
            )
            env.step(action)

            if t % 100 == 0:
                print(f"[ReplayUmiOnFairino5] step {t}/{n_replay_steps}")
    except KeyboardInterrupt:
        print("[ReplayUmiOnFairino5] Interrupted.")
    finally:
        env.close()
        print("[ReplayUmiOnFairino5] Done.")


if __name__ == "__main__":
    main()
