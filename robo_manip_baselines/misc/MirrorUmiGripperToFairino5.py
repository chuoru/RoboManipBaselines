"""Live "gripper-only shadow teleop": mirrors the UMI handheld rig's
ArUco/AprilTag-measured gripper width (see
RealUMIEnvBase._estimate_gripper_percent_closed_from_markers,
common/utils/ArucoGripperUtils.py) onto a REAL Fairino5 robot arm's real
gripper, in real time, WITHOUT tracking/mirroring the UMI rig's pose -- the
FR5 arm stays exactly where move_to_init_pose() puts it (its own
init_qpos); every step re-sends its own just-measured joint position as the
command (a hold-still command, not a fresh target), so
overwrite_command_for_safety's velocity clamp always sees ~zero commanded
delta for the arm. Only the gripper channel carries a live, changing
command.

Runs two independent envs side by side (same general shape as
misc/TeleopUmiWithMujocoMirror.py's real-UMI + MuJoCo-mirror pair, but here
both sides are real hardware and only the gripper channel is bridged, not
the arm pose): RealUMIDemoEnv (for the marker-measured gripper width) and
RealFairino5DemoEnv (for the real gripper actuation). Neither is driven
through TeleopBase/bin/Teleop.py's phase machinery -- this is a standalone
loop, not a data-recording session.

HARDWARE SAFETY:
  - The currently-mounted FR5 gripper is "tool_do" (binary IAI gripper, see
    RealFairino5EnvBase._send_gripper_command): it thresholds the UMI
    percent-closed at 50%, so what you'll see is a binary snap open/closed
    as the real UMI gripper crosses its halfway point, NOT smooth
    continuous tracking.
  - move_to_init_pose() is called once at startup (same as normal
    bin/Teleop.py RealFairino5Demo sessions) and DOES physically move the
    arm to its init pose before this script's hold-still loop begins.
  - Run with --dry_run first (no robot connection at all; FR5 commands are
    printed instead of transmitted -- see RealFairino5EnvBase's dry_run) to
    check the gripper values flowing through look right before touching
    the real arm.

Usage:
    # 1. Validate logic first, no real robot connection:
    python ./misc/MirrorUmiGripperToFairino5.py \\
        --umi_config ./envs/configs/RealUMIDemo.yaml --dry_run
    # 2. Once confirmed, drop --dry_run to actually drive the FR5's gripper:
    python ./misc/MirrorUmiGripperToFairino5.py \\
        --umi_config ./envs/configs/RealUMIDemo.yaml
Press Ctrl+C to stop.
"""

import argparse
import time

import gymnasium as gym
import numpy as np
import yaml

import robo_manip_baselines.envs  # noqa: F401 (registers the gym envs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--umi_config",
        type=str,
        default="./envs/configs/RealUMIDemo.yaml",
        help="same config bin/Teleop.py RealUMIDemo uses -- must set "
        "gripper_marker_config and whichever camera_ids/pointcloud_camera_ids "
        "provides its camera_name",
    )
    parser.add_argument("--robot_ip", type=str, default="192.168.57.2")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="skip the real FR5 connection entirely; print commands instead "
        "of transmitting them (see RealFairino5EnvBase's dry_run)",
    )
    parser.add_argument("--hz", type=float, default=20.0, help="loop rate")
    args = parser.parse_args()

    with open(args.umi_config, "r") as f:
        umi_config = yaml.safe_load(f)
    if umi_config.get("gripper_marker_config") is None:
        raise RuntimeError(
            f"[{__file__}] {args.umi_config} has no gripper_marker_config -- "
            "nothing to mirror."
        )

    print(f"[{__name__}] Connecting to the UMI rig (marker-measured gripper)...")
    umi_env = gym.make("robo_manip_baselines/RealUMIDemoEnv-v0", **umi_config)
    umi_env.reset()

    print(
        f"[{__name__}] Connecting to the FR5 "
        f"{'(dry_run: no real connection)' if args.dry_run else f'at {args.robot_ip}'}..."
    )
    fr5_env = gym.make(
        "robo_manip_baselines/RealFairino5DemoEnv-v0",
        robot_ip=args.robot_ip,
        camera_ids=None,
        gelsight_ids=None,
        dry_run=args.dry_run,
    )
    fr5_unwrapped = fr5_env.unwrapped
    fr5_env.reset()

    print(f"[{__name__}] Moving FR5 to its init pose (physical motion)...")
    fr5_unwrapped.move_to_init_pose()

    gripper_joint_idx = fr5_unwrapped.body_config_list[0].gripper_joint_idxes[0]
    print(
        f"[{__name__}] Mirroring UMI gripper width onto the FR5's real gripper. "
        "Move the UMI rig's gripper by hand. Ctrl+C to stop."
    )

    period = 1.0 / args.hz
    try:
        while True:
            loop_start = time.time()

            umi_env.step(umi_env.unwrapped.action_space.sample() * 0)
            umi_obs = umi_env.unwrapped._get_obs()
            gripper_percent_closed = float(umi_obs["joint_pos"][-1])

            # Hold the arm still: re-send its own just-measured joint
            # position as the command (not a fresh target), so
            # overwrite_command_for_safety's velocity clamp always sees
            # ~zero commanded delta for the arm -- only the gripper element
            # of this action actually changes step to step.
            fr5_action = np.zeros(fr5_unwrapped.action_space.shape, dtype=np.float32)
            fr5_action[fr5_unwrapped.body_config_list[0].arm_joint_idxes] = (
                fr5_unwrapped.arm_joint_pos_actual
            )
            fr5_action[gripper_joint_idx] = gripper_percent_closed
            fr5_env.step(fr5_action)

            print(
                f"\r[{__name__}] UMI gripper: {gripper_percent_closed:5.1f}% closed  "
                f"-> FR5 gripper: {'CLOSED' if gripper_percent_closed >= 50.0 else 'OPEN  '}",
                end="",
                flush=True,
            )

            elapsed = time.time() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print(f"\n[{__name__}] Stopping.")
    finally:
        umi_env.close()
        fr5_env.close()
