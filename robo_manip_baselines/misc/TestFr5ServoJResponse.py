"""Measure how well the real FR5 actually follows a ServoJ command stream,
using the simplest possible motion: a slow, pure rotation of the TCP about one
of its own axes, with the arm otherwise still.

WHY THIS EXISTS: replaying a recorded UMI demo on the real arm
(misc/ReplayUmiOnFairino5.py --real) produced motion that was visibly missing
its roll component, and the log showed every rotation axis coming out
attenuated rather than lagged-but-complete:

    TCP-local rotation amplitude, commanded -> measured
      pitch axis   26.7 deg -> 12.8 deg  (48%)
      yaw axis     85.6 deg -> 33.8 deg  (40%)
      roll axis    88.4 deg -> 23.1 deg  (26%)

and per joint, tracking got worse the more motion was asked for:

      J1  14.9 deg -> 14.8 (100%)    J4  105.5 deg -> 29.2 ( 28%)
      J3  16.9 deg -> 16.6 ( 98%)    J6   93.7 deg -> 24.0 ( 26%)

That pattern is not a speed-limit violation (commanded speeds averaged
0.5 deg/s and peaked at 31 deg/s, against the arm's 30 deg/s configured
limit and a far higher physical one), and the usual suspects were ruled out
against the recorded log: overwrite_command_for_safety's velocity clamp never
engaged (0.00% of steps), and the command_smoothing_alpha EMA preserves
amplitude (105.5 deg -> 105.1 deg). The same trajectory replays correctly in
MuJoCo, whose step() applies none of the real path's command processing.

So the remaining suspects are on the ServoJ side, which this script isolates:
its cmdT/filterT/gain parameters, and whether the controller simply cannot
track a continuously-updating target at the rate we stream it.

WHAT IT DOES: drives ONE TCP-local rotation axis through a slow sinusoid
(default +/-15 deg over 20 s) while holding position fixed, via the same
ArmManager IK -> env.step() path the replay uses, then reports commanded vs
measured amplitude and phase lag. Sweeping --servoj_gain / --servoj_filter_t
across runs shows directly whether those parameters are the cause.

SAFETY: small, slow, single-axis wrist motion with the arm parked at its ready
pose -- deliberately the least dangerous thing that still reproduces the
symptom. Defaults to dry_run; pass --real to move the arm.

Usage:
    # See the commands without moving anything
    python ./misc/TestFr5ServoJResponse.py --axis roll

    # On hardware, current settings (the ones that tracked poorly)
    python ./misc/TestFr5ServoJResponse.py --axis roll --real

    # Then sweep the ServoJ knobs to see if they are the cause
    python ./misc/TestFr5ServoJResponse.py --axis roll --real --servoj_gain 100
    python ./misc/TestFr5ServoJResponse.py --axis roll --real --servoj_filter_t 0.01
"""

import argparse
import time

import gymnasium as gym
import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import ArmManager, get_se3_from_pose

# TCP-local axis each name rotates about, in the TCP convention this rig uses
# (forward = +Z, down = +Y, left = +X -- see teleop/calibrate_vive_rotation.py).
AXIS_VECTORS = {
    "roll": np.array([0.0, 0.0, 1.0]),
    "pitch": np.array([1.0, 0.0, 0.0]),
    "yaw": np.array([0.0, 1.0, 0.0]),
}


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--axis",
        type=str,
        default="roll",
        choices=[*AXIS_VECTORS.keys(), "all"],
        help="which TCP-local rotation axis to exercise ('all' runs each in turn)",
    )
    parser.add_argument(
        "--amplitude_deg",
        type=float,
        default=15.0,
        help="peak rotation amplitude [deg] (motion is +/- this)",
    )
    parser.add_argument(
        "--period_sec",
        type=float,
        default=20.0,
        help="time for one full back-and-forth cycle [s]. Combined with "
        "--amplitude_deg this sets the peak commanded TCP angular speed "
        "(2*pi*amplitude/period).",
    )
    parser.add_argument(
        "--cycles", type=float, default=2.0, help="number of full cycles to run"
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="actually connect to and move the FR5 (omit to stay in dry_run, "
        "which only prints the commands)",
    )
    parser.add_argument("--robot_ip", type=str, default="192.168.57.2")
    parser.add_argument("--gripper_hand_type", type=str, default="right")
    parser.add_argument("--gripper_modbus_port", type=str, default="/dev/ttyUSB0")
    parser.add_argument(
        "--gripper_type", type=str, default="tool_do", choices=["linker_hand", "tool_do"]
    )
    parser.add_argument("--gripper_do_close_id", type=int, default=0)
    parser.add_argument("--gripper_do_open_id", type=int, default=1)
    parser.add_argument(
        "--servoj_gain",
        type=float,
        default=0.0,
        help="ServoJ's gain parameter (default 0.0 matches the vendor "
        "examples and the current replay path)",
    )
    parser.add_argument(
        "--servoj_filter_t",
        type=float,
        default=0.0,
        help="ServoJ's filterT parameter (default 0.0, as above)",
    )
    parser.add_argument(
        "--command_smoothing_alpha",
        type=float,
        default=0.3,
        help="the env's own EMA on the commanded joint position. 1.0 "
        "disables it, which isolates ServoJ from this smoothing.",
    )
    parser.add_argument(
        "--log_csv",
        type=str,
        default=None,
        help="path to save the per-step log (default: skip saving)",
    )
    return parser.parse_args()


def run_axis(env, arm_manager, axis_name, args):
    """Drive one TCP-local axis through a sinusoid; return the per-step log."""
    axis = AXIS_VECTORS[axis_name]
    amplitude = np.deg2rad(args.amplitude_deg)
    dt = env.unwrapped.dt
    n_steps = int(round(args.cycles * args.period_sec / dt))

    # Everything is relative to wherever the arm is right now, so the arm
    # only ever rotates in place -- no approach move, no translation.
    obs = env.unwrapped._get_obs()
    measured_arm_joint_pos = env.unwrapped.get_joint_pos_from_obs(obs)[
        arm_manager.body_config.arm_joint_idxes
    ]
    arm_manager.arm_joint_pos = measured_arm_joint_pos.copy()
    arm_manager.forward_kinematics()
    base_se3 = arm_manager.current_se3.copy()

    peak_speed_deg_s = np.rad2deg(2 * np.pi * amplitude / args.period_sec)
    print(
        f"[TestFr5ServoJResponse] axis={axis_name}: +/-{args.amplitude_deg} deg "
        f"over {args.period_sec}s x {args.cycles} cycles "
        f"({n_steps} steps at dt={dt * 1000:.0f}ms), peak TCP rate "
        f"{peak_speed_deg_s:.1f} deg/s"
    )

    log = {"t": [], "cmd_ang": [], "meas_ang": [], "cmd_q": [], "meas_q": []}
    t0 = time.time()
    for i in range(n_steps):
        phase = 2 * np.pi * (i * dt) / args.period_sec
        angle = amplitude * np.sin(phase)
        # RIGHT-multiply: rotate about the TCP's OWN axis, matching how the
        # replay composes its recorded rotation deltas.
        target_se3 = pin.SE3(
            base_se3.rotation @ pin.exp3(angle * axis), base_se3.translation
        )
        arm_manager.set_command_eef_pose(target_se3)
        command_arm_joint_pos = arm_manager.arm_joint_pos.copy()
        action = np.concatenate([command_arm_joint_pos, arm_manager.gripper_joint_pos])
        obs = env.step(action)[0]

        measured_arm_joint_pos = env.unwrapped.get_joint_pos_from_obs(obs)[
            arm_manager.body_config.arm_joint_idxes
        ]
        measured_se3 = get_se3_from_pose(
            arm_manager.get_eef_pose_from_joint_pos(measured_arm_joint_pos)
        )
        # Component of the measured rotation about the axis we asked for.
        measured_angle = pin.log3(base_se3.rotation.T @ measured_se3.rotation) @ axis

        log["t"].append(time.time() - t0)
        log["cmd_ang"].append(np.rad2deg(angle))
        log["meas_ang"].append(np.rad2deg(measured_angle))
        log["cmd_q"].append(np.rad2deg(command_arm_joint_pos))
        log["meas_q"].append(np.rad2deg(measured_arm_joint_pos))

    return {k: np.array(v) for k, v in log.items()}


def report(axis_name, log, args):
    cmd, meas = log["cmd_ang"], log["meas_ang"]
    cmd_amp, meas_amp = np.ptp(cmd), np.ptp(meas)
    ratio = meas_amp / max(cmd_amp, 1e-9)

    # Phase lag, from the shift that best correlates the two (in steps, then
    # converted with the run's own measured cadence).
    c, m = cmd - cmd.mean(), meas - meas.mean()
    best_lag, best_cc = 0, -np.inf
    for lag in range(0, max(1, len(c) // 3)):
        cc = np.corrcoef(c[: len(c) - lag], m[lag:])[0, 1] if len(c) - lag > 2 else -1
        if np.isfinite(cc) and cc > best_cc:
            best_cc, best_lag = cc, lag
    step_sec = np.median(np.diff(log["t"])) if len(log["t"]) > 1 else 0.0

    print(f"\n[TestFr5ServoJResponse] === {axis_name} ===")
    print(
        f"  amplitude: commanded {cmd_amp:.2f} deg -> measured {meas_amp:.2f} deg "
        f"({100 * ratio:.1f}%)"
    )
    print(
        f"  phase lag: {best_lag} steps ({best_lag * step_sec:.3f}s), "
        f"correlation {best_cc:.3f}"
    )
    print(f"  loop cadence: {step_sec * 1000:.1f} ms/step")
    print(
        f"  per-joint commanded range [deg]: "
        f"{np.round(np.ptp(log['cmd_q'], axis=0), 2)}"
    )
    print(
        f"  per-joint measured  range [deg]: "
        f"{np.round(np.ptp(log['meas_q'], axis=0), 2)}"
    )
    if ratio > 0.9:
        verdict = "tracks well -- ServoJ is fine at this speed/settings"
    elif ratio > 0.5:
        verdict = "PARTIAL tracking -- attenuated, same failure mode as the replay"
    else:
        verdict = "POOR tracking -- reproduces the replay's missing-rotation symptom"
    print(f"  verdict: {verdict}")
    print(
        f"  (servoj_gain={args.servoj_gain}, servoj_filter_t={args.servoj_filter_t}, "
        f"command_smoothing_alpha={args.command_smoothing_alpha})"
    )
    return ratio


def main():
    args = parse_argument()
    dry_run = not args.real

    print(
        f"[TestFr5ServoJResponse] "
        f"{'DRY RUN (nothing will move)' if dry_run else '*** REAL HARDWARE -- the FR5 WILL move ***'}"
    )
    if not dry_run:
        input(
            "[TestFr5ServoJResponse] The arm will rotate its wrist in place. "
            "Press Enter to confirm, or Ctrl+C to abort..."
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
        gripper_do_close_id=args.gripper_do_close_id,
        gripper_do_open_id=args.gripper_do_open_id,
        dry_run=dry_run,
        servoj_gain=args.servoj_gain,
        servoj_filter_t=args.servoj_filter_t,
        command_smoothing_alpha=args.command_smoothing_alpha,
    )
    env.reset()
    env.unwrapped.move_to_init_pose()
    arm_manager = ArmManager(env.unwrapped, env.unwrapped.body_config_list[0])

    axes = list(AXIS_VECTORS.keys()) if args.axis == "all" else [args.axis]
    logs = {}
    try:
        for axis_name in axes:
            logs[axis_name] = run_axis(env, arm_manager, axis_name, args)
            report(axis_name, logs[axis_name], args)
    except KeyboardInterrupt:
        print("\n[TestFr5ServoJResponse] Interrupted.")
    finally:
        env.close()

    if args.log_csv and logs:
        import csv

        with open(args.log_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["axis", "t", "cmd_ang_deg", "meas_ang_deg"]
                + [f"cmd_j{j + 1}_deg" for j in range(6)]
                + [f"meas_j{j + 1}_deg" for j in range(6)]
            )
            for axis_name, log in logs.items():
                for i in range(len(log["t"])):
                    w.writerow(
                        [axis_name, log["t"][i], log["cmd_ang"][i], log["meas_ang"][i]]
                        + list(log["cmd_q"][i])
                        + list(log["meas_q"][i])
                    )
        print(f"[TestFr5ServoJResponse] Saved log to {args.log_csv}")


if __name__ == "__main__":
    main()
