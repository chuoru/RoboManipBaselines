"""Plots an ORB_SLAM3 camera trajectory (SaveTrajectoryEuRoC format:
"timestamp_ns tx ty tz qx qy qz qw" per line, see
AddSlamPoseToRmb.py's docstring) so position/orientation and tracking
coverage can be visually sanity-checked before trusting it -- same role as
misc/PlotInsta360Imu.py plays for raw IMU data.

Usage:
    python ./misc/insta360_slam_pipeline/PlotSlamTrajectory.py \\
        <trajectory.txt> [<trajectory.txt> ...] \\
        [--plot_output <plot_output.png>] [--show]
"""

import argparse
import os

import numpy as np
from scipy.spatial.transform import Rotation


def parse_argument():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "trajectory_path",
        type=str,
        nargs="+",
        help="one or more ORB_SLAM3 *_CameraTrajectory.txt file(s) (see "
        "AddSlamPoseToRmb.py's run_mono_inertial_video)",
    )
    parser.add_argument(
        "--plot_output",
        type=str,
        default=None,
        help="output plot PNG path. If omitted, or if multiple "
        "trajectory_path are given, derived as <trajectory_path>_plot.png "
        "next to each input",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="also display the plot interactively",
    )
    return parser.parse_args()


def load_trajectory(trajectory_path):
    """Returns {"t": (N,) seconds relative to the first pose,
    "pos": (N,3), "rotvec_deg": (N,3)}."""
    rows = np.loadtxt(trajectory_path)
    if rows.ndim == 1:
        rows = rows[None, :]
    t = rows[:, 0] * 1e-9
    t = t - t[0]
    pos = rows[:, 1:4]
    quat_xyzw = rows[:, 4:8]
    rotvec_deg = Rotation.from_quat(quat_xyzw).as_rotvec(degrees=True)
    return {"t": t, "pos": pos, "rotvec_deg": rotvec_deg}


def plot_trajectory(data, trajectory_path, plot_output, show):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, pos, rotvec_deg = data["t"], data["pos"], data["rotvec_deg"]

    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

    for i, axis_name in enumerate("xyz"):
        axes[0].plot(t, pos[:, i], label=axis_name)
    axes[0].set_ylabel("position [m]")
    axes[0].legend()
    axes[0].set_title("Camera position (SLAM world frame)")
    axes[0].grid(True, alpha=0.3)

    for i, axis_name in enumerate("xyz"):
        axes[1].plot(t, rotvec_deg[:, i], label=axis_name)
    axes[1].set_ylabel("rotation vector [deg]")
    axes[1].legend()
    axes[1].set_title("Camera orientation (axis-angle, SLAM world frame)")
    axes[1].grid(True, alpha=0.3)

    # Large frame-to-frame jumps are the single most useful "did tracking
    # glitch/relocalize" signal available from this trajectory file alone
    # (SaveTrajectoryEuRoC silently drops untracked frames entirely rather
    # than flagging them -- see AddSlamPoseToRmb.py's docstring -- so a
    # jump across a GAP in sample spacing, not just a large raw delta, is
    # the thing to look for here).
    pos_speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) / np.diff(t)
    axes[2].plot(t[1:], pos_speed)
    axes[2].set_ylabel("speed [m/s]")
    axes[2].set_title("Frame-to-frame position speed (spikes suggest a "
                       "tracking glitch/relocalization)")
    axes[2].grid(True, alpha=0.3)

    dt_ms = np.diff(t) * 1000.0
    axes[3].plot(t[1:], dt_ms)
    axes[3].set_ylabel("sample interval [ms]")
    axes[3].set_xlabel("time [s]")
    axes[3].set_title("Sample interval (a gap here = frames SLAM lost "
                       "tracking on and silently dropped)")
    axes[3].grid(True, alpha=0.3)

    fig.suptitle(f"SLAM trajectory check: {os.path.basename(trajectory_path)} "
                 f"({len(t)} poses, {t[-1]:.1f}s span)")
    fig.tight_layout()

    fig.savefig(plot_output)
    print(f"[PlotSlamTrajectory] Saved plot to {plot_output}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    args = parse_argument()
    multiple = len(args.trajectory_path) > 1

    for trajectory_path in args.trajectory_path:
        data = load_trajectory(trajectory_path)

        if args.plot_output is not None and not multiple:
            plot_output = args.plot_output
        else:
            plot_output = os.path.splitext(trajectory_path)[0] + "_plot.png"

        plot_trajectory(data, trajectory_path, plot_output, args.show)


if __name__ == "__main__":
    main()
