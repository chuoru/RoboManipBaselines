"""Offline check of whether a UMI-collected demonstration (RealUMIDemoEnv, a
robot-less handheld rig -- see envs/real/umi/RealUMIEnvBase.py) is retargetable
onto the real FR5 arm (RealFairino5DemoEnv), without moving any hardware.

The UMI rig and the FR5 base have no calibrated common frame yet (unlike the
teleoperated-robot envs, where ViveInputDevice's vive_world_to_base_frame_rotation
ties the tracker to the robot base -- see teleop/README.md and
teleop/calibrate_vive_world_frame.py). So this script does not claim to compute
the actual retargeting transform; it only applies a naive delta-pose retarget
(the UMI trajectory's motion relative to its first frame, added onto the FR5's
init_qpos EEF pose with an identity rotation frame for translation) to sanity-check that the
*scale* of motion in the recorded demo is plausible for the FR5 to execute --
i.e. it stays within the arm's reach and joint limits and does not run through
a singularity. It is not a substitute for calibrating the real retargeting
transform before running on hardware.

Note on rotation: the recorded MEASURED_EEF_POSE already has the UMI rig's
vive_to_eef_frame_rotation conjugation baked in by ViveInputDevice.set_command_data
at recording time (RealUMIEnvBase._set_action echoes the command straight back
as "measured", since the rig has no physical plant to converge). So the
rotation delta computed here between recorded frames is used as-is -- it must
NOT be conjugated by vive_to_eef_frame_rotation again, which would apply that
rotation twice.
"""

import argparse
import os

import numpy as np
import pinocchio as pin

import robo_manip_baselines.envs as envs_pkg
from robo_manip_baselines.common import DataKey, RmbData
from robo_manip_baselines.envs.real.fairino5.RealFairino5DemoEnv import (
    RealFairino5DemoEnv,
)
from robo_manip_baselines.envs.real.fairino5.RealFairino5EnvBase import (
    RealFairino5EnvBase,
)

FR5_URDF_PATH = os.path.join(
    os.path.dirname(envs_pkg.__file__),
    "assets/common/robots/fairino5_v6/fairino5_v6.urdf",
)
EEF_JOINT_ID = 6  # matches ik_eef_joint_id in RealFairino5EnvBase's ArmConfig
IK_MAX_ITERS = 200
IK_EPS = 1e-4  # [m]/[rad] combined SE3 log-map error norm


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("rmb_filename", type=str, help="UMI demo .rmb/.hdf5 file")
    parser.add_argument(
        "--pos_scale",
        type=float,
        default=1.0,
        help="scale applied to the UMI translation delta before retargeting",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="if set, save a joint-angle-vs-limit plot (png) to this path",
    )
    parser.add_argument(
        "--max_linear_vel",
        type=float,
        default=2.0,
        help="[m/s] frames whose translation implies a higher instantaneous "
        "speed than this (relative to the last accepted frame) are treated as "
        "Vive tracking glitches (lighthouse occlusion/relock) and skipped -- "
        "see the ~40-50cm single-frame jumps found in early UMI recordings",
    )
    parser.add_argument(
        "--max_angular_vel",
        type=float,
        default=720.0,
        help="[deg/s] frames whose rotation implies a higher instantaneous "
        "angular speed than this (relative to the last accepted frame) are "
        "treated as Vive tracking glitches and skipped",
    )
    return parser.parse_args()


def filter_glitches(se3_list, time_seq, max_linear_vel, max_angular_vel):
    """Detect and skip Vive tracking-glitch frames (lighthouse occlusion causing
    libsurvive's pose solver to momentarily jump to a wrong solution -- see the
    ~40-50cm single-frame position jumps found in early UMI recordings). A
    glitched frame is replaced by the last accepted (non-glitched) pose, so
    downstream retargeting sees a brief hold instead of a physically impossible
    jump. Returns (filtered_se3_list, skipped_mask)."""
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


def plot_pose_triad(ax, position, rotation, length, label=None):
    """Draw one RGB xyz-axis triad (quiver arrows) at `position` oriented by
    `rotation`, on a 3D `Axes3D`. Used to make rotation, not just position,
    visually inspectable along a plotted trajectory."""
    colors = ("r", "g", "b")
    for axis_idx, color in enumerate(colors):
        axis_dir = rotation[:, axis_idx]
        ax.quiver(
            *position,
            *(length * axis_dir),
            color=color,
            linewidth=1.5,
            label=label if axis_idx == 0 else None,
        )


def solve_ik(model, data, q_init, target_se3, eef_joint_id):
    """Damped least-squares IK, same formulation as ArmManager.inverse_kinematics,
    but iterated to convergence (or IK_MAX_ITERS) instead of one step per call --
    appropriate here since this is an offline reachability check, not a
    real-time control loop."""
    q = q_init.copy()
    err_norm = np.inf
    for n_iters in range(1, IK_MAX_ITERS + 1):
        pin.forwardKinematics(model, data, q)
        current_se3 = data.oMi[eef_joint_id]
        err_se3 = current_se3.actInv(target_se3)
        err_vec = pin.log(err_se3).vector
        err_norm = np.linalg.norm(err_vec)
        if err_norm < IK_EPS:
            break
        J = pin.computeJointJacobian(model, data, q, eef_joint_id)
        J = -1 * np.dot(pin.Jlog6(err_se3.inverse()), J)
        damping_scale = 1e-6
        dq = -1 * J.T.dot(
            np.linalg.solve(
                J.dot(J.T) + (np.dot(err_vec, err_vec) + damping_scale) * np.eye(6),
                err_vec,
            )
        )
        q = pin.integrate(model, q, dq)
    return q, err_norm, n_iters


def main():
    args = parse_argument()

    model = pin.buildModelFromUrdf(FR5_URDF_PATH)
    data = model.createData()

    # Sourced from RealFairino5DemoEnv.READY_POSE_DEG (not from constructing the
    # env itself, which would try to connect to hardware).
    init_qpos = np.concatenate(
        [np.deg2rad(RealFairino5DemoEnv.READY_POSE_DEG), [0.0]]
    )
    arm_low = RealFairino5EnvBase.action_space.low[0:6].astype(np.float64)
    arm_high = RealFairino5EnvBase.action_space.high[0:6].astype(np.float64)

    q_arm = init_qpos[0:6].copy()
    pin.forwardKinematics(model, data, q_arm)
    init_se3 = data.oMi[EEF_JOINT_ID].copy()

    print(f"[CheckUmiFairino5Reachability] Load {args.rmb_filename}")
    with RmbData(args.rmb_filename) as rmb_data:
        umi_pose = rmb_data[DataKey.MEASURED_EEF_POSE][:]  # (T, 7): tx,ty,tz,qw,qx,qy,qz
        time_seq = rmb_data[DataKey.TIME][:]

    n_steps = umi_pose.shape[0]
    umi_se3_list = [
        pin.SE3(pin.Quaternion(*umi_pose[t, 3:7]), umi_pose[t, 0:3])
        for t in range(n_steps)
    ]
    umi_se3_list, skipped_mask = filter_glitches(
        umi_se3_list, time_seq, args.max_linear_vel, args.max_angular_vel
    )
    n_skipped = int(skipped_mask.sum())
    print(
        f"[CheckUmiFairino5Reachability] Skipped {n_skipped}/{n_steps} glitch "
        f"frames (> {args.max_linear_vel} m/s or {args.max_angular_vel} deg/s)"
    )
    umi_se3_0 = umi_se3_list[0]

    q_log = np.zeros((n_steps, 6))
    err_log = np.zeros(n_steps)
    converged_log = np.zeros(n_steps, dtype=bool)
    limit_violation_log = np.zeros(n_steps, dtype=bool)
    target_se3_list = []

    for t in range(n_steps):
        umi_se3_t = umi_se3_list[t]
        # Naive retarget: translation delta (scaled) added in the FR5 base frame
        # with an identity base-frame rotation, since no real calibration exists
        # yet between this UMI rig and the FR5 base (see module docstring).
        # Rotation delta is used as-is: it already has vive_to_eef_frame_rotation
        # baked in from recording (see module docstring's rotation note) --
        # conjugating by it again here would apply it twice.
        target_translation = init_se3.translation + args.pos_scale * (
            umi_se3_t.translation - umi_se3_0.translation
        )
        delta_umi_rotation = umi_se3_0.rotation.T @ umi_se3_t.rotation
        target_rotation = init_se3.rotation @ delta_umi_rotation
        target_se3 = pin.SE3(target_rotation, target_translation)
        target_se3_list.append(target_se3)

        q_arm, err_norm, _n_iters = solve_ik(model, data, q_arm, target_se3, EEF_JOINT_ID)

        q_log[t] = q_arm
        err_log[t] = err_norm
        converged_log[t] = err_norm < IK_EPS
        limit_violation_log[t] = np.any(q_arm < arm_low) | np.any(q_arm > arm_high)

    kept_mask = ~skipped_mask
    n_kept = int(kept_mask.sum())
    n_converged = int(converged_log[kept_mask].sum())
    n_violations = int(limit_violation_log[kept_mask].sum())
    print(
        f"[CheckUmiFairino5Reachability] {n_steps} steps over "
        f"{time_seq[-1] - time_seq[0]:.1f}s ({n_kept} kept after glitch filtering)"
    )
    print(
        f"[CheckUmiFairino5Reachability] IK converged (err < {IK_EPS}) among kept "
        f"frames: {n_converged}/{n_kept} ({100.0 * n_converged / n_kept:.1f}%)"
    )
    print(
        f"[CheckUmiFairino5Reachability] Joint-limit violations among kept "
        f"frames: {n_violations}/{n_kept} ({100.0 * n_violations / n_kept:.1f}%)"
    )
    print(
        "[CheckUmiFairino5Reachability] Per-joint range used vs. limit [deg]:"
    )
    for j in range(6):
        used_min, used_max = np.rad2deg(q_log[kept_mask, j].min()), np.rad2deg(
            q_log[kept_mask, j].max()
        )
        lim_min, lim_max = np.rad2deg(arm_low[j]), np.rad2deg(arm_high[j])
        flag = (
            " <-- HITS LIMIT"
            if (used_min <= lim_min + 1e-3 or used_max >= lim_max - 1e-3)
            else ""
        )
        print(
            f"  J{j + 1}: used [{used_min:7.2f}, {used_max:7.2f}] deg, "
            f"limit [{lim_min:7.2f}, {lim_max:7.2f}] deg{flag}"
        )
    if n_converged < n_kept:
        bad_idxes = np.nonzero((~converged_log) & kept_mask)[0]
        print(
            f"[CheckUmiFairino5Reachability] First unreachable step: {bad_idxes[0]} "
            f"(t={time_seq[bad_idxes[0]] - time_seq[0]:.2f}s, "
            f"residual={err_log[bad_idxes[0]]:.4f})"
        )

    if args.plot is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(6, 1, figsize=(10, 12), sharex=True)
        rel_time = time_seq - time_seq[0]
        for j in range(6):
            axes[j].plot(rel_time, np.rad2deg(q_log[:, j]), label=f"J{j + 1}")
            axes[j].axhline(np.rad2deg(arm_low[j]), color="r", linestyle="--")
            axes[j].axhline(np.rad2deg(arm_high[j]), color="r", linestyle="--")
            axes[j].set_ylabel(f"J{j + 1} [deg]")
        axes[-1].set_xlabel("time [s]")
        fig.suptitle("Retargeted FR5 joint trajectory vs. limits (dashed red)")
        fig.tight_layout()
        fig.savefig(args.plot)
        print(f"[CheckUmiFairino5Reachability] Saved plot to {args.plot}")

        # 3D pose-trajectory comparison: recorded UMI motion (as delta from
        # its first frame) vs. the retargeted FR5 target motion (as delta
        # from init_se3). This makes a rotation-direction mismatch between
        # the two directly visible, rather than only inferred from the
        # joint-angle plot above.
        umi_positions = np.array(
            [se3.translation - umi_se3_0.translation for se3 in umi_se3_list]
        )
        target_positions = np.array(
            [se3.translation - init_se3.translation for se3 in target_se3_list]
        )

        fig3d = plt.figure(figsize=(8, 8))
        ax3d = fig3d.add_subplot(projection="3d")
        ax3d.plot(*umi_positions.T, color="tab:blue", label="recorded UMI (delta)")
        ax3d.plot(
            *target_positions.T, color="tab:orange", label="retargeted FR5 (delta)"
        )

        triad_length = 0.15 * max(np.ptp(umi_positions, axis=0).max(), 1e-3)
        triad_stride = max(1, n_steps // 15)
        for t in range(0, n_steps, triad_stride):
            plot_pose_triad(
                ax3d,
                umi_positions[t],
                umi_se3_0.rotation.T @ umi_se3_list[t].rotation,
                triad_length,
            )
            plot_pose_triad(
                ax3d,
                target_positions[t],
                init_se3.rotation.T @ target_se3_list[t].rotation,
                triad_length,
            )

        ax3d.set_xlabel("x [m]")
        ax3d.set_ylabel("y [m]")
        ax3d.set_zlabel("z [m]")
        ax3d.set_title(
            "Recorded vs. retargeted EEF pose trajectory (delta from start, RGB = xyz)"
        )
        ax3d.legend()
        fig3d.tight_layout()
        base, ext = os.path.splitext(args.plot)
        pose3d_path = f"{base}_pose3d{ext}"
        fig3d.savefig(pose3d_path)
        print(
            f"[CheckUmiFairino5Reachability] Saved 3D pose comparison plot to {pose3d_path}"
        )


if __name__ == "__main__":
    main()
