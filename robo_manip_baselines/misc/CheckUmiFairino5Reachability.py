"""Offline check of whether a UMI-collected demonstration (RealUMIDemoEnv, a
robot-less handheld rig -- see envs/real/umi/RealUMIEnvBase.py) is retargetable
onto the real FR5 arm (RealFairino5DemoEnv), without moving any hardware.

RETARGETING: this MUST stay identical to misc/ReplayUmiOnFairino5.py's, or
the check stops predicting the script it exists to predict. Both rotation and
translation are pure EEF/TCP-LOCAL reproductions -- "do the same relative
motion, on FR5's own gripper, starting from FR5's own init pose" -- which is
inherently independent of how the recording room relates to the FR5's base,
so no vive_world_to_base_frame_rotation (room-vs-base) calibration is used
anywhere here. See ReplayUmiOnFairino5.py's docstring for the full
derivation, and for the earlier, wrong versions (a room-frame rotation
composition that scrambled axes, and a batch translation delta that broke as
soon as the demo also rotated) that this replaced.

--vive_config / --pos_scale / --pose_key: pass the SAME values you plan to
pass to ReplayUmiOnFairino5.py. Only pos_scale is read from the config.

This sanity-checks that the *scale* of motion in the recorded demo is
plausible for the FR5 to execute -- i.e. it stays within the arm's reach and
joint limits and does not run through a singularity -- before any hardware
moves.

Manipulability check: beyond hard joint-limit/reachability failures, this
also flags segments with LOW manipulability (small Jacobian minimum singular
value) even when IK converges and limits are respected. This matters because
low manipulability means a modest EEF-space motion demands disproportionately
large joint motion -- confirmed on real hardware to be a distinct failure
mode from "this segment was recorded fast": a segment with unremarkable
recorded EEF speed still required 25-36 deg/s joint swings the arm couldn't
track in real time, well past --time_scale slowdown in
ReplayUmiOnFairino5.py. This check exists to catch that BEFORE moving real
hardware, not just after.

Note on rotation: the recorded pose already has the UMI rig's
vive_to_eef_frame_rotation conjugation baked in by
ViveInputDevice.set_command_data at recording time (RealUMIEnvBase._set_action
echoes the command straight back as "measured", since the rig has no physical
plant to converge). So the rotation delta computed here between recorded
frames is used as-is -- it must NOT be conjugated by
vive_to_eef_frame_rotation again, which would apply that rotation twice.
"""

import argparse
import os

import numpy as np
import pinocchio as pin
import yaml

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
        "--pose_key",
        type=str,
        default=DataKey.COMMAND_EEF_POSE,
        choices=[DataKey.COMMAND_EEF_POSE, DataKey.MEASURED_EEF_POSE],
        help="which recorded EEF pose sequence to retarget. Must match "
        "ReplayUmiOnFairino5.py's --pose_key (same default) for this check "
        "to predict that script -- see its help for why command_eef_pose is "
        "the right one.",
    )
    parser.add_argument(
        "--vive_config",
        type=str,
        default=None,
        help="teleop/configs/*.yaml used to record this demo (e.g. "
        "teleop/configs/ViveUMI.yaml) -- only its pos_scale is used, as the "
        "--pos_scale default if --pos_scale is not passed explicitly. Pass "
        "the SAME file (and the same --pos_scale/--pose_key) you plan to "
        "pass to ReplayUmiOnFairino5.py for this check to actually predict "
        "that script's behavior. See the module docstring for why "
        "vive_world_to_base_frame_rotation is not used by the retargeting.",
    )
    parser.add_argument(
        "--pos_scale",
        type=float,
        default=None,
        help="scale applied to the UMI translation delta before retargeting "
        "(default: the --vive_config's own pos_scale if given, else 1.0)",
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
    parser.add_argument(
        "--warmup_seconds",
        type=float,
        default=8.0,
        help="[s] drop this much from the START of the recording, treating "
        "it as the Vive tracker's multi-lighthouse pose-solver convergence "
        "transient rather than real motion (see trim_warmup() in "
        "misc/ReplayUmiOnFairino5.py -- same logic, kept in sync). Set <= 0 "
        "to disable.",
    )
    parser.add_argument(
        "--drift_linear_vel",
        type=float,
        default=0.01,
        help="[m/s] below this frame-to-frame speed, motion is treated as "
        "sensor drift rather than intentional movement and held instead of "
        "followed (see filter_drift() in misc/ReplayUmiOnFairino5.py -- "
        "same filter, kept in sync between both scripts). Set <= 0 to "
        "disable.",
    )
    parser.add_argument(
        "--drift_angular_vel",
        type=float,
        default=3.0,
        help="[deg/s] same as --drift_linear_vel but for rotation",
    )
    parser.add_argument(
        "--drift_confirm_frames",
        type=int,
        default=2,
        help="require this many CONSECUTIVE frames above "
        "--drift_linear_vel/--drift_angular_vel before treating it as real "
        "motion (debounce against a single noisy/glitchy fast frame)",
    )
    parser.add_argument(
        "--min_manipulability",
        type=float,
        default=0.16,
        help="Jacobian minimum-singular-value threshold below which a kept "
        "frame is flagged as low-manipulability (see module docstring's "
        "manipulability check note). The real segment that caused tracking "
        "trouble on hardware had a minimum singular value of 0.147-0.151 "
        "against a ~0.18-0.19 baseline elsewhere in that same demo (verified "
        "by re-running this exact check offline against that demo file after "
        "the fact) -- this default is deliberately a bit above that observed "
        "failure value, but re-check this number against your own arm/demo "
        "rather than trusting it blindly.",
    )
    return parser.parse_args()


def trim_warmup(se3_list, time_seq, warmup_seconds=8.0):
    """Drop the first warmup_seconds of a recording, on the theory that it's
    the Vive tracker's multi-lighthouse pose-solver convergence transient,
    not real operator motion. See
    misc/ReplayUmiOnFairino5.py's copy of this function (same logic, plus a
    gripper_list argument) for the full explanation and the measurement
    (misc/MeasureViveDrift.py) this is grounded in -- kept in sync between
    both scripts, since this check should reflect what
    ReplayUmiOnFairino5.py will actually do.

    Returns (se3_list, time_seq) starting from the first frame at or after
    warmup_seconds. If the whole recording is shorter than warmup_seconds,
    returns everything UNCHANGED instead of trimming to nothing."""
    if warmup_seconds <= 0:
        return list(se3_list), list(time_seq)
    t0 = time_seq[0]
    start_idx = None
    for i, t in enumerate(time_seq):
        if t - t0 >= warmup_seconds:
            start_idx = i
            break
    if start_idx is None:
        return list(se3_list), list(time_seq)
    return list(se3_list[start_idx:]), list(time_seq[start_idx:])


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


def filter_drift(
    se3_list, time_seq, max_linear_vel=0.01, max_angular_vel_deg=3.0, confirm_frames=2
):
    """Suppress slow sensor drift (Vive tracker IMU/lighthouse drift that
    accumulates over many seconds even when the operator believes the
    tracker is stationary). See misc/ReplayUmiOnFairino5.py's copy of this
    function for the full explanation, real-data confirmation, and known
    limitation (mitigates, does not eliminate, bursty drift) -- kept in sync
    between both scripts, since this check should reflect what
    ReplayUmiOnFairino5.py will actually do."""
    max_angular_vel = np.deg2rad(max_angular_vel_deg)
    held_se3 = se3_list[0]
    filtered_se3_list = [held_se3]
    streak = 0
    for t in range(1, len(se3_list)):
        dt = max(time_seq[t] - time_seq[t - 1], 1e-6)
        linear_vel = (
            np.linalg.norm(se3_list[t].translation - se3_list[t - 1].translation) / dt
        )
        angular_vel = (
            np.linalg.norm(
                pin.log3(se3_list[t - 1].rotation.T @ se3_list[t].rotation)
            )
            / dt
        )
        if linear_vel > max_linear_vel or angular_vel > max_angular_vel:
            streak += 1
        else:
            streak = 0
        if streak >= confirm_frames:
            held_se3 = se3_list[t]
        filtered_se3_list.append(held_se3)
    return filtered_se3_list


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

    # This check only means something if it retargets EXACTLY the way
    # misc/ReplayUmiOnFairino5.py does -- see that module's docstring for the
    # derivation. Keep the two in sync. In particular
    # vive_world_to_base_frame_rotation (room-vs-base placement) is NOT read:
    # both channels are pure EEF/TCP-local reproductions, which makes
    # retargeting independent of how the room relates to the robot's base.
    # vive_to_eef_frame_rotation is not reapplied either -- it is already
    # baked into the recording (see the docstring's rotation note), and
    # applying it again would double it. So --vive_config is read only for
    # its pos_scale.
    config_pos_scale = None
    if args.vive_config is not None:
        print(f"[CheckUmiFairino5Reachability] Load {args.vive_config}")
        with open(args.vive_config, "r") as f:
            vive_config = yaml.safe_load(f)
        config_pos_scale = vive_config.get("pos_scale")
    pos_scale = args.pos_scale
    if pos_scale is None:
        pos_scale = config_pos_scale if config_pos_scale is not None else 1.0
    print(f"[CheckUmiFairino5Reachability] Using pos_scale={pos_scale}")

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
        umi_pose = rmb_data[args.pose_key][:]  # (T, 7): tx,ty,tz,qw,qx,qy,qz
        time_seq = rmb_data[DataKey.TIME][:]

    n_steps = umi_pose.shape[0]
    umi_se3_list = [
        pin.SE3(pin.Quaternion(*umi_pose[t, 3:7]), umi_pose[t, 0:3])
        for t in range(n_steps)
    ]

    if args.warmup_seconds > 0:
        trimmed_se3_list, trimmed_time_seq = trim_warmup(
            umi_se3_list, time_seq, warmup_seconds=args.warmup_seconds
        )
        if len(trimmed_se3_list) == len(umi_se3_list):
            print(
                f"[CheckUmiFairino5Reachability] WARNING: recording is "
                f"shorter than --warmup_seconds={args.warmup_seconds}s -- "
                "nothing trimmed. This recording may be entirely pose-"
                "solver convergence transient rather than real motion "
                "(see trim_warmup())."
            )
        else:
            print(
                f"[CheckUmiFairino5Reachability] Trimmed first "
                f"{args.warmup_seconds}s ({n_steps - len(trimmed_se3_list)}/"
                f"{n_steps} frames) as Vive pose-solver warmup"
            )
        # back to ndarray: downstream code (e.g. the --plot block's
        # `time_seq - time_seq[0]`) does vectorized numpy ops on the whole
        # array, which a plain list doesn't support.
        umi_se3_list, time_seq = trimmed_se3_list, np.array(trimmed_time_seq)
        n_steps = len(umi_se3_list)

    umi_se3_list, skipped_mask = filter_glitches(
        umi_se3_list, time_seq, args.max_linear_vel, args.max_angular_vel
    )
    n_skipped = int(skipped_mask.sum())
    print(
        f"[CheckUmiFairino5Reachability] Skipped {n_skipped}/{n_steps} glitch "
        f"frames (> {args.max_linear_vel} m/s or {args.max_angular_vel} deg/s)"
    )

    if args.drift_linear_vel > 0:
        pre_drift_range = np.ptp(
            np.array([se3.translation for se3 in umi_se3_list]), axis=0
        )
        umi_se3_list = filter_drift(
            umi_se3_list,
            time_seq,
            max_linear_vel=args.drift_linear_vel,
            max_angular_vel_deg=args.drift_angular_vel,
            confirm_frames=args.drift_confirm_frames,
        )
        post_drift_range = np.ptp(
            np.array([se3.translation for se3 in umi_se3_list]), axis=0
        )
        print(
            f"[CheckUmiFairino5Reachability] Drift filter (< "
            f"{args.drift_linear_vel} m/s or {args.drift_angular_vel} deg/s "
            f"for {args.drift_confirm_frames} consecutive frames = held): "
            f"position range {np.round(pre_drift_range, 4)} -> "
            f"{np.round(post_drift_range, 4)} m"
        )

    umi_se3_0 = umi_se3_list[0]

    q_log = np.zeros((n_steps, 6))
    err_log = np.zeros(n_steps)
    converged_log = np.zeros(n_steps, dtype=bool)
    limit_violation_log = np.zeros(n_steps, dtype=bool)
    min_singular_value_log = np.zeros(n_steps)
    target_se3_list = []

    # Running state for the translation accumulation below.
    fr5_translation = init_se3.translation.copy()
    prev_umi_se3 = umi_se3_0

    for t in range(n_steps):
        umi_se3_t = umi_se3_list[t]
        # Retarget exactly as misc/ReplayUmiOnFairino5.py does -- both
        # channels are pure EEF/TCP-LOCAL reproductions, so no
        # vive_world_to_base_frame_rotation is involved. See that module's
        # docstring for the derivation and the wrong versions this replaced;
        # if that math changes, change it here too or this check stops
        # predicting the replay it exists to predict.
        #
        # Rotation: delta_umi_rotation is already an EEF-local delta
        # (ViveInputDevice.set_command_data composed it via right-multiply
        # onto eef_se3_at_enable.rotation), so it is reapplied onto FR5's own
        # init orientation the same way -- RIGHT-multiply.
        delta_umi_rotation = umi_se3_0.rotation.T @ umi_se3_t.rotation
        target_rotation = init_se3.rotation @ delta_umi_rotation

        # Translation: the recorded position is an ACCUMULATION of per-frame
        # EEF-local increments, each re-projected by that frame's own
        # orientation, so it cannot be retargeted as one batch delta from
        # t=0. Recover this frame's local increment with THIS frame's own
        # rotation, then re-accumulate it through FR5's own (already
        # retargeted) orientation.
        raw_translation_delta = umi_se3_t.translation - prev_umi_se3.translation
        translation_delta_eef_local = umi_se3_t.rotation.T @ raw_translation_delta
        fr5_translation = fr5_translation + pos_scale * (
            target_rotation @ translation_delta_eef_local
        )
        prev_umi_se3 = umi_se3_t

        target_se3 = pin.SE3(target_rotation, fr5_translation.copy())
        target_se3_list.append(target_se3)

        q_arm, err_norm, _n_iters = solve_ik(model, data, q_arm, target_se3, EEF_JOINT_ID)

        q_log[t] = q_arm
        err_log[t] = err_norm
        converged_log[t] = err_norm < IK_EPS
        limit_violation_log[t] = np.any(q_arm < arm_low) | np.any(q_arm > arm_high)

        # Jacobian minimum singular value at the converged solution -- see
        # module docstring's manipulability check note. A small value means a
        # modest EEF-space motion through this configuration would demand
        # disproportionately large joint motion, independent of whether IK
        # converges or joint limits are respected.
        J = pin.computeJointJacobian(model, data, q_arm, EEF_JOINT_ID)
        min_singular_value_log[t] = np.linalg.svd(J, compute_uv=False).min()

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
    low_manip_mask = kept_mask & (min_singular_value_log < args.min_manipulability)
    n_low_manip = int(low_manip_mask.sum())
    print(
        f"[CheckUmiFairino5Reachability] Low-manipulability frames (Jacobian "
        f"min singular value < {args.min_manipulability}) among kept frames: "
        f"{n_low_manip}/{n_kept} ({100.0 * n_low_manip / n_kept:.1f}%), "
        f"min={min_singular_value_log[kept_mask].min():.3f}, "
        f"mean={min_singular_value_log[kept_mask].mean():.3f}"
    )
    if n_low_manip > 0:
        low_manip_idxes = np.nonzero(low_manip_mask)[0]
        # Report contiguous runs, not every individual flagged frame, so a
        # single problem segment doesn't flood the output with hundreds of
        # near-duplicate lines.
        run_starts = [low_manip_idxes[0]]
        run_ends = []
        for i in range(1, len(low_manip_idxes)):
            if low_manip_idxes[i] != low_manip_idxes[i - 1] + 1:
                run_ends.append(low_manip_idxes[i - 1])
                run_starts.append(low_manip_idxes[i])
        run_ends.append(low_manip_idxes[-1])
        print(
            "[CheckUmiFairino5Reachability] Low-manipulability segment(s) -- "
            "expect the real arm to lag/overshoot the Vive-driven target here "
            "even with a high --time_scale in ReplayUmiOnFairino5.py, since "
            "this is a geometric (Jacobian conditioning) issue, not just a "
            "recorded-speed issue:"
        )
        for start, end in zip(run_starts, run_ends):
            print(
                f"  t={time_seq[start] - time_seq[0]:.2f}-"
                f"{time_seq[end] - time_seq[0]:.2f}s (steps {start}-{end}), "
                f"min singular value={min_singular_value_log[start:end + 1].min():.3f}"
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
