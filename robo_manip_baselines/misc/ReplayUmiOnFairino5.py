"""Replay a UMI-collected demonstration (RealUMIDemoEnv, a robot-less handheld
rig -- see envs/real/umi/RealUMIEnvBase.py) onto the FR5 arm
(RealFairino5DemoEnv), by retargeting the recorded EEF trajectory into a
sparse sequence of CHECKPOINT poses and driving the arm to each one in turn
with closed-loop PI control.

Unlike bin/Teleop.py's built-in --replay_log (which replays a log recorded by
the SAME env, using its own DOF/frame conventions unchanged -- see
teleop/TeleopBase.py's ReplayPhase), this script retargets across envs: the
UMI rig's recorded EEF pose has no calibrated common frame with the FR5 base
UNLESS --vive_config is given (see below). Without it, this falls back to the
naive delta-pose retarget also used by misc/CheckUmiFairino5Reachability.py
(translation delta added onto the FR5's init-pose EEF position with an
identity room-to-base rotation; rotation delta used as-is, see this
docstring's rotation note below).

CONTROL STRATEGY -- checkpoint tracking, not feedforward interpolation:
an earlier version of this script retargeted EVERY recorded UMI frame,
densely interpolated between them, and drove one feedforward damped-least-
squares IK step per interpolated pose, warm-started from its OWN internal
kinematic state (never re-checking the real arm's actual measured position).
That is open-loop: if the arm fell behind at any single step (e.g. a
low-manipulability/near-singular segment), nothing corrected it, and error
could accumulate for the rest of the replay.

This version instead (see sample_checkpoints/track_checkpoint):
  1. Downsamples the recorded trajectory to a sparse sequence of CHECKPOINT
     poses (~--checkpoint_hz, e.g. 10 Hz) -- A -> B -> C -> ... -- each one a
     discrete waypoint to reach in turn, not a point on a path to track
     smoothly.
  2. For each checkpoint, runs closed-loop PI control: every iteration reads
     the arm's REAL measured EEF pose (via env's obs, not an assumed internal
     state), computes the pose error against the checkpoint, and commands a
     small clamped step toward it. If a checkpoint sits near a singularity,
     this is allowed to take many iterations (--max_iters_per_checkpoint) --
     it does NOT need to reproduce the original Vive path between checkpoints
     to get there, only to eventually arrive at the checkpoint itself. TCP
     motion smoothness is intentionally traded away for this -- exact-enough
     arrival at each checkpoint is the priority, not a smooth path between
     them.
  3. Once within --pos_tol/--rot_tol_deg of a checkpoint (or
     --max_iters_per_checkpoint is exhausted -- logged as a warning, then
     moves on regardless rather than hanging), proceeds to the next
     checkpoint.

--low_manip_threshold/--max_extra_damping/--max_joint_step_deg tune the
per-iteration IK step's (adaptive_ik_step) response to low-manipulability
configurations. Both default to 0 (disabled): exact TCP tracking is
prioritized over motion smoothness for this task, so by default this reduces
to ArmManager's own plain fixed-damping IK step -- see that function's
docstring if smoothing is ever needed instead.

--vive_config: pass the SAME teleop/configs/*.yaml used to record this demo
(e.g. teleop/configs/ViveUMI.yaml) to use its calibrated
vive_world_to_base_frame_rotation for the translation delta, exactly as live
teleop's ViveInputDevice.set_command_data does:
    target_translation = init_se3.translation + pos_scale * (
        vive_world_to_base_frame_rotation @ (umi_se3_t.translation - umi_se3_0.translation)
    )
(left-multiply, not transposed -- see ViveInputDevice.py's set_command_data).
Without --vive_config this matrix defaults to identity (the old naive
behavior). vive_to_eef_frame_rotation from that same config is NOT reapplied
here -- see the rotation note below for why. If --pos_scale is not passed on
the command line, the config's own pos_scale is used as the default (falling
back to 1.0 if the config has none either), matching how it was recorded.

SAFETY: defaults to dry_run (no hardware connection; env prints the ServoJ
command it would have sent). Pass --real only once you have verified the
printed commands look sane (e.g. via CheckUmiFairino5Reachability.py and a
dry-run read-through of this script's own printed output) and are ready to
move the physical arm.

Note on rotation: the recorded MEASURED_EEF_POSE already has the UMI rig's
vive_to_eef_frame_rotation conjugation baked in by ViveInputDevice.set_command_data
at recording time (RealUMIEnvBase._set_action echoes the command straight back
as "measured", since the rig has no physical plant to converge). So the
rotation delta computed here between recorded frames is used as-is -- it must
NOT be conjugated by vive_to_eef_frame_rotation again, which would apply that
rotation twice. It IS applied to init_se3.rotation by LEFT-multiply (world-
frame composition), not right-multiply (body-frame composition) -- see the
retarget loop's own comment in main() for why (umi.urdf's floating joint has
zero rotation offset from world_link, so delta_umi_rotation is effectively
already a world-frame quantity; right-multiplying it silently reinterprets
each world-frame axis through FR5's own unrelated initial orientation,
confirmed on real hardware to scramble roll/pitch/yaw -- a demo recorded with
almost no roll/pitch showed the FR5 target rolling/pitching heavily under the
old right-multiply code).
"""

import argparse
import os
import time

import gymnasium as gym
import numpy as np
import pinocchio as pin
import yaml

from robo_manip_baselines.common import (
    ArmManager,
    DataKey,
    RmbData,
    get_pose_from_se3,
    get_se3_from_pose,
)

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
        "--vive_config",
        type=str,
        default=None,
        help="teleop/configs/*.yaml used to record this demo (e.g. "
        "teleop/configs/ViveUMI.yaml) -- if given, its calibrated "
        "vive_world_to_base_frame_rotation is used for the translation "
        "retarget instead of the identity-rotation naive fallback (see "
        "module docstring). Its pos_scale is also used as the --pos_scale "
        "default if --pos_scale is not passed explicitly.",
    )
    parser.add_argument(
        "--pos_scale",
        type=float,
        default=None,
        help="scale applied to the UMI translation delta before retargeting "
        "(default: the --vive_config's own pos_scale if given, else 1.0)",
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
        "--warmup_seconds",
        type=float,
        default=8.0,
        help="[s] drop this much from the START of the recording, treating "
        "it as the Vive tracker's multi-lighthouse pose-solver convergence "
        "transient rather than real motion (see trim_warmup() -- grounded "
        "in a direct measurement showing ~5-10s of convergence after a "
        "stationary tracker was first acquired). Set <= 0 to disable. If "
        "the whole recording is shorter than this, nothing is trimmed and "
        "a warning is printed instead -- that recording may be entirely "
        "convergence transient.",
    )
    parser.add_argument(
        "--drift_linear_vel",
        type=float,
        default=0.01,
        help="[m/s] below this frame-to-frame speed, motion is treated as "
        "sensor drift rather than intentional movement and held instead of "
        "followed (see filter_drift() -- mitigates but does not eliminate "
        "drift; verified to NOT affect a real demo's recorded motion at "
        "this default). Set <= 0 to disable this filter entirely.",
    )
    parser.add_argument(
        "--drift_angular_vel",
        type=float,
        default=3.0,
        help="[deg/s] same as --drift_linear_vel but for rotation, see "
        "filter_drift()",
    )
    parser.add_argument(
        "--drift_confirm_frames",
        type=int,
        default=2,
        help="require this many CONSECUTIVE frames above "
        "--drift_linear_vel/--drift_angular_vel before treating it as real "
        "motion (debounce against a single noisy/glitchy fast frame), see "
        "filter_drift()",
    )
    parser.add_argument(
        "--checkpoint_hz",
        type=float,
        default=10.0,
        help="downsample the recorded UMI trajectory to a checkpoint every "
        "~1/checkpoint_hz seconds (always keeping the first and last "
        "frame). Each checkpoint is tracked to convergence in turn by "
        "closed-loop PI control (see track_checkpoint) -- this is NOT a "
        "smoothing/interpolation rate like the old --time_scale, it's the "
        "spacing of discrete waypoints the arm must actually arrive at.",
    )
    parser.add_argument(
        "--pos_tol",
        type=float,
        default=0.003,
        help="[m] a checkpoint counts as reached once position error drops "
        "below this",
    )
    parser.add_argument(
        "--rot_tol_deg",
        type=float,
        default=1.0,
        help="[deg] a checkpoint counts as reached once orientation error "
        "drops below this",
    )
    parser.add_argument(
        "--kp_pos", type=float, default=0.5, help="PI controller position P gain"
    )
    parser.add_argument(
        "--ki_pos",
        type=float,
        default=0.1,
        help="PI controller position I gain (integral accumulates the raw "
        "position error each iteration, clamped by --integral_clamp_pos to "
        "prevent windup on an unreachable checkpoint)",
    )
    parser.add_argument(
        "--kp_rot", type=float, default=0.5, help="PI controller orientation P gain"
    )
    parser.add_argument(
        "--ki_rot",
        type=float,
        default=0.1,
        help="PI controller orientation I gain (see --ki_pos; clamped by "
        "--integral_clamp_rot_deg)",
    )
    parser.add_argument(
        "--max_step_pos",
        type=float,
        default=0.005,
        help="[m] hard cap on the PI controller's commanded position step "
        "per control iteration, independent of the P/I gains -- this is "
        "what makes it safe to run a checkpoint for many iterations near a "
        "singularity: each individual iteration's motion stays bounded "
        "regardless of how large the instantaneous error or integral term "
        "gets.",
    )
    parser.add_argument(
        "--max_step_rot_deg",
        type=float,
        default=2.0,
        help="[deg] hard cap on the PI controller's commanded orientation "
        "step per control iteration -- see --max_step_pos.",
    )
    parser.add_argument(
        "--max_iters_per_checkpoint",
        type=int,
        default=300,
        help="give up on a checkpoint (log a warning, keep the arm where it "
        "is, and move on to the next checkpoint) after this many control "
        "iterations without reaching --pos_tol/--rot_tol_deg. At env.dt="
        "0.02s nominal, 300 iterations is up to ~6s per checkpoint -- "
        "generous on purpose, since a low-manipulability checkpoint is "
        "explicitly allowed to take longer (see module docstring) rather "
        "than being rushed and left inaccurate.",
    )
    parser.add_argument(
        "--integral_clamp_pos",
        type=float,
        default=0.02,
        help="[m] anti-windup bound on the accumulated position integral "
        "term -- without this, a checkpoint that turns out to be "
        "unreachable (e.g. past a joint limit) would make --ki_pos's "
        "integral term grow without bound, commanding ever-larger steps "
        "against a limit it can never actually close.",
    )
    parser.add_argument(
        "--integral_clamp_rot_deg",
        type=float,
        default=10.0,
        help="[deg] anti-windup bound on the accumulated orientation "
        "integral term, see --integral_clamp_pos.",
    )
    parser.add_argument(
        "--compare_plot",
        type=str,
        default=None,
        help="path to save a plot comparing the recorded Vive/UMI-retargeted "
        "path against the FR5's actual measured path during this replay "
        "(png). If omitted, saved next to rmb_filename as "
        "'<rmb_basename>_vive_vs_robot.png'. Pass an empty string to skip "
        "saving. NOTE: with dry_run (the default, no --real), the FR5's "
        "'measured' pose is just an echo of the commanded pose (see "
        "RealFairino5EnvBase._get_obs), so the two paths will match "
        "trivially -- this comparison is only meaningful with --real, where "
        "the measured path comes from the arm's actual encoders.",
    )
    parser.add_argument(
        "--log_csv",
        type=str,
        default=None,
        help="path to save a per-step CSV log of this replay (time, the 6 "
        "commanded arm joint angles [deg] sent to the FR5 -- i.e. the IK "
        "solution driven by the Vive-retargeted target -- the 6 measured "
        "joint angles [deg] read back from the arm, and the commanded vs. "
        "measured EEF pose [tx,ty,tz,qw,qx,qy,qz]). If omitted, saved next "
        "to rmb_filename as '<rmb_basename>_replay_log.csv'. Pass an empty "
        "string to skip saving.",
    )
    parser.add_argument(
        "--low_manip_threshold",
        type=float,
        default=0.16,
        help="Jacobian minimum-singular-value threshold below which "
        "adaptive_ik_step's damping is increased (see that function's "
        "docstring). Same default as CheckUmiFairino5Reachability.py's "
        "--min_manipulability -- run that script with --vive_config first "
        "to see whether/where this demo has segments below this value "
        "before replaying on --real.",
    )
    parser.add_argument(
        "--max_extra_damping",
        type=float,
        default=0.0,
        help="Extra IK damping applied at zero manipulability (ramps down "
        "to 0 at --low_manip_threshold, see adaptive_ik_step). Higher "
        "values favor smoother/slower motion over exact tracking through "
        "low-manipulability segments -- defaults to 0 (disabled), which "
        "falls back to ArmManager's own exact-tracking fixed-damping IK "
        "behavior, since precise TCP tracking matters more than smoothness "
        "for this task. Only raise this if you've confirmed (e.g. via large "
        "single-step joint jumps in --log_csv) that a specific demo's "
        "low-manipulability segment needs it, and are OK trading tracking "
        "accuracy there for smoother motion.",
    )
    parser.add_argument(
        "--max_joint_step_deg",
        type=float,
        default=0.0,
        help="Hard per-step joint motion cap [deg] in adaptive_ik_step, on "
        "top of damping -- a second, unconditional layer under the env's "
        "own overwrite_command_for_safety velocity clamp. Defaults to 0 "
        "(disabled) for the same reason as --max_extra_damping: it trades "
        "tracking accuracy for a bounded step size. Set to a positive "
        "value (e.g. 1.0) to enable it.",
    )
    return parser.parse_args()


def trim_warmup(se3_list, gripper_list, time_seq, warmup_seconds=8.0):
    """Drop the first warmup_seconds of a recording, on the theory that it's
    the Vive tracker's multi-lighthouse pose-solver convergence transient,
    not real operator motion.

    Grounded in a direct measurement (misc/MeasureViveDrift.py) of a
    stationary tracker's raw pose: position climbed ~2cm and orientation
    ~7deg over roughly the first 5-10s after the tracker was first acquired
    (visible in libsurvive's own log as successive "Global solve with N
    scenes" refinements), then stayed flat within ~1-2mm/~1deg noise for the
    rest of a 60s hold. teleop/ViveInputDevice.py's MIN_ANCHOR_DELAY now
    waits this long before anchoring teleop for NEWLY recorded demos, but
    older recordings (or any from a build predating that fix) may still
    have this transient baked into their first several seconds -- this
    trims it defensively at replay/check time regardless of when the demo
    was recorded.

    Returns (se3_list, gripper_list, time_seq) starting from the first
    frame at or after warmup_seconds (re-indexed so downstream code treats
    that as frame 0). If the whole recording is shorter than
    warmup_seconds, returns everything UNCHANGED instead of trimming to
    nothing -- the caller should treat that case as a warning sign (the
    entire recording may be convergence transient, not real motion; compare
    against --checkpoint_hz's sampled range or just re-record with the demo
    starting later)."""
    if warmup_seconds <= 0:
        return list(se3_list), list(gripper_list), list(time_seq)
    t0 = time_seq[0]
    start_idx = None
    for i, t in enumerate(time_seq):
        if t - t0 >= warmup_seconds:
            start_idx = i
            break
    if start_idx is None:
        return list(se3_list), list(gripper_list), list(time_seq)
    return (
        list(se3_list[start_idx:]),
        list(gripper_list[start_idx:]),
        list(time_seq[start_idx:]),
    )


def interpolate_se3(se3_a, se3_b, alpha):
    """Screw-motion interpolation between two poses (alpha=0 -> se3_a,
    alpha=1 -> se3_b), via the same pin.log6/pin.integrate-family machinery
    ArmManager's IK uses elsewhere in this codebase."""
    return se3_a * pin.exp6(alpha * pin.log6(se3_a.inverse() * se3_b))


def sample_checkpoints(se3_list, gripper_list, time_seq, checkpoint_hz):
    """Resample a recorded trajectory to a sequence of checkpoint poses at
    (approximately) checkpoint_hz, evenly spaced in time from the first to
    the last recorded frame. Each checkpoint is a discrete waypoint the arm
    must reach in turn (A -> B -> C -> ...) via closed-loop PI control (see
    track_checkpoint) -- NOT a point on a path to be tracked smoothly.

    This INTERPOLATES (screw motion, see interpolate_se3) between the two
    nearest recorded frames for each checkpoint time, rather than just
    picking the nearest existing frame -- confirmed necessary on real data:
    a demo recorded at the UMI rig's native ~10Hz was replayed with
    --checkpoint_hz 40 expecting 4x denser checkpoints (to shrink the gap
    between consecutive checkpoints, since the real arm's per-checkpoint
    convergence was falling badly behind during faster-recorded segments --
    see the on-hardware analysis this fix followed from), but a
    nearest-frame-only version of this function silently produced the exact
    same checkpoints as --checkpoint_hz 10 or 20, because there was no
    recorded data any denser than ~10Hz to select from. Interpolating lets
    --checkpoint_hz exceed the recording's native rate and actually produce
    finer checkpoints; requesting a checkpoint_hz BELOW the native rate
    still works as plain downsampling (interpolation with alpha snapped
    close to 0 or 1 in that regime)."""
    if checkpoint_hz <= 0:
        return list(se3_list), list(gripper_list), list(time_seq)
    period = 1.0 / checkpoint_hz
    n = len(time_seq)
    total_duration = time_seq[-1] - time_seq[0]
    n_checkpoints = max(1, round(total_duration / period)) + 1

    out_se3_list = []
    out_gripper_list = []
    out_time_seq = []
    seg_idx = 0  # se3_list[seg_idx] -> se3_list[seg_idx + 1] brackets t_target
    for k in range(n_checkpoints):
        t_target = min(time_seq[0] + k * period, time_seq[-1])
        while seg_idx < n - 2 and time_seq[seg_idx + 1] < t_target:
            seg_idx += 1
        t0, t1 = time_seq[seg_idx], time_seq[min(seg_idx + 1, n - 1)]
        alpha = (t_target - t0) / (t1 - t0) if t1 > t0 else 0.0
        alpha = min(max(alpha, 0.0), 1.0)
        idx_b = min(seg_idx + 1, n - 1)
        out_se3_list.append(interpolate_se3(se3_list[seg_idx], se3_list[idx_b], alpha))
        out_gripper_list.append(
            (1 - alpha) * gripper_list[seg_idx] + alpha * gripper_list[idx_b]
        )
        out_time_seq.append(t_target)

    if out_time_seq[-1] < time_seq[-1]:
        out_se3_list.append(se3_list[-1])
        out_gripper_list.append(gripper_list[-1])
        out_time_seq.append(time_seq[-1])

    return out_se3_list, out_gripper_list, out_time_seq


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


def filter_drift(
    se3_list, time_seq, max_linear_vel=0.01, max_angular_vel_deg=3.0, confirm_frames=2
):
    """Suppress slow sensor drift (Vive tracker IMU/lighthouse drift that
    accumulates over many seconds even when the operator believes the
    tracker is stationary) with a velocity-gated hold filter: a HELD
    reference pose only advances to a new frame once that frame's velocity
    relative to the PREVIOUS raw frame exceeds max_linear_vel/
    max_angular_vel_deg for confirm_frames consecutive frames (debounced, so
    a single noisy/glitchy fast frame can't falsely "confirm" motion).
    Frames that don't confirm real motion are replaced by the held pose.

    CONFIRMED ON REAL DATA, WITH A CAVEAT: a demo recorded with no
    intentional motion still showed the tracker's position climbing
    ~5cm/~5.7deg over 10.5s when replayed naively (this is what motivated
    this filter). This filter, at reasonable thresholds, verifiably does
    NOT affect real recorded motion (a genuine demo's position/rotation
    range was completely unchanged when run through it). But it only
    PARTIALLY suppresses drift that itself contains brief fast bursts (real
    tracker noise, not glitch-filter-worthy jumps) -- on the same drift-only
    recording above, this reduced total apparent drift by roughly 15%, not
    to zero, because those bursts momentarily exceed the velocity
    threshold and get "confirmed" as real motion. This is a mitigation, not
    a complete fix -- for a demo you suspect is mostly/entirely drift
    (e.g. checked via --vive_config with CheckUmiFairino5Reachability.py
    showing implausibly large motion for an intentionally-still recording),
    the more reliable fix is operational: let the tracker settle a few
    seconds before recording starts, and keep recordings reasonably short.

    UPDATE (misc/MeasureViveDrift.py measurement): the "5cm/5.7deg over
    10.5s" case above turned out to be almost entirely the pose-solver's
    OWN initial convergence transient (~5-10s after the tracker is first
    acquired), not ongoing sensor drift -- after that transient, a
    stationary tracker's pose stayed flat within ~1-2mm/~1deg noise for the
    rest of a 60s hold. trim_warmup() (see above in this file) targets that
    transient specifically and is now the PRIMARY defense, applied before
    this filter; this filter's velocity gate is a secondary safety net for
    ongoing low-level noise, not the main fix for the original observation.

    Returns a new se3_list of the same length (time_seq is unchanged)."""
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


def adaptive_ik_step(
    arm_manager,
    target_se3,
    low_manip_threshold=0.16,
    max_extra_damping=0.05,
    max_joint_step_deg=None,
):
    """One damped-least-squares Newton IK step -- same formulation as
    ArmManager.inverse_kinematics -- but with damping that grows as the
    current configuration's manipulability drops, instead of
    ArmManager.inverse_kinematics's fixed damping_scale=1e-6 floor.

    Why: for a non-redundant 6-DOF arm chasing a full 6-DOF pose target,
    there is no joint-space freedom left to "dodge" a singularity while
    still exactly tracking the target through it -- if the Vive-derived
    target trajectory passes near one, the arm fundamentally cannot both
    track exactly AND move smoothly there. ArmManager's fixed small damping
    resolves that conflict by favoring exact tracking, which is what
    produced the 25-36 deg/s single-joint swings observed on real hardware
    in exactly the segment misc/CheckUmiFairino5Reachability.py's
    --min_manipulability check flags (Jacobian min singular value ~0.14-0.15
    there vs ~0.18-0.24 baseline). Raising damping specifically when
    manipulability is low instead favors smooth, bounded joint motion over
    exact tracking FOR THAT MOMENT -- the arm falls slightly behind the
    target while passing through the difficult region, then damping drops
    back to the floor and it converges again once manipulability recovers
    (this is the "filter singular points while still reaching the target"
    behavior requested -- reaching happens over the region as a whole, not
    at every single instant inside it, which is not physically possible
    here). A hard max_joint_step_deg cap on top gives an absolute bound
    regardless of tuning, as a second layer under the env's own
    overwrite_command_for_safety velocity clamp.

    Mutates arm_manager.arm_joint_pos/target_se3 in place, matching
    ArmManager.set_command_eef_pose's own side effects, so it's a drop-in
    replacement for that call in the replay loop."""
    arm_manager.target_se3 = target_se3
    error_se3 = arm_manager.current_se3.actInv(target_se3)
    error_vec = pin.log(error_se3).vector
    q = arm_manager.arm_joint_pos
    J = pin.computeJointJacobian(
        arm_manager.pin_model, arm_manager.pin_data, q,
        arm_manager.body_config.ik_eef_joint_id,
    )
    J = -1 * np.dot(pin.Jlog6(error_se3.inverse()), J)

    min_singular_value = np.linalg.svd(J, compute_uv=False).min()
    if min_singular_value < low_manip_threshold:
        # Smooth ramp from 0 (at the threshold) up to max_extra_damping (as
        # min_singular_value -> 0), same shape as the classic
        # manipulability-based variable-damping DLS scheme (Nakamura &
        # Hanafusa).
        extra_damping = max_extra_damping * (
            1.0 - min_singular_value / low_manip_threshold
        ) ** 2
    else:
        extra_damping = 0.0
    damping_scale = 1e-6 + extra_damping  # 1e-6 floor matches ArmManager's own

    delta_arm_joint_pos = -1 * J.T.dot(
        np.linalg.solve(
            J.dot(J.T) + (np.dot(error_vec, error_vec) + damping_scale) * np.eye(6),
            error_vec,
        )
    )

    if max_joint_step_deg is not None:
        max_joint_step_rad = np.deg2rad(max_joint_step_deg)
        step_norm = np.max(np.abs(delta_arm_joint_pos))
        if step_norm > max_joint_step_rad:
            delta_arm_joint_pos = delta_arm_joint_pos * (
                max_joint_step_rad / step_norm
            )

    arm_manager.arm_joint_pos = pin.integrate(
        arm_manager.pin_model, q, delta_arm_joint_pos
    )
    arm_manager.forward_kinematics()
    return min_singular_value


def track_checkpoint(
    env,
    arm_manager,
    target_se3,
    gripper_target,
    obs,
    replay_start_time,
    pos_tol=0.003,
    rot_tol=np.deg2rad(1.0),
    kp_pos=0.5,
    ki_pos=0.1,
    kp_rot=0.5,
    ki_rot=0.1,
    max_step_pos=0.005,
    max_step_rot=np.deg2rad(2.0),
    max_iters=300,
    integral_clamp_pos=0.02,
    integral_clamp_rot=np.deg2rad(10.0),
    low_manip_threshold=0.16,
    max_extra_damping=0.0,
    max_joint_step_deg=None,
):
    """Closed-loop PI controller that drives the FR5 toward ONE checkpoint
    pose, re-reading the arm's REAL measured EEF pose (via env's obs) every
    iteration -- unlike a feedforward approach that trusts its own internal
    kinematic state, this re-anchors to the actual measured position each
    time, so it self-corrects instead of accumulating drift, and can work
    through a difficult (low-manipulability) checkpoint over many small
    iterations instead of needing one feedforward step to get exactly right.
    Does not track any particular path to the checkpoint -- only converges
    toward target_se3, taking as many iterations as max_iters allows (see
    module docstring for why this trade-off was chosen).

    Each iteration: compute the pose error (in the arm's current LOCAL
    frame) between the real measured EEF pose and target_se3; if within
    pos_tol/rot_tol, stop (checkpoint reached). Otherwise accumulate a
    clamped PI term, cap it to a small per-iteration step, turn that into a
    nearby SE3 sub-target one step ahead of where the arm actually is right
    now, re-sync arm_manager's IK state to the real measured joints, and run
    one adaptive_ik_step toward that nearby sub-target (never toward the far
    checkpoint directly -- IK only ever has to solve a small local step).

    Returns (reached: bool, n_iters, final_measured_se3, obs, log) where log
    is a dict of per-iteration lists: min_singular_value, command_joint_deg,
    measured_joint_deg, measured_se3, step_time (actual wall-clock elapsed
    since replay_start_time, for --log_csv/plot_joint_comparison -- see
    their docstrings for why nominal step_idx*dt is not used on --real)."""
    integral_pos = np.zeros(3)
    integral_rot = np.zeros(3)
    log = {
        "min_singular_value": [],
        "command_joint_deg": [],
        "measured_joint_deg": [],
        "measured_se3": [],
        "step_time": [],
    }

    measured_se3 = None
    n_iters = 0
    reached = False
    for i in range(max_iters):
        measured_joint_pos = env.unwrapped.get_joint_pos_from_obs(obs)
        measured_arm_joint_pos = measured_joint_pos[
            arm_manager.body_config.arm_joint_idxes
        ]
        measured_se3 = get_se3_from_pose(
            arm_manager.get_eef_pose_from_joint_pos(measured_arm_joint_pos)
        )

        error_se3 = measured_se3.actInv(target_se3)
        error_vec = pin.log6(error_se3).vector  # (linear, angular), local frame
        pos_err, rot_err = error_vec[:3], error_vec[3:]
        n_iters = i
        if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(rot_err) < rot_tol:
            reached = True
            break

        integral_pos = np.clip(
            integral_pos + pos_err, -integral_clamp_pos, integral_clamp_pos
        )
        integral_rot = np.clip(
            integral_rot + rot_err, -integral_clamp_rot, integral_clamp_rot
        )
        step_pos = kp_pos * pos_err + ki_pos * integral_pos
        step_rot = kp_rot * rot_err + ki_rot * integral_rot

        step_pos_norm = np.linalg.norm(step_pos)
        if step_pos_norm > max_step_pos:
            step_pos = step_pos * (max_step_pos / step_pos_norm)
        step_rot_norm = np.linalg.norm(step_rot)
        if step_rot_norm > max_step_rot:
            step_rot = step_rot * (max_step_rot / step_rot_norm)

        sub_target_se3 = measured_se3 * pin.exp6(
            np.concatenate([step_pos, step_rot])
        )

        # Re-sync IK's internal state to the REAL measured joints before
        # solving -- this closed-loop resync is what lets the controller
        # self-correct instead of drifting from an unvalidated internal
        # model, and is the key difference from the old feedforward replay
        # loop.
        arm_manager.arm_joint_pos = measured_arm_joint_pos.copy()
        arm_manager.forward_kinematics()
        min_singular_value = adaptive_ik_step(
            arm_manager,
            sub_target_se3,
            low_manip_threshold=low_manip_threshold,
            max_extra_damping=max_extra_damping,
            max_joint_step_deg=max_joint_step_deg,
        )
        arm_manager.set_command_gripper_joint_pos(gripper_target)

        command_arm_joint_pos = arm_manager.arm_joint_pos.copy()
        action = np.concatenate(
            [command_arm_joint_pos, arm_manager.gripper_joint_pos]
        )
        obs = env.step(action)[0]

        result_joint_pos = env.unwrapped.get_joint_pos_from_obs(obs)
        result_arm_joint_pos = result_joint_pos[
            arm_manager.body_config.arm_joint_idxes
        ]
        log["min_singular_value"].append(min_singular_value)
        log["command_joint_deg"].append(np.rad2deg(command_arm_joint_pos))
        log["measured_joint_deg"].append(np.rad2deg(result_arm_joint_pos))
        log["measured_se3"].append(
            get_se3_from_pose(
                arm_manager.get_eef_pose_from_joint_pos(result_arm_joint_pos)
            )
        )
        log["step_time"].append(time.time() - replay_start_time)

    return reached, n_iters, measured_se3, obs, log


def plot_pose_triad(ax, position, rotation, length, label=None):
    """Draw one RGB xyz-axis triad (quiver arrows) at `position` oriented by
    `rotation`, on a 3D `Axes3D`. Same helper as misc/VisualizeData.py and
    misc/CheckUmiFairino5Reachability.py."""
    for axis_idx, color in enumerate(("r", "g", "b")):
        ax.quiver(
            *position,
            *(length * rotation[:, axis_idx]),
            color=color,
            linewidth=1.5,
            label=label if axis_idx == 0 else None,
        )


def plot_vive_vs_robot(target_se3_list, measured_se3_list, out_path):
    """Save a plot comparing the retargeted Vive/UMI path (target_se3_list,
    what was commanded each step) against the FR5's actual measured path
    (measured_se3_list, read back from the arm's encoders via forward
    kinematics after each env.step()). Close overlap is the visual
    trust-check for whether this demo's retargeting/replay is sane; large or
    systematic gaps mean the arm is not tracking the recorded Vive motion
    (e.g. IK not converging per-step, joint-limit clamping, or the per-step
    velocity clamp in overwrite_command_for_safety lagging behind).

    Returns (max_pos_err, mean_pos_err, max_rot_err_deg, mean_rot_err_deg)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = min(len(target_se3_list), len(measured_se3_list))
    target_se3_list = target_se3_list[:n]
    measured_se3_list = measured_se3_list[:n]

    target_pos = np.array([se3.translation for se3 in target_se3_list])
    measured_pos = np.array([se3.translation for se3 in measured_se3_list])
    pos_err = np.linalg.norm(target_pos - measured_pos, axis=1)
    rot_err_deg = np.array(
        [
            np.rad2deg(
                np.linalg.norm(
                    pin.log3(target_se3.rotation.T @ measured_se3.rotation)
                )
            )
            for target_se3, measured_se3 in zip(target_se3_list, measured_se3_list)
        ]
    )

    fig = plt.figure(figsize=(14, 7))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot(*target_pos.T, color="tab:orange", linestyle="--", label="Vive-retargeted (commanded)")
    ax3d.plot(*measured_pos.T, color="tab:blue", label="FR5 actual (measured)")
    triad_length = 0.15 * max(np.ptp(target_pos, axis=0).max(), 1e-3)
    triad_stride = max(1, n // 15)
    for t in range(0, n, triad_stride):
        plot_pose_triad(ax3d, target_pos[t], target_se3_list[t].rotation, triad_length)
        plot_pose_triad(ax3d, measured_pos[t], measured_se3_list[t].rotation, triad_length)
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.set_title("Vive-retargeted vs. actual FR5 EEF path (RGB = xyz)")
    ax3d.legend()

    ax_err = fig.add_subplot(1, 2, 2)
    step_idx = np.arange(n)
    ax_err.plot(step_idx, pos_err, color="tab:blue", label="position error [m]")
    ax_err.set_xlabel("replay step")
    ax_err.set_ylabel("position error [m]", color="tab:blue")
    ax_err_twin = ax_err.twinx()
    ax_err_twin.plot(step_idx, rot_err_deg, color="tab:red", label="orientation error [deg]")
    ax_err_twin.set_ylabel("orientation error [deg]", color="tab:red")
    ax_err.set_title("Commanded-vs-measured tracking error per step")

    fig.tight_layout()
    fig.savefig(out_path)

    return pos_err.max(), pos_err.mean(), rot_err_deg.max(), rot_err_deg.mean()


def plot_joint_comparison(command_joint_deg, measured_joint_deg, rel_time, out_path):
    """Save a per-joint plot of the commanded arm joint angles (the IK
    solution driven by the Vive-retargeted target each step) against the
    joint angles actually read back from the FR5's encoders. Companion to
    plot_vive_vs_robot's EEF-space comparison, but in joint space -- useful
    for spotting which specific joint is lagging/clamped when the EEF-space
    error looks large (e.g. one joint sitting at a limit).

    rel_time must be actual measured elapsed wall-clock seconds per step
    (see step_time_list in main()), NOT step_idx * a nominal dt -- on --real,
    the true per-step cadence is jittery (XML-RPC round-trip latency), so a
    nominal-dt axis would mislabel how much real time each step took."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n, n_joints = command_joint_deg.shape
    fig, axes = plt.subplots(n_joints, 1, figsize=(10, 2 * n_joints), sharex=True)
    for j in range(n_joints):
        axes[j].plot(rel_time, command_joint_deg[:, j], linestyle="--", label="commanded")
        axes[j].plot(rel_time, measured_joint_deg[:, j], label="measured")
        axes[j].set_ylabel(f"J{j + 1} [deg]")
        if j == 0:
            axes[j].legend(fontsize=8)
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Commanded (Vive-driven IK) vs. measured FR5 joint angles")
    fig.tight_layout()
    fig.savefig(out_path)


def main():
    args = parse_argument()

    # vive_world_to_base_frame_rotation calibrates the room-to-base rotation
    # applied to the translation delta -- see ViveInputDevice.set_command_data
    # and this module's docstring. vive_to_eef_frame_rotation from the same
    # config is intentionally NOT loaded/reapplied here: it's already baked
    # into the recorded MEASURED_EEF_POSE rotation at record time (see the
    # docstring's rotation note), and reapplying it here would double it.
    vive_world_to_base_frame_rotation = np.eye(3)
    config_pos_scale = None
    if args.vive_config is not None:
        print(f"[ReplayUmiOnFairino5] Load {args.vive_config}")
        with open(args.vive_config, "r") as f:
            vive_config = yaml.safe_load(f)
        if "vive_world_to_base_frame_rotation" in vive_config:
            vive_world_to_base_frame_rotation = np.array(
                vive_config["vive_world_to_base_frame_rotation"], dtype=np.float64
            )
            assert vive_world_to_base_frame_rotation.shape == (3, 3)
        else:
            print(
                "[ReplayUmiOnFairino5] Warning: --vive_config has no "
                "vive_world_to_base_frame_rotation; falling back to identity "
                "(same as not passing --vive_config at all)."
            )
        config_pos_scale = vive_config.get("pos_scale")
    pos_scale = args.pos_scale
    if pos_scale is None:
        pos_scale = config_pos_scale if config_pos_scale is not None else 1.0
    print(f"[ReplayUmiOnFairino5] Using pos_scale={pos_scale}")

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
    umi_gripper = list(umi_gripper)

    if args.warmup_seconds > 0:
        trimmed_se3_list, trimmed_gripper, trimmed_time_seq = trim_warmup(
            umi_se3_list, umi_gripper, time_seq, warmup_seconds=args.warmup_seconds
        )
        if len(trimmed_se3_list) == len(umi_se3_list):
            print(
                f"[ReplayUmiOnFairino5] WARNING: recording is shorter than "
                f"--warmup_seconds={args.warmup_seconds}s -- nothing trimmed. "
                "This recording may be entirely pose-solver convergence "
                "transient rather than real motion (see trim_warmup())."
            )
        else:
            print(
                f"[ReplayUmiOnFairino5] Trimmed first {args.warmup_seconds}s "
                f"({n_steps - len(trimmed_se3_list)}/{n_steps} frames) as "
                "Vive pose-solver warmup"
            )
        umi_se3_list, umi_gripper, time_seq = (
            trimmed_se3_list,
            trimmed_gripper,
            trimmed_time_seq,
        )
        n_steps = len(umi_se3_list)

    umi_se3_list, skipped_mask = filter_glitches(
        umi_se3_list, time_seq, args.max_linear_vel, args.max_angular_vel
    )
    print(
        f"[ReplayUmiOnFairino5] Skipped {int(skipped_mask.sum())}/{n_steps} glitch "
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
            f"[ReplayUmiOnFairino5] Drift filter (< {args.drift_linear_vel} m/s "
            f"or {args.drift_angular_vel} deg/s for {args.drift_confirm_frames} "
            f"consecutive frames = held): position range {np.round(pre_drift_range, 4)} "
            f"-> {np.round(post_drift_range, 4)} m"
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

    # Translation delta is rotated into the FR5 base frame by
    # vive_world_to_base_frame_rotation (identity, i.e. the old naive
    # behavior, unless --vive_config was given), the same left-multiply as
    # live teleop's ViveInputDevice.set_command_data. Rotation delta is used
    # as-is: it already has vive_to_eef_frame_rotation baked in from
    # recording (see this module's docstring's rotation note) -- conjugating
    # by it again here would apply it twice.
    #
    # delta_umi_rotation is applied to init_se3.rotation by LEFT-multiply
    # (world-frame composition), NOT right-multiply (body-frame
    # composition). umi.urdf's floating joint has zero rotation offset from
    # world_link, so delta_umi_rotation (= umi_se3_0.rotation.T @
    # umi_se3_t.rotation) is effectively already expressed in WORLD-frame
    # terms (umi_se3_0.rotation is close to identity). Right-multiplying it
    # onto init_se3.rotation would instead interpret it as a delta in FR5's
    # OWN current-orientation-dependent local frame, silently reinterpreting
    # each world-frame axis through init_se3.rotation -- confirmed on a real
    # demo: a recorded rotation axis of e.g. [0.8, -0.57, 0.19] came out as
    # [-0.8, -0.17, 0.58] after right-multiply retargeting (same angle,
    # scrambled axis), which is exactly why a demo recorded with almost no
    # roll/pitch showed the FR5 target rolling/pitching heavily. Left-
    # multiplying instead reproduces the recorded world-frame axis exactly.
    target_se3_list = []
    for umi_se3_t in umi_se3_list:
        target_translation = init_se3.translation + pos_scale * (
            vive_world_to_base_frame_rotation
            @ (umi_se3_t.translation - umi_se3_0.translation)
        )
        delta_umi_rotation = umi_se3_0.rotation.T @ umi_se3_t.rotation
        target_rotation = delta_umi_rotation @ init_se3.rotation
        target_se3_list.append(pin.SE3(target_rotation, target_translation))

    checkpoint_se3_list, checkpoint_gripper_list, _checkpoint_time_seq = (
        sample_checkpoints(
            target_se3_list, list(umi_gripper), time_seq, args.checkpoint_hz
        )
    )
    n_checkpoints = len(checkpoint_se3_list)
    print(
        f"[ReplayUmiOnFairino5] Sampled {n_checkpoints} checkpoints "
        f"(~{args.checkpoint_hz} Hz) from {n_steps} recorded frames -- "
        f"tracking each in turn with closed-loop PI control (max "
        f"{args.max_iters_per_checkpoint} iterations each)..."
    )

    max_joint_step_deg = (
        None if args.max_joint_step_deg <= 0 else args.max_joint_step_deg
    )
    rot_tol = np.deg2rad(args.rot_tol_deg)
    max_step_rot = np.deg2rad(args.max_step_rot_deg)
    integral_clamp_rot = np.deg2rad(args.integral_clamp_rot_deg)

    checkpoint_target_se3_list = []  # one entry per LOGGED control iteration
    measured_se3_list = []
    command_joint_deg_list = []
    measured_joint_deg_list = []
    min_singular_value_list = []
    # Actual wall-clock timestamp per iteration, NOT iteration_idx * env.dt:
    # on --real, _set_action's own comments note that the real per-call
    # cadence is jittery (XML-RPC round trips for GetActualJointPosRadian
    # etc. add latency well beyond env.dt), so a nominal-dt time axis would
    # mislabel how much real time each iteration actually took -- misleading
    # for exactly the kind of tracking-lag analysis --log_csv/
    # plot_joint_comparison is for.
    step_time_list = []
    replay_start_time = time.time()
    obs = env.unwrapped._get_obs()
    n_reached = 0
    try:
        for cp_idx in range(n_checkpoints):
            reached, n_iters, _final_se3, obs, log = track_checkpoint(
                env,
                arm_manager,
                checkpoint_se3_list[cp_idx],
                checkpoint_gripper_list[cp_idx],
                obs,
                replay_start_time,
                pos_tol=args.pos_tol,
                rot_tol=rot_tol,
                kp_pos=args.kp_pos,
                ki_pos=args.ki_pos,
                kp_rot=args.kp_rot,
                ki_rot=args.ki_rot,
                max_step_pos=args.max_step_pos,
                max_step_rot=max_step_rot,
                max_iters=args.max_iters_per_checkpoint,
                integral_clamp_pos=args.integral_clamp_pos,
                integral_clamp_rot=integral_clamp_rot,
                low_manip_threshold=args.low_manip_threshold,
                max_extra_damping=args.max_extra_damping,
                max_joint_step_deg=max_joint_step_deg,
            )
            n_logged = len(log["min_singular_value"])
            checkpoint_target_se3_list.extend(
                [checkpoint_se3_list[cp_idx]] * n_logged
            )
            measured_se3_list.extend(log["measured_se3"])
            command_joint_deg_list.extend(log["command_joint_deg"])
            measured_joint_deg_list.extend(log["measured_joint_deg"])
            min_singular_value_list.extend(log["min_singular_value"])
            step_time_list.extend(log["step_time"])
            n_reached += int(reached)

            status = "reached" if reached else "*** TIMED OUT -- moving on ***"
            print(
                f"[ReplayUmiOnFairino5] checkpoint {cp_idx + 1}/{n_checkpoints}: "
                f"{status} in {n_iters} iterations"
            )
    except KeyboardInterrupt:
        print("[ReplayUmiOnFairino5] Interrupted.")
    finally:
        env.close()
        print(
            f"[ReplayUmiOnFairino5] Done. Reached {n_reached}/{n_checkpoints} "
            "checkpoints within tolerance."
        )
    target_se3_list = checkpoint_target_se3_list

    if len(min_singular_value_list) > 0:
        min_singular_value_arr = np.array(min_singular_value_list)
        n_low_manip_steps = int(
            (min_singular_value_arr < args.low_manip_threshold).sum()
        )
        print(
            f"[ReplayUmiOnFairino5] adaptive_ik_step: min singular value over "
            f"replay = {min_singular_value_arr.min():.3f}, "
            f"{n_low_manip_steps}/{len(min_singular_value_arr)} steps "
            f"({100.0 * n_low_manip_steps / len(min_singular_value_arr):.1f}%) "
            f"got extra IK damping (< --low_manip_threshold="
            f"{args.low_manip_threshold})"
        )

    if args.compare_plot != "" and len(measured_se3_list) > 1:
        compare_plot_path = args.compare_plot
        if compare_plot_path is None:
            base, _ext = os.path.splitext(os.path.normpath(args.rmb_filename))
            compare_plot_path = f"{base}_vive_vs_robot.png"
        max_pos_err, mean_pos_err, max_rot_err_deg, mean_rot_err_deg = (
            plot_vive_vs_robot(
                target_se3_list, measured_se3_list, compare_plot_path
            )
        )
        print(
            f"[ReplayUmiOnFairino5] Vive-vs-robot tracking error over "
            f"{len(measured_se3_list)} steps: "
            f"position max={max_pos_err:.4f} m / mean={mean_pos_err:.4f} m, "
            f"orientation max={max_rot_err_deg:.2f} deg / "
            f"mean={mean_rot_err_deg:.2f} deg"
        )
        if dry_run:
            print(
                "[ReplayUmiOnFairino5] NOTE: this was a dry_run, so the "
                "'measured' path above is just an echo of the commanded "
                "path (near-zero error is expected and does NOT confirm "
                "hardware tracking) -- rerun with --real for a meaningful "
                "comparison."
            )
        print(f"[ReplayUmiOnFairino5] Saved comparison plot to {compare_plot_path}")

    if args.log_csv != "" and len(measured_se3_list) > 1:
        log_csv_path = args.log_csv
        if log_csv_path is None:
            base, _ext = os.path.splitext(os.path.normpath(args.rmb_filename))
            log_csv_path = f"{base}_replay_log.csv"

        command_joint_deg = np.array(command_joint_deg_list)
        measured_joint_deg = np.array(measured_joint_deg_list)
        n_logged = len(measured_se3_list)

        import csv

        n_joints = command_joint_deg.shape[1]
        header = (
            ["step", "time", "min_singular_value"]
            + [f"command_joint{j + 1}_deg" for j in range(n_joints)]
            + [f"measured_joint{j + 1}_deg" for j in range(n_joints)]
            + [
                f"command_eef_{name}"
                for name in ("tx", "ty", "tz", "qw", "qx", "qy", "qz")
            ]
            + [
                f"measured_eef_{name}"
                for name in ("tx", "ty", "tz", "qw", "qx", "qy", "qz")
            ]
        )
        with open(log_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for t in range(n_logged):
                command_pose_t = get_pose_from_se3(target_se3_list[t])
                measured_pose_t = get_pose_from_se3(measured_se3_list[t])
                writer.writerow(
                    [t, step_time_list[t], min_singular_value_list[t]]
                    + list(command_joint_deg[t])
                    + list(measured_joint_deg[t])
                    + list(command_pose_t)
                    + list(measured_pose_t)
                )
        print(
            f"[ReplayUmiOnFairino5] Saved {n_logged}-step joint/TCP log to "
            f"{log_csv_path}"
        )

        joint_plot_path = os.path.splitext(log_csv_path)[0] + "_joints.png"
        plot_joint_comparison(
            command_joint_deg,
            measured_joint_deg,
            np.array(step_time_list),
            joint_plot_path,
        )
        print(f"[ReplayUmiOnFairino5] Saved joint-angle comparison plot to {joint_plot_path}")


if __name__ == "__main__":
    main()
