"""Replay a UMI-collected demonstration (RealUMIDemoEnv, a robot-less handheld
rig -- see envs/real/umi/RealUMIEnvBase.py) onto the FR5 arm
(RealFairino5DemoEnv), by retargeting the recorded EEF trajectory and
tracking it with closed-loop, pure-pursuit path following.

Unlike bin/Teleop.py's built-in --replay_log (which replays a log recorded by
the SAME env, using its own DOF/frame conventions unchanged -- see
teleop/TeleopBase.py's ReplayPhase), this script retargets across envs: from
the UMI rig's own (robot-less, virtual) body onto the FR5's.

RETARGETING: both rotation and translation are retargeted as pure EEF/TCP-
LOCAL reproductions -- "do the same relative motion, on FR5's own gripper,
starting from FR5's own init pose" -- matching how ViveInputDevice.
set_command_data itself drives a robot during live single-robot teleop. This
is inherently independent of how the recording room relates to FR5's base
placement, so no vive_world_to_base_frame_rotation (room-vs-base) calibration
is used anywhere here -- see the rotation note below for the full derivation
and the two wrong versions this went through before landing here.

CONTROL STRATEGY -- pure pursuit, not a fixed checkpoint schedule:
this script went through two earlier control strategies before landing here,
both confirmed broken on real hardware and in a MuJoCo mirror
(misc/TeleopUmiWithMujocoMirror.py):
  (1) Feedforward interpolation: retarget EVERY recorded UMI frame, densely
      interpolate between them, and drive one feedforward damped-least-
      squares IK step per interpolated pose, warm-started from its OWN
      internal kinematic state (never re-checking the real arm's actual
      measured position). Open-loop: if the arm fell behind at any single
      step (e.g. a low-manipulability/near-singular segment), nothing
      corrected it, and error accumulated for the rest of the replay.
  (2) Checkpoint tracking: downsample to a sparse, FIXED, time-indexed
      sequence of checkpoints (~--checkpoint_hz) and run closed-loop PI
      control (re-reading the arm's real measured pose every iteration) to
      convergence on each in turn, giving up and moving to the next after
      --max_iters_per_checkpoint. This closed the loop WITHIN each
      checkpoint, but the checkpoint SCHEDULE itself was still open-loop --
      it never waited for the arm's actual progress. Confirmed on real data:
      MuJoCo's own actuators (particularly the low-gain wrist joints)
      settle to a small but real steady-state position error under a
      commanded step -- a physics/actuator-dynamics limitation, not a
      kinematics problem (pure IK converges to ~0 error on the same,
      collision-free target instantly; raising --max_iters_per_checkpoint to
      3000 changed nothing, error was already at its physical floor).
      Once even slightly behind, the gap to the NEXT (already-scheduled,
      oblivious) checkpoint only grew -- confirmed to run away past 0.4m
      of position error over a couple hundred checkpoints. Loosening
      --pos_tol only delayed the onset, since the schedule still never
      waits.

This version (see sample_checkpoints/track_path_pure_pursuit) instead
continuously re-anchors the immediate target to the arm's OWN CURRENT
progress along the (densified, ~--checkpoint_hz) path:
  1. Every iteration, find the path point closest (by position) to the arm's
     REAL measured EEF pose, searching only FORWARD from the previous
     closest point (--search_window points ahead, never backward).
  2. Aim --lookahead_points further ahead of that closest point as the
     immediate target (the "carrot") -- never the closest point itself.
  3. Take one clamped PI-controlled step toward the carrot, exactly like
     strategy (2)'s inner loop (re-sync to the real measured joints, run one
     adaptive_ik_step toward a nearby SE3 sub-target).
If the arm falls behind, the closest point (and the carrot) simply advances
more slowly -- there is no fixed schedule to be left behind by, so a
transient slowdown causes temporary lag, not permanent, compounding
divergence. Finishes once the closest point reaches the end of the path AND
the remaining error is within --pos_tol/--rot_tol_deg, or after
--max_total_iters (a whole-path safety cap, not a per-checkpoint one).

--low_manip_threshold/--max_extra_damping/--max_joint_step_deg tune the
per-iteration IK step's (adaptive_ik_step) response to low-manipulability
configurations. Both default to 0 (disabled): exact TCP tracking is
prioritized over motion smoothness for this task, so by default this reduces
to ArmManager's own plain fixed-damping IK step -- see that function's
docstring if smoothing is ever needed instead.

--vive_config: pass the SAME teleop/configs/*.yaml used to record this demo
(e.g. teleop/configs/ViveUMI.yaml) -- the only thing actually read from it is
its own pos_scale, used as the --pos_scale default if --pos_scale is not
passed explicitly (falling back to 1.0 if the config has none either),
matching how the demo was recorded. Its vive_world_to_base_frame_rotation and
vive_to_eef_frame_rotation are NOT read here: see the rotation note below for
why neither applies to this retargeting.

SAFETY: defaults to dry_run (no hardware connection; env prints the ServoJ
command it would have sent). Pass --real only once you have verified the
printed commands look sane (e.g. via CheckUmiFairino5Reachability.py and a
dry-run read-through of this script's own printed output) and are ready to
move the physical arm.

Note on rotation: ViveInputDevice.set_command_data computes the commanded
rotation as
    eef_se3_at_enable.rotation @ (vive_to_eef_frame_rotation
        @ delta_vive_rotation @ vive_to_eef_frame_rotation.T)
i.e. composed via RIGHT-multiply as an EEF/TCP-LOCAL delta relative to the
tool's enable-time orientation (see that method's own comment). Since UMI's
eef_se3_at_enable is identity (its virtual body's floating joint has zero
rotation offset from world_link), umi_se3_t.rotation IS that local delta
directly, and delta_umi_rotation (= umi_se3_0.rotation.T @ umi_se3_t.rotation,
umi_se3_0.rotation ~= I) is too. Reapplying a LOCAL delta onto a different
robot's own init orientation uses the same right-multiply composition:
target_rotation = init_se3.rotation @ delta_umi_rotation.

Note on translation: ViveInputDevice.set_command_data accumulates it as, each
frame, this frame's tiny room-frame motion converted to an EEF-local
increment (via the tracker's own current rotation, then
vive_to_eef_frame_rotation) and re-projected into the WORLD/base frame by
THAT frame's own (evolving) target_rotation before being added onto the
running position -- "push the tracker forward always means push the TCP
forward along its own CURRENT Z, even while simultaneously rotating" (see
that method's own comment). So umi_se3_t.translation is not itself a
local quantity, but each frame-to-frame INCREMENT can be recovered as a
local one by un-rotating it with THAT frame's own umi_se3_t.rotation, then
re-accumulated through FR5's own (already correctly retargeted)
target_rotation at each frame -- reproducing the same "push forward along
the TCP's own current, possibly-tilted, Z" on FR5 exactly as it happened on
the UMI rig. This only needs the recorded pose sequence (umi_se3_list), not
live UMI/Vive hardware, so it works identically for offline replay -- see
main()'s retarget loop.

Neither channel uses vive_world_to_base_frame_rotation (vwtb, room-vs-base
placement) at all: once both are local-frame reproductions, retargeting is
inherently independent of how the room relates to any robot's base --
vwtb calibrates a physical relationship (lighthouse vs. robot mounting) that
simply doesn't enter into "reproduce this local motion on a different
gripper."

This went through several wrong versions before landing here, all confirmed
on real hardware and/or in a MuJoCo mirror
(misc/TeleopUmiWithMujocoMirror.py):
(1) Rotation via LEFT-multiply with no vwtb (the original code) scrambled
axes -- e.g. a recorded rotation axis of [0.8, -0.57, 0.19] came out as
[-0.8, -0.17, 0.58] (same angle, wrong axis) -- because it silently
reinterpreted an EEF-local delta as if it were a world-frame one being
composed onto FR5's own current-orientation-dependent local frame.
(2) Assuming rotation WAS room-frame like translation and conjugating it by
vwtb before LEFT-multiplying produced a different but equally wrong result
(recorded yaw came out as commanded pitch, roll as -yaw) -- rotation was
never room-frame to begin with.
(3) Translation as one batch delta from t=0 rotated by a fixed vwtb (the
original code, and still correct in isolation for a NON-rotating demo) broke
down once the demo also rotated: recorded forward/up/right motion came out
as commanded up/left/backward, because a fixed room-frame rotation can't
account for the commanded direction changing as the (now-correctly-tracked)
TCP orientation changes mid-demo -- translation needed the same per-frame
local-frame treatment rotation got, not a global room-frame one.
"""

import argparse
import os
import time

import gymnasium as gym
import mujoco
import numpy as np
import pinocchio as pin
import videoio
import yaml

from robo_manip_baselines.common import (
    ArmManager,
    DataKey,
    EpisodeRelativeEefPoseRetargeter,
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
        "--sim",
        action="store_true",
        help="replay against the MuJoCo FR5 (MujocoFairino5CableEnv) instead of "
        "real hardware -- opens a viewer window, no hardware risk. Mutually "
        "exclusive with --real; --robot_ip/--gripper_*/--real are ignored when set.",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="actually connect to and move the FR5 (DANGEROUS -- omit this to "
        "stay in dry_run, which only prints the commands). Ignored if --sim.",
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
    parser.add_argument("--gripper_do_close_id", type=int, default=0)
    parser.add_argument("--gripper_do_open_id", type=int, default=1)
    parser.add_argument(
        "--vive_config",
        type=str,
        default=None,
        help="teleop/configs/*.yaml used to record this demo (e.g. "
        "teleop/configs/ViveUMI.yaml) -- only its pos_scale is used, as the "
        "--pos_scale default if --pos_scale is not passed explicitly (see "
        "module docstring for why vive_world_to_base_frame_rotation is not "
        "used by this script's retargeting).",
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
        help="densify the recorded UMI trajectory to a path point every "
        "~1/checkpoint_hz seconds (always keeping the first and last "
        "frame, see sample_checkpoints()). This is the reference path "
        "track_path_pure_pursuit continuously follows -- NOT a fixed "
        "waypoint schedule the arm must arrive at in turn (see that "
        "function's docstring for why the earlier checkpoint-schedule "
        "design was replaced).",
    )
    parser.add_argument(
        "--pose_key",
        type=str,
        default=DataKey.COMMAND_EEF_POSE,
        choices=[DataKey.COMMAND_EEF_POSE, DataKey.MEASURED_EEF_POSE],
        help="which recorded EEF pose sequence to retarget. Default is "
        "command_eef_pose because that is literally the quantity the "
        "recording mirror (misc/TeleopUmiWithMujocoMirror.py) consumed live "
        "-- it reads the UMI body manager's target_se3, and "
        "ArmManager.get_command_eef_pose() returns exactly that -- so "
        "replaying it reproduces the recorded motion. measured_eef_pose is "
        "the forward-kinematics readback of the UMI rig's virtual body, "
        "which differs from the command by up to ~0.17m on real data (the "
        "virtual free-flyer's IK does not land exactly on target), and was "
        "the old, wrong default.",
    )
    parser.add_argument(
        "--time_scale",
        type=float,
        default=1.0,
        help="playback speed. 1.0 replays at the speed the demo was actually "
        "performed (the trajectory is resampled onto the env's control "
        "period using the recorded timestamps -- see the resampling comment "
        "in main()). Values >1 stretch the demo out and move the arm more "
        "slowly, which is the safe direction; e.g. 2.0 halves every "
        "commanded joint speed. Values <1 speed it up.",
    )
    parser.add_argument(
        "--lever_arm_correction",
        type=str,
        default="config",
        help="undo the TCP->tracker lever arm in demos recorded before "
        "ViveInputDevice corrected for it. 'config' uses --vive_config's "
        "vive_to_eef_translation (and is a no-op if the config has none); "
        "'none' disables it -- use that for demos recorded AFTER the fix, "
        "which already have the TCP position baked in and would otherwise be "
        "corrected twice. A literal 'x,y,z' in metres overrides both.",
    )
    parser.add_argument(
        "--mirror_exact",
        action="store_true",
        help="reproduce the recording as faithfully as possible: disables "
        "--warmup_seconds trimming and the glitch/drift filters, so the "
        "replayed pose sequence is exactly the one the recording mirror "
        "saw. Those filters are useful protection when driving REAL "
        "hardware from a noisy Vive recording, but they alter the sequence "
        "(and, via warmup trimming, the retarget's t=0 reference pose), so "
        "they make the replay diverge from the recording.",
    )
    parser.add_argument(
        "--closed_loop",
        action="store_true",
        help="use the closed-loop pure-pursuit tracker instead of the "
        "default open-loop replay. NOT recommended for faithful replay: "
        "closing the loop on measured state makes the controller fight "
        "MuJoCo's permanent actuator droop and drift off the recorded path "
        "(see track_path_open_loop's docstring). Kept for experimentation.",
    )
    parser.add_argument(
        "--ik_steps_per_frame",
        type=int,
        default=1,
        help="open-loop replay only: IK steps per recorded frame. 1 (the "
        "default) exactly reproduces what the recording mirror did, "
        "including its own tracking lag -- this is what makes replay match "
        "the recording. Raise it (e.g. 20) to instead track the idealized "
        "retargeted trajectory tightly, which will NOT match the recording.",
    )
    parser.add_argument(
        "--save_video",
        type=str,
        default=None,
        help="sim only: path to save an mp4 of the replay, for visual "
        "comparison against the recording's *_mujoco_mirror.mp4. If "
        "omitted, saved next to rmb_filename as "
        "'<rmb_basename>_replay_sim.mp4'. Pass an empty string to skip.",
    )
    parser.add_argument(
        "--lookahead_points",
        type=int,
        default=10,
        help="pure pursuit: aim this many path points ahead of the arm's "
        "current closest point on the path (at --checkpoint_hz, e.g. 10 "
        "points ~= 0.5s ahead at 20Hz) -- the 'carrot' the controller "
        "chases. Larger = smoother but cuts corners more; smaller = "
        "tighter path tracking but jerkier.",
    )
    parser.add_argument(
        "--search_window",
        type=int,
        default=50,
        help="pure pursuit: how many path points ahead of the previous "
        "closest point to search for the new closest point each iteration "
        "(forward-only). Must be large enough that the arm's per-iteration "
        "progress along the path never exceeds it, or the closest-point "
        "search will fail to advance past a temporarily-stalled segment.",
    )
    parser.add_argument(
        "--pos_tol",
        type=float,
        default=0.003,
        help="[m] the replay counts as finished once the arm's closest "
        "point has reached the end of the path AND the remaining position "
        "error to the final pose drops below this",
    )
    parser.add_argument(
        "--rot_tol_deg",
        type=float,
        default=1.0,
        help="[deg] see --pos_tol -- the orientation equivalent",
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
        "--max_total_iters",
        type=int,
        default=100000,
        help="give up on the ENTIRE replay (log a warning, stop where the "
        "arm is) after this many control iterations total without the "
        "closest-point-on-path reaching the end within --pos_tol/"
        "--rot_tol_deg. This is a whole-path safety cap, not a per-"
        "checkpoint one (see track_path_pure_pursuit) -- pure pursuit has "
        "no per-waypoint timeout to begin with, since it re-anchors to the "
        "arm's own progress every iteration.",
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
    """Resample a recorded trajectory to a sequence of poses at
    (approximately) checkpoint_hz, evenly spaced in time from the first to
    the last recorded frame. This is the dense reference PATH
    track_path_pure_pursuit continuously follows (see that function's
    docstring) -- despite the name (kept for continuity with an earlier,
    now-replaced control strategy that treated these as fixed waypoints to
    reach in turn), these are points ON a path to be tracked smoothly, not
    discrete stops.

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


def track_path_open_loop(
    env,
    arm_manager,
    target_se3_list,
    gripper_list,
    obs,
    replay_start_time,
    ik_steps_per_frame=1,
    frame_callback=None,
):
    """Reproduce the recorded demo by replaying the EXACT control algorithm
    that generated it, open-loop: for each retargeted frame, run
    ik_steps_per_frame damped-least-squares IK step(s) warm-started from the
    controller's OWN previous joint solution (never re-synced to the
    simulator's/robot's measured state), command that solution, step once.

    This is byte-for-byte the same computation misc/TeleopUmiWithMujocoMirror.py
    performs live while recording (ArmManager.set_command_eef_pose() ->
    one inverse_kinematics() step -> env.step()), so given the same recorded
    poses it produces the same joint-command stream, and therefore the same
    arm motion. That equivalence is the whole point: the mirror is the
    ground truth of "what the arm did during recording", so matching it
    exactly is what makes replay faithful.

    WHY OPEN LOOP (this was the hard-won lesson, see module docstring's
    CONTROL STRATEGY): closing the loop on measured state actively BREAKS
    fidelity here. MuJoCo's position actuators carry a large permanent
    steady-state droop -- measured directly: commanding a fixed joint target
    and holding it for 400 steps still leaves ~2.9 deg of error on joint 2,
    and it is NOT contact-related (identical with all collisions disabled),
    it is gravity sag the actuator's finite gain never closes. The recording
    mirror had that exact same droop and simply never looked at it. A
    closed-loop replay, by contrast, SEES the droop, treats it as tracking
    error, and commands past the target to correct it -- which is both
    unachievable (the droop returns immediately) and actively harmful: the
    integral term winds up, the commanded pose drifts off the recorded path,
    and the arm walks itself into self-collision (forearm_link vs
    wrist2_link) that the recorded motion never had. Open loop reproduces
    the droop instead of fighting it, so the motion matches.

    Note that one IK step per frame does NOT converge to the target within
    a frame -- on real data it lags the retargeted target by ~5cm mean /
    ~23cm peak during fast motion. That lag is not a defect to be fixed
    here: it is part of what the mirror actually did, so reproducing it is
    correct. Raising ik_steps_per_frame (e.g. 20 converges to ~0 lag) makes
    the arm track the retargeted target more tightly than the recording did
    -- useful if the goal is the idealized trajectory rather than a faithful
    replay, but it will NOT match the recorded motion.

    Returns (n_iters, final_measured_se3, obs, log) with the same log keys
    as track_path_pure_pursuit (path_idx is the frame index here)."""
    log = {
        "min_singular_value": [],
        "command_joint_deg": [],
        "measured_joint_deg": [],
        "measured_se3": [],
        "target_se3": [],
        "path_idx": [],
        "step_time": [],
    }
    measured_se3 = None
    for frame_idx, (target_se3, gripper_target) in enumerate(
        zip(target_se3_list, gripper_list)
    ):
        for _ in range(ik_steps_per_frame):
            arm_manager.set_command_eef_pose(target_se3)
        arm_manager.set_command_gripper_joint_pos(gripper_target)

        command_arm_joint_pos = arm_manager.arm_joint_pos.copy()
        action = np.concatenate(
            [command_arm_joint_pos, arm_manager.gripper_joint_pos]
        )
        obs = env.step(action)[0]

        result_arm_joint_pos = env.unwrapped.get_joint_pos_from_obs(obs)[
            arm_manager.body_config.arm_joint_idxes
        ]
        measured_se3 = get_se3_from_pose(
            arm_manager.get_eef_pose_from_joint_pos(result_arm_joint_pos)
        )
        # Jacobian conditioning is reported for parity with the closed-loop
        # tracker's diagnostics; it does not feed back into control here.
        J = pin.computeJointJacobian(
            arm_manager.pin_model,
            arm_manager.pin_data,
            arm_manager.arm_joint_pos,
            arm_manager.body_config.ik_eef_joint_id,
        )
        log["min_singular_value"].append(np.linalg.svd(J, compute_uv=False).min())
        log["command_joint_deg"].append(np.rad2deg(command_arm_joint_pos))
        log["measured_joint_deg"].append(np.rad2deg(result_arm_joint_pos))
        log["measured_se3"].append(measured_se3)
        log["target_se3"].append(target_se3)
        log["path_idx"].append(frame_idx)
        log["step_time"].append(time.time() - replay_start_time)

        if frame_callback is not None:
            frame_callback()

    return len(target_se3_list), measured_se3, obs, log


def track_path_pure_pursuit(
    env,
    arm_manager,
    dense_se3_list,
    dense_gripper_list,
    obs,
    replay_start_time,
    lookahead_points=10,
    search_window=50,
    pos_tol=0.003,
    rot_tol=np.deg2rad(1.0),
    kp_pos=0.5,
    ki_pos=0.1,
    kp_rot=0.5,
    ki_rot=0.1,
    max_step_pos=0.005,
    max_step_rot=np.deg2rad(2.0),
    max_total_iters=100000,
    integral_clamp_pos=0.02,
    integral_clamp_rot=np.deg2rad(10.0),
    low_manip_threshold=0.16,
    max_extra_damping=0.0,
    max_joint_step_deg=None,
):
    """Pure-pursuit style path tracking, replacing the earlier fixed,
    time-indexed CHECKPOINT schedule (formerly track_checkpoint): that
    approach pre-computed a checkpoint every ~1/checkpoint_hz seconds and
    marched through them in order regardless of whether the arm actually
    caught up to the previous one. Once the arm fell even slightly behind
    (confirmed on real hardware and in a MuJoCo mirror: MuJoCo's own
    actuators, especially the low-gain wrist joints, settle to a small but
    real STEADY-STATE position error under a commanded step -- a physics/
    actuator-dynamics limitation, not a kinematics or singularity problem;
    pure IK converges to ~0 error on the same target instantly, and the
    target itself is provably collision-free), the gap to the NEXT
    (already-scheduled, oblivious-to-progress) checkpoint only grew --
    confirmed to run away to >0.4m of position error over a couple hundred
    checkpoints. Loosening --pos_tol only delayed the onset; it could not
    fix the underlying issue, because the schedule itself never waits.

    This function instead continuously re-anchors the immediate target to
    the arm's OWN CURRENT progress along the path:
      1. Every iteration, find the point on dense_se3_list closest (by
         position) to the arm's REAL measured EEF pose, searching only
         FORWARD from the previous closest point (up to search_window
         points ahead) -- never backward, since the path is not meant to be
         retraced.
      2. Aim lookahead_points further ahead of that closest point as the
         immediate target (the "carrot") -- never the closest point itself,
         which would have the arm constantly chasing its own current
         position and stalling.
      3. Take one clamped PI-controlled step toward the carrot (same
         inner mechanics as the old track_checkpoint: re-sync arm_manager's
         IK state to the real measured joints, run one adaptive_ik_step
         toward a nearby SE3 sub-target -- IK never has to solve a large
         jump in one step).
    If the arm falls behind, the closest point (and therefore the carrot)
    simply advances more slowly -- there is no fixed schedule to be left
    behind by, so a transient slowdown (actuator settling, a difficult
    segment) causes temporary lag, not permanent, compounding divergence.

    Finishes once the closest point reaches the end of dense_se3_list AND
    the remaining error to the final pose is within pos_tol/rot_tol, or
    after max_total_iters iterations (logged as a warning -- this is a
    whole-path budget now, not a per-checkpoint one).

    Returns (reached_end: bool, n_iters, final_path_idx, final_measured_se3,
    obs, log) where log is a dict of per-iteration lists: min_singular_value,
    command_joint_deg, measured_joint_deg, measured_se3, target_se3 (the
    carrot at that iteration, for --compare_plot/--log_csv), path_idx,
    step_time (actual wall-clock elapsed since replay_start_time, for
    --log_csv/plot_joint_comparison -- see their docstrings for why nominal
    step_idx*dt is not used on --real)."""
    integral_pos = np.zeros(3)
    integral_rot = np.zeros(3)
    log = {
        "min_singular_value": [],
        "command_joint_deg": [],
        "measured_joint_deg": [],
        "measured_se3": [],
        "target_se3": [],
        "path_idx": [],
        "step_time": [],
    }

    n_path = len(dense_se3_list)
    path_idx = 0
    measured_se3 = None
    n_iters = 0
    reached_end = False
    for i in range(max_total_iters):
        measured_joint_pos = env.unwrapped.get_joint_pos_from_obs(obs)
        measured_arm_joint_pos = measured_joint_pos[
            arm_manager.body_config.arm_joint_idxes
        ]
        measured_se3 = get_se3_from_pose(
            arm_manager.get_eef_pose_from_joint_pos(measured_arm_joint_pos)
        )

        # Advance path_idx to the path point closest to where the arm
        # REALLY is right now (forward-only search). A recorded demo can
        # hold nearly still for a while (e.g. a few seconds before the
        # operator starts moving) -- dozens of consecutive path points can
        # then sit at EXACTLY identical positions (filter_drift() literally
        # repeats/pins the held value, not just "close" -- confirmed
        # bit-identical, 0.0 distance apart). Picking the single FIRST
        # (earliest-index) minimum would get permanently stuck at the start
        # of such a flat run (confirmed: path_idx never left 0 across 8000
        # iterations on real data with a ~30-frame near-zero-motion opening
        # segment), since nothing ever looks strictly closer than that first
        # point. So: find the best distance in the window, then take the
        # LATEST index within tie_eps of it, using position+rotation
        # together (see se3_dist below) so two path points that merely
        # happen to share a similar POSITION -- but different orientation,
        # or reached at a very different time -- are never confused for
        # duplicates.
        #
        # tie_eps must stay very tight: it exists ONLY to race through truly
        # duplicate/near-duplicate held frames, not to forgive any real
        # tracking lag. Two wrong, looser versions were tried first, both
        # confirmed on real data: (1) tie_eps proportional to best_dist
        # widened right along with genuine lag and made path_idx jump from
        # 0 straight to the LAST path index within a few iterations --
        # pursuit degenerated into one giant beeline to the final pose
        # (looked like the arm "wasn't moving at all": creeping, tiny
        # clamped step by tiny clamped step, toward a single far-away
        # point). (2) tie_eps fixed at --pos_tol (3mm, chosen because it's
        # already "close enough" for the FINAL completion check) was STILL
        # too loose whenever a demo's motion revisits a spatial
        # neighborhood it passed through earlier (common in natural hand
        # motion -- move out, come back near the same spot, continue) --
        # points that are position-close but hours (well, seconds) apart in
        # the recording got merged, and path_idx galloped through the
        # revisited region straight to wherever in the window looked
        # "closest enough", again skipping the actual intervening path.
        # DUPLICATE_TIE_EPS is deliberately unrelated to pos_tol: it is
        # sized to catch true (near-)duplicates only.
        DUPLICATE_TIE_EPS = 1e-4  # [m equivalent, see se3_dist] exact-hold detection only

        def se3_dist(a, b):
            pos_d = np.linalg.norm(a.translation - b.translation)
            rot_d = np.linalg.norm(pin.log3(a.rotation.T @ b.rotation))
            return pos_d + rot_d * 0.05  # ~0.05m per radian of orientation diff

        best_dist = se3_dist(dense_se3_list[path_idx], measured_se3)
        search_end = min(path_idx + search_window, n_path - 1)
        for j in range(path_idx + 1, search_end + 1):
            d = se3_dist(dense_se3_list[j], measured_se3)
            if d < best_dist:
                best_dist = d
        best_idx = path_idx
        for j in range(path_idx, search_end + 1):
            d = se3_dist(dense_se3_list[j], measured_se3)
            if d <= best_dist + DUPLICATE_TIE_EPS:
                best_idx = j
        path_idx = best_idx

        carrot_idx = min(path_idx + lookahead_points, n_path - 1)
        target_se3 = dense_se3_list[carrot_idx]
        gripper_target = dense_gripper_list[carrot_idx]

        n_iters = i
        if path_idx >= n_path - 1:
            final_error_se3 = measured_se3.actInv(dense_se3_list[-1])
            final_error_vec = pin.log6(final_error_se3).vector
            if (
                np.linalg.norm(final_error_vec[:3]) < pos_tol
                and np.linalg.norm(final_error_vec[3:]) < rot_tol
            ):
                reached_end = True
                break

        error_se3 = measured_se3.actInv(target_se3)
        error_vec = pin.log6(error_se3).vector  # (linear, angular), local frame
        pos_err, rot_err = error_vec[:3], error_vec[3:]

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
        # model.
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
        log["target_se3"].append(target_se3)
        log["path_idx"].append(path_idx)
        log["step_time"].append(time.time() - replay_start_time)

    return reached_end, n_iters, path_idx, measured_se3, obs, log


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

    # vive_world_to_base_frame_rotation (room-vs-base calibration) is NOT
    # used by the retargeting below -- see the module docstring's rotation
    # note: both translation and rotation are now retargeted as pure
    # EEF/TCP-LOCAL reproductions (matching how ViveInputDevice.
    # set_command_data itself works for live single-robot teleop), which is
    # inherently independent of how the room relates to any robot's base.
    # --vive_config is only still read here for its pos_scale default.
    # vive_to_eef_frame_rotation from that config is NOT reapplied either:
    # it's already baked into the recorded MEASURED_EEF_POSE at record time
    # (see the docstring's rotation note), and reapplying it would double it.
    config_pos_scale = None
    if args.vive_config is not None:
        print(f"[ReplayUmiOnFairino5] Load {args.vive_config}")
        with open(args.vive_config, "r") as f:
            vive_config = yaml.safe_load(f)
        config_pos_scale = vive_config.get("pos_scale")
        config_lever_arm = vive_config.get("vive_to_eef_translation")
    pos_scale = args.pos_scale
    if pos_scale is None:
        pos_scale = config_pos_scale if config_pos_scale is not None else 1.0
    print(f"[ReplayUmiOnFairino5] Using pos_scale={pos_scale}")

    # Lever-arm correction for demos recorded BEFORE ViveInputDevice applied
    # it (see that class's vive_to_eef_translation). Those recordings tracked
    # the tracker, which sits some distance from the TCP, so the tool's own
    # rotation injected phantom translation into them.
    #
    # It can be undone exactly, after the fact, from the recording alone.
    # At record time the accumulated translation is
    #     p_rec(t) - p_rec(0) = pos_scale * C @ (p_tracker(t) - p_tracker(0))
    # with C = M @ R_vive(0).T CONSTANT (the per-frame rotations telescope:
    # target_rotation(t) @ M @ R_vive(t).T == M @ R_vive(0).T). Substituting
    # p_tcp = p_tracker - R_vive @ M.T @ r and using C @ R_vive(t) @ M.T ==
    # target_rotation(t) collapses the whole thing to
    #     p_corrected(t) = p_rec(t) - pos_scale * (R_rec(t) - I) @ r
    # i.e. it needs only the recorded rotation and the record-time pos_scale.
    # Verified numerically exact (residual 1.7e-16 m) against a simulated
    # record-then-correct round trip.
    #
    # NOTE the pos_scale here must be the one used at RECORD time, which is
    # why --vive_config should be the same file the demo was recorded with.
    # Demos recorded after the ViveInputDevice fix already have the TCP
    # position baked in and must NOT be corrected again -- pass
    # --lever_arm_correction none for those.
    lever_arm = None
    if args.lever_arm_correction == "none":
        pass
    elif args.lever_arm_correction == "config":
        if config_lever_arm is not None:
            lever_arm = np.array(config_lever_arm, dtype=np.float64)
    else:
        lever_arm = np.array(
            [float(v) for v in args.lever_arm_correction.split(",")], dtype=np.float64
        )
    if lever_arm is not None:
        assert lever_arm.shape == (3,)
        print(
            f"[ReplayUmiOnFairino5] Retroactively correcting the recorded poses "
            f"for a TCP->tracker lever arm of {lever_arm} m "
            f"({np.linalg.norm(lever_arm) * 100:.1f}cm)."
        )

    print(f"[ReplayUmiOnFairino5] Load {args.rmb_filename}")
    with RmbData(args.rmb_filename) as rmb_data:
        umi_pose = rmb_data[args.pose_key][:]  # (T,7): tx,ty,tz,qw,qx,qy,qz
        umi_gripper = rmb_data[DataKey.COMMAND_GRIPPER_JOINT_POS][:]  # (T,1) percent closed
        time_seq = rmb_data[DataKey.TIME][:]

    n_steps = umi_pose.shape[0]
    umi_se3_list = [
        pin.SE3(pin.Quaternion(*umi_pose[t, 3:7]), umi_pose[t, 0:3])
        for t in range(n_steps)
    ]
    if lever_arm is not None:
        # p_corrected(t) = p_rec(t) - pos_scale * (R_rec(t) - I) @ r
        # See the derivation where lever_arm is parsed. Applied before any
        # filtering or retargeting so everything downstream sees TCP poses.
        identity = np.eye(3)
        pre_range = np.ptp(
            np.array([se3.translation for se3 in umi_se3_list]), axis=0
        )
        umi_se3_list = [
            pin.SE3(
                se3.rotation,
                se3.translation - pos_scale * ((se3.rotation - identity) @ lever_arm),
            )
            for se3 in umi_se3_list
        ]
        post_range = np.ptp(
            np.array([se3.translation for se3 in umi_se3_list]), axis=0
        )
        print(
            f"[ReplayUmiOnFairino5] Lever-arm correction: position range "
            f"{np.round(pre_range, 4)} -> {np.round(post_range, 4)} m"
        )
    umi_gripper = list(umi_gripper)

    if args.mirror_exact:
        # Feed the recording mirror's own pose sequence through untouched --
        # see --mirror_exact's help for why each filter would otherwise make
        # the replay diverge from the recording.
        print(
            "[ReplayUmiOnFairino5] --mirror_exact: skipping warmup trim and "
            "glitch/drift filters."
        )
        args.warmup_seconds = 0.0
        args.max_linear_vel = np.inf
        args.max_angular_vel = np.inf
        args.drift_linear_vel = 0.0

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

    if args.sim:
        dry_run = False  # sim has no dry_run concept -- always "runs", just in MuJoCo
        print("[ReplayUmiOnFairino5] SIM MODE: replaying against MuJoCo FR5 (no hardware).")
        env = gym.make(
            "robo_manip_baselines/MujocoFairino5CableEnv-v0",
            render_mode="human",
        )
    else:
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
            gripper_do_close_id=args.gripper_do_close_id,
            gripper_do_open_id=args.gripper_do_open_id,
            dry_run=dry_run,
        )
    env.reset()
    if not args.sim:
        # MuJoCo's reset() already places the arm at init_qpos; move_to_init_pose()
        # is a real-hardware-only slow, safe approach move and isn't defined on the
        # MuJoCo env.
        env.unwrapped.move_to_init_pose()
    elif args.closed_loop:
        # Scene workarounds for the closed-loop tracker ONLY. It re-anchors
        # to measured state every iteration, so any contact that physically
        # resists the commanded motion stalls it permanently (confirmed:
        # error plateaus at a fixed nonzero value even with max_iters raised
        # to 3000, with min singular value ~0.24 throughout -- a rigid
        # contact equilibrium, not a singularity). Moving push_block out of
        # the way and lowering the table reduces, though does not eliminate
        # (a forearm_link/wrist2_link self-collision remains), how often
        # that happens.
        #
        # The default OPEN-LOOP replay deliberately does NOT do any of this:
        # it commands a kinematically-integrated joint trajectory without
        # consulting measured state, exactly like the recording mirror, so
        # contacts perturb it no more than they perturbed the recording --
        # and keeping the scene byte-identical to the recording's is what
        # makes the replay video directly comparable to the mirror video.
        push_block_body = env.unwrapped.model.body("push_block")
        qpos_adr = env.unwrapped.model.jnt_qposadr[push_block_body.jntadr[0]]
        env.unwrapped.data.qpos[qpos_adr : qpos_adr + 3] = [10.0, 10.0, 10.0]
        qvel_adr = env.unwrapped.model.jnt_dofadr[push_block_body.jntadr[0]]
        env.unwrapped.data.qvel[qvel_adr : qvel_adr + 6] = 0.0

        # The table body's two geoms are unnamed in the XML, so they're
        # found by body membership and told apart by z-size (the base is the
        # tall one, the plate is the thin one on top). Done at runtime
        # rather than by editing the shared XML asset, which
        # MujocoFairino5CableEnv's actual cable task also depends on.
        TABLE_LOWER_M = 0.1
        table_body_id = env.unwrapped.model.body("table").id
        table_geom_ids = [
            i
            for i in range(env.unwrapped.model.ngeom)
            if env.unwrapped.model.geom_bodyid[i] == table_body_id
        ]
        table_geom_ids.sort(
            key=lambda i: env.unwrapped.model.geom_size[i, 2], reverse=True
        )
        base_id, plate_id = table_geom_ids[0], table_geom_ids[1]
        env.unwrapped.model.geom_size[base_id, 2] -= TABLE_LOWER_M / 2
        env.unwrapped.model.geom_pos[base_id, 2] -= TABLE_LOWER_M / 2
        env.unwrapped.model.geom_pos[plate_id, 2] -= TABLE_LOWER_M

        mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
        print(
            f"[ReplayUmiOnFairino5] SIM MODE (closed-loop): relocated "
            f"push_block out of the workspace, and lowered the table top by "
            f"{TABLE_LOWER_M}m."
        )

    arm_manager = ArmManager(env.unwrapped, env.unwrapped.body_config_list[0])
    init_se3 = arm_manager.current_se3.copy()

    # Both rotation AND translation are retargeted as pure EEF/TCP-LOCAL
    # reproductions, matching how ViveInputDevice.set_command_data itself
    # drives a robot in live single-robot teleop -- see the module
    # docstring's rotation note for the full derivation and fix history
    # (including why vive_world_to_base_frame_rotation, needed by an earlier,
    # wrong version of this retarget, is not used at all here: once both
    # channels are local-frame reproductions, retargeting is inherently
    # independent of how the room relates to any robot's base).
    #
    # Rotation: delta_umi_rotation (~= umi_se3_t.rotation, since
    # umi_se3_0.rotation ~= I) is already the EEF-local delta
    # ViveInputDevice.set_command_data composed via right-multiply onto
    # eef_se3_at_enable.rotation -- reapplied here onto FR5's own init
    # orientation the same way.
    #
    # Translation: umi_se3_t.translation is an ACCUMULATION of per-frame
    # EEF-local increments, each rotated into the (evolving) target frame by
    # THAT frame's own target_rotation before being added on -- see
    # ViveInputDevice.set_command_data's translation comment ("push the
    # tracker forward always means push the TCP forward along its own
    # CURRENT Z, even while simultaneously rotating"). So it cannot be
    # retargeted as one batch delta from t=0 rotated by a fixed matrix (the
    # earlier, wrong version of this code): un-rotating each frame's
    # increment by THAT frame's own umi_se3_t.rotation recovers the local
    # increment, which is then re-accumulated through FR5's OWN (already
    # correctly retargeted) target_rotation at each frame -- reproducing
    # "push forward along the TCP's own current, possibly-tilted, Z" on FR5
    # exactly as it happened on the UMI rig, frame by frame. This only needs
    # the recorded pose sequence, not live UMI/Vive hardware, so it works
    # identically for offline replay.
    # EpisodeRelativeEefPoseRetargeter.to_absolute() implements exactly this
    # loop's math (right-multiply rotation composition onto init_se3, and
    # per-frame local-increment translation re-accumulation) -- factored out
    # so the same, single validated implementation is shared with online
    # policy rollout (see RolloutBase.get_measured_data_for_policy/
    # set_command_data in common/base/RolloutBase.py), rather than
    # maintaining two copies that could silently drift apart.
    retargeter = EpisodeRelativeEefPoseRetargeter(init_se3, pos_scale=pos_scale)
    target_se3_list = [retargeter.to_absolute(umi_se3_t) for umi_se3_t in umi_se3_list]

    # Put the trajectory on the env's OWN control-period time grid before
    # replaying it. The recording's frame rate is not env.dt: one recorded
    # frame is one teleop loop iteration, and that loop runs at whatever rate
    # the UMI rig allows (measured ~8.4 Hz / ~119 ms per frame, dominated by
    # Vive tracker reads, and jittery -- 96 to 205 ms). Replaying one recorded
    # frame per env.step() therefore compresses ~119 ms of real motion into
    # env.dt (32 ms), i.e. plays the demo back ~3.7x too fast, and unevenly
    # (3.0x to 6.4x, tracking the recording's own loop jitter).
    #
    # That is not cosmetic: measured on real data, it turns commanded joint
    # speeds of at most 124 deg/s into 465 deg/s -- over 15x
    # RealFairino5EnvBase's own 30 deg/s joint_vel_limit, exceeded on 30% of
    # steps. On hardware overwrite_command_for_safety would clamp those, but
    # a clamped command is a command the arm cannot follow, so the replay
    # would silently stop matching the recording exactly where it moves
    # fastest.
    #
    # Resampling by TIME fixes both: sample_checkpoints() interpolates the
    # trajectory evenly in recorded time (screw interpolation, see
    # interpolate_se3), so consuming one resampled frame per env.step()
    # advances exactly env.dt of recorded motion per env.dt of control time
    # -- real-time playback, with the recording's jitter smoothed out rather
    # than turned into speed spikes.
    # Recorded time consumed per control step. Dividing by time_scale means
    # time_scale=2 advances only half as much recorded motion per step, so
    # the demo takes twice as long and every commanded joint speed halves.
    replay_dt = env.unwrapped.dt / max(args.time_scale, 1e-3)
    target_se3_list, resampled_gripper, _resampled_time = sample_checkpoints(
        target_se3_list, list(umi_gripper), time_seq, 1.0 / replay_dt
    )
    umi_gripper = resampled_gripper
    recorded_duration = float(time_seq[-1]) - float(time_seq[0])
    print(
        f"[ReplayUmiOnFairino5] Resampled {n_steps} recorded frames "
        f"({recorded_duration:.2f}s at ~{(n_steps - 1) / max(recorded_duration, 1e-9):.1f} Hz) "
        f"to {len(target_se3_list)} frames on the env's {env.unwrapped.dt * 1000:.0f}ms "
        f"control period"
        + (
            ""
            if args.time_scale == 1.0
            else f", slowed by --time_scale={args.time_scale}"
        )
        + f" -> replay takes {len(target_se3_list) * env.unwrapped.dt:.2f}s."
    )

    max_joint_step_deg = (
        None if args.max_joint_step_deg <= 0 else args.max_joint_step_deg
    )
    rot_tol = np.deg2rad(args.rot_tol_deg)
    max_step_rot = np.deg2rad(args.max_step_rot_deg)
    integral_clamp_rot = np.deg2rad(args.integral_clamp_rot_deg)

    # Capture the replay as video (sim only) so it can be compared frame by
    # frame against the recording's own *_mujoco_mirror.mp4 -- the whole
    # point of the open-loop design is that these two should match.
    video_frames = []
    video_camera = None
    if args.sim and args.save_video != "":
        camera_names = env.unwrapped.camera_names
        video_camera = env.unwrapped.cameras[
            "front" if "front" in camera_names else camera_names[0]
        ]

    def capture_frame():
        if video_camera is not None:
            video_camera["viewer"].make_context_current()
            video_frames.append(
                video_camera["viewer"].render(
                    render_mode="rgb_array", camera_id=video_camera["id"]
                )
            )

    replay_start_time = time.time()
    obs = env.unwrapped._get_obs()
    reached_end = False
    n_iters = 0
    final_path_idx = 0
    log = {
        "min_singular_value": [],
        "command_joint_deg": [],
        "measured_joint_deg": [],
        "measured_se3": [],
        "target_se3": [],
        "path_idx": [],
        "step_time": [],
    }
    if args.closed_loop:
        dense_se3_list, dense_gripper_list, _dense_time_seq = sample_checkpoints(
            target_se3_list, list(umi_gripper), time_seq, args.checkpoint_hz
        )
        n_path = len(dense_se3_list)
        print(
            f"[ReplayUmiOnFairino5] CLOSED-LOOP mode: densified to {n_path} "
            f"path points (~{args.checkpoint_hz} Hz) from {n_steps} recorded "
            f"frames -- tracking with pure pursuit "
            f"(lookahead={args.lookahead_points} points, "
            f"search_window={args.search_window}, max {args.max_total_iters} "
            "iterations total)..."
        )
    else:
        n_path = len(target_se3_list)
        print(
            f"[ReplayUmiOnFairino5] OPEN-LOOP mode (default): replaying "
            f"{n_path} retargeted frames with {args.ik_steps_per_frame} IK "
            "step(s) each, exactly reproducing the recording mirror's own "
            "control loop..."
        )
    try:
        if args.closed_loop:
            reached_end, n_iters, final_path_idx, _final_se3, obs, log = (
                track_path_pure_pursuit(
                    env,
                    arm_manager,
                    dense_se3_list,
                    dense_gripper_list,
                    obs,
                    replay_start_time,
                    lookahead_points=args.lookahead_points,
                    search_window=args.search_window,
                    pos_tol=args.pos_tol,
                    rot_tol=rot_tol,
                    kp_pos=args.kp_pos,
                    ki_pos=args.ki_pos,
                    kp_rot=args.kp_rot,
                    ki_rot=args.ki_rot,
                    max_step_pos=args.max_step_pos,
                    max_step_rot=max_step_rot,
                    max_total_iters=args.max_total_iters,
                    integral_clamp_pos=args.integral_clamp_pos,
                    integral_clamp_rot=integral_clamp_rot,
                    low_manip_threshold=args.low_manip_threshold,
                    max_extra_damping=args.max_extra_damping,
                    max_joint_step_deg=max_joint_step_deg,
                )
            )
        else:
            n_iters, _final_se3, obs, log = track_path_open_loop(
                env,
                arm_manager,
                target_se3_list,
                list(umi_gripper),
                obs,
                replay_start_time,
                ik_steps_per_frame=args.ik_steps_per_frame,
                frame_callback=capture_frame,
            )
            reached_end = True
            final_path_idx = n_path - 1
    except KeyboardInterrupt:
        print("[ReplayUmiOnFairino5] Interrupted.")
    finally:
        if len(video_frames) > 0:
            video_path = args.save_video
            if video_path is None:
                base, _ext = os.path.splitext(os.path.normpath(args.rmb_filename))
                video_path = f"{base}_replay_sim.mp4"
            # One captured frame per env.step(), and after resampling each
            # env.step() is exactly env.dt of real time, so 1/env.dt is the
            # true playback rate. (With --time_scale != 1 the video plays at
            # the same slowed/sped-up rate the arm actually moved at, which
            # is what you want when checking the motion.)
            fps = 1.0 / env.unwrapped.dt
            videoio.videosave(video_path, np.array(video_frames), fps=fps)
            print(
                f"[ReplayUmiOnFairino5] Saved replay video ({len(video_frames)} "
                f"frames, {fps:.2f} fps) to {video_path}"
            )
        try:
            env.close()
        except AttributeError:
            # gymnasium's MujocoEnv.close() unconditionally closes its
            # generic viewer, which MujocoEnvBase never creates when frames
            # are rendered per-camera instead (see capture_frame above), so
            # it hits a None. Harmless -- only the cleanup call fails.
            pass
        status = "reached the end of the path" if reached_end else (
            f"*** DID NOT finish -- stopped at path point {final_path_idx}/"
            f"{n_path - 1} after {n_iters} iterations ***"
        )
        print(f"[ReplayUmiOnFairino5] Done. {status}")

    measured_se3_list = log["measured_se3"]
    target_se3_list = log["target_se3"]  # one entry per LOGGED iteration (the carrot at that time)
    command_joint_deg_list = log["command_joint_deg"]
    measured_joint_deg_list = log["measured_joint_deg"]
    min_singular_value_list = log["min_singular_value"]
    step_time_list = log["step_time"]

    # Joint-speed audit of what was actually commanded. This is the number
    # that matters before touching hardware: RealFairino5EnvBase clamps
    # anything faster (overwrite_command_for_safety), and a clamped command
    # is one the arm cannot follow, so exceeding the limit means the replay
    # silently stops matching the recording. --time_scale is the knob to fix
    # it (2.0 halves every speed here).
    if len(command_joint_deg_list) > 1:
        cmd_deg = np.array(command_joint_deg_list)
        joint_speed = np.abs(np.diff(cmd_deg, axis=0)) / env.unwrapped.dt
        limit_deg = np.rad2deg(getattr(env.unwrapped, "joint_vel_limit", np.deg2rad(30)))
        n_over = int((joint_speed > limit_deg).sum())
        print(
            f"[ReplayUmiOnFairino5] Commanded joint speed: max="
            f"{joint_speed.max():.1f} deg/s, mean={joint_speed.mean():.1f} deg/s "
            f"(per-joint max {np.round(joint_speed.max(axis=0), 1)}) vs the "
            f"arm's {limit_deg:.0f} deg/s limit -- "
            f"{n_over}/{joint_speed.size} samples ({100.0 * n_over / joint_speed.size:.1f}%) over."
        )
        if n_over > 0:
            print(
                f"[ReplayUmiOnFairino5] WARNING: commands exceed the joint "
                f"velocity limit. On --real these get clamped, so the arm "
                f"will lag the recorded motion there. Re-run with "
                f"--time_scale {max(2.0, joint_speed.max() / max(limit_deg, 1e-9)):.1f} "
                "to bring them under the limit."
            )

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
