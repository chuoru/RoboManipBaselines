"""Calibrate ViveInputDevice's vive_to_eef_frame_rotation from real tracker motion.

Unlike calibrate_vive_rotation.py (which assumes you hold the tracker in the exact
posture the runtime code will see and derives R purely from that single orientation
snapshot), this script has you physically move the tracker along the directions you
want to call "forward"/"right"/"up", records the real displacement vectors, and
fits the best proper rotation matrix to them via the orthogonal Procrustes problem
(SVD-based Kabsch). This is more robust than reasoning about signs/axes by hand from
qualitative reports, and always outputs a valid (determinant +1) rotation.

What "forward"/"right"/"up" mean in TCP coordinates is robot/mounting-specific --
override --forward_axis/--right_axis/--up_axis (each one of +x/-x/+y/-y/+z/-z, in
the arm's own base frame, matching what a teach pendant would show) if the default
of forward=+z/right=+y/up=-x doesn't match your setup.

Usage:
    python ./teleop/calibrate_vive_axes.py --side right --serial_number LHR-208C8961
"""

import argparse
import time

import gymnasium as gym
import numpy as np
import pinocchio as pin

import robo_manip_baselines  # noqa: F401  (registers gym envs)
from robo_manip_baselines.common import MotionManager

AXIS_VECTORS = {
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
}


def get_eef_rotation(side_idx):
    print("[calibrate] Building kinematic model (dry-run, no hardware connection)...")
    env = gym.make(
        "robo_manip_baselines/RealFairinoDualDemoEnv-v0",
        robot_ip_left="0.0.0.0",
        robot_ip_right="0.0.0.0",
        camera_ids=None,
        gelsight_ids=None,
        dry_run=True,
    )
    env.reset()
    motion_manager = MotionManager(env)
    eef_rotation = motion_manager.body_manager_list[side_idx].current_se3.rotation.copy()
    env.close()
    return eef_rotation


def wait_for_tracker(pysurvive, ctx, serial_number, wait_sec):
    known_serials = set()
    start = time.time()
    while time.time() - start < wait_sec:
        obj = ctx.NextUpdated()
        if obj is not None:
            found_serial = pysurvive.simple_serial_number(obj.ptr)
            if isinstance(found_serial, bytes):
                found_serial = found_serial.decode()
            known_serials.add(found_serial)
            if found_serial == serial_number:
                return obj
        time.sleep(0.01)
    raise RuntimeError(
        f"[calibrate] Tracker with serial '{serial_number}' was not detected "
        f"within {wait_sec}s. Detected serials so far: {sorted(known_serials)}"
    )


def capture_position(tracker_object, label):
    input(f"[calibrate] {label} Press Enter when ready and holding steady.")
    pose, timecode = tracker_object.Pose()
    if timecode <= 0:
        raise RuntimeError(
            "[calibrate] Tracker has no valid pose yet -- make sure it's visible "
            "to both lighthouses and retry."
        )
    return np.array(pose.Pos[:3], dtype=np.float64), pose


def fit_rotation(measured_local, desired_local):
    """Best proper rotation R (measured_local -> desired_local) via Kabsch/SVD.

    measured_local, desired_local: (3, N) arrays of corresponding unit vectors.
    """
    covariance = desired_local @ measured_local.T
    u, _, vt = np.linalg.svd(covariance)
    d = np.sign(np.linalg.det(u @ vt))
    correction = np.diag([1.0, 1.0, d])
    return u @ correction @ vt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=["left", "right"], required=True)
    parser.add_argument(
        "--serial_number",
        required=True,
        help="Tracker's hardware serial number (e.g. LHR-904A2704), from "
        "teleop/check_vive_devices.py",
    )
    parser.add_argument("--forward_axis", default="+z", choices=AXIS_VECTORS.keys())
    parser.add_argument("--right_axis", default="+y", choices=AXIS_VECTORS.keys())
    parser.add_argument("--up_axis", default="-x", choices=AXIS_VECTORS.keys())
    parser.add_argument("--wait_sec", type=float, default=10.0)
    args = parser.parse_args()
    side_idx = 0 if args.side == "left" else 1

    eef_rotation = get_eef_rotation(side_idx)
    print(f"[calibrate] EEF rotation at init pose ({args.side} arm):\n{eef_rotation}\n")

    import pysurvive

    print("[calibrate] Connecting to libsurvive...")
    ctx = pysurvive.SimpleContext([])
    try:
        print(f"[calibrate] Looking for tracker {args.serial_number}...")
        tracker_object = wait_for_tracker(
            pysurvive, ctx, args.serial_number, args.wait_sec
        )
        print("[calibrate] Tracker found.\n")

        p_ref, pose_ref = capture_position(
            tracker_object,
            "Hold the tracker in your normal starting posture (reference point).",
        )
        # This orientation defines the tracker's local axes, exactly like
        # vive_se3_at_enable does in ViveInputDevice.read().
        tracker_ref_rotation = pin.Quaternion(*pose_ref.Rot[:4]).toRotationMatrix()

        local_deltas = []
        for label, axis in (
            ("forward", args.forward_axis),
            ("right", args.right_axis),
            ("up", args.up_axis),
        ):
            p, _ = capture_position(
                tracker_object,
                f"Now move the tracker ~20-30cm in the direction you want to call "
                f"'{label}' ({axis} in TCP coordinates), then hold still.",
            )
            raw_delta_world = p - p_ref
            norm = np.linalg.norm(raw_delta_world)
            if norm < 0.03:
                raise RuntimeError(
                    f"[calibrate] Barely any motion detected for '{label}' "
                    f"({norm * 100:.1f} cm) -- move it further and retry."
                )
            # Project into the tracker's reference-orientation local frame, matching
            # ViveInputDevice's delta_vive_se3.translation.
            local_delta = tracker_ref_rotation.T @ (raw_delta_world / norm)
            local_deltas.append(local_delta)
            print(f"[calibrate]   -> moved {norm * 100:.1f} cm\n")
    except KeyboardInterrupt:
        print("\n[calibrate] Interrupted by Ctrl+C.")
        return
    finally:
        pysurvive.simple_close(ctx.ptr)

    measured_local = np.stack(local_deltas, axis=1)  # (3, 3): columns=forward,right,up

    desired_world = np.stack(
        [AXIS_VECTORS[args.forward_axis], AXIS_VECTORS[args.right_axis], AXIS_VECTORS[args.up_axis]],
        axis=1,
    )
    desired_local = eef_rotation.T @ desired_world

    rotation = fit_rotation(measured_local, desired_local)

    # Report fit quality: angle (deg) between what this R actually produces for each
    # test motion and what was desired.
    achieved_local = rotation @ measured_local
    print("[calibrate] Fit quality (should be close to 0 deg):")
    for label, i in (("forward", 0), ("right", 1), ("up", 2)):
        cos_angle = np.clip(np.dot(achieved_local[:, i], desired_local[:, i]), -1.0, 1.0)
        angle_deg = np.rad2deg(np.arccos(cos_angle))
        print(f"    {label}: {angle_deg:.2f} deg error")

    print(
        f"\n[calibrate] Add this under device_params for the {args.side} arm in "
        "teleop/configs/ViveDual.yaml:\n"
    )
    print("    vive_to_eef_frame_rotation:")
    for row in rotation:
        print(f"      - [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}]")


if __name__ == "__main__":
    main()
