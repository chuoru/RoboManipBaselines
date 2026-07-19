"""Calibrate ViveInputDevice's vive_to_eef_frame_rotation from real tracker rotation.

vive_to_eef_frame_rotation conjugates a tracker-local *rotation* delta into the
EEF's own local TCP/tool frame (see ViveInputDevice.set_command_data():
adjusted_rotation_delta = R @ delta_vive_rotation @ R.T, then composed onto
eef_se3_at_enable.rotation). Like vive_to_eef_frame_rotation's old role for
translation, this is purely a TCP-local relationship -- it doesn't depend on the
robot's base orientation or current pose, so no forward kinematics or hardware
connection is needed to calibrate it.

Trying to calibrate this by just holding the tracker in "the same orientation as
the gripper" (the old approach) is hard to get precise -- a few degrees of
mismatch can visibly cross-couple axes (e.g. yaw motion coming out as pitch). This
script instead has you physically ROTATE the tracker about the three axes you
want to call "roll"/"pitch"/"yaw", measures the real rotation axis for each, and
fits the best proper rotation matrix via the orthogonal Procrustes problem
(SVD-based Kabsch) -- the same robust approach used for translation in
calibrate_vive_world_frame.py.

--roll_axis/--pitch_axis/--yaw_axis (each one of +x/-x/+y/-y/+z/-z) say what each
test rotation should correspond to in TCP-local coordinates. Defaults assume a TCP
frame where +z is forward, +y is left, +x is down: roll (twisting the wrist like a
screwdriver, about the forward axis) = +z; pitch (nodding, about the left/right
axis) = -y; yaw (turning, about the up/down axis) = -x. Override them if your
robot's TCP convention differs, or if these labels don't match how you think about
the three rotations.

Usage:
    python ./teleop/calibrate_vive_rotation.py --side right --serial_number LHR-208C8961
"""

import argparse
import time

import numpy as np
import pinocchio as pin

AXIS_VECTORS = {
    "+x": np.array([1.0, 0.0, 0.0]),
    "-x": np.array([-1.0, 0.0, 0.0]),
    "+y": np.array([0.0, 1.0, 0.0]),
    "-y": np.array([0.0, -1.0, 0.0]),
    "+z": np.array([0.0, 0.0, 1.0]),
    "-z": np.array([0.0, 0.0, -1.0]),
}


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


def capture_rotation(tracker_object, label):
    input(f"[calibrate] {label} Press Enter when ready and holding steady.")
    pose, timecode = tracker_object.Pose()
    if timecode <= 0:
        raise RuntimeError(
            "[calibrate] Tracker has no valid pose yet -- make sure it's visible "
            "to both lighthouses and retry."
        )
    # libsurvive quaternions are stored as (w, x, y, z).
    return pin.Quaternion(*pose.Rot[:4]).toRotationMatrix()


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
    parser.add_argument("--roll_axis", default="+z", choices=AXIS_VECTORS.keys())
    parser.add_argument("--pitch_axis", default="-y", choices=AXIS_VECTORS.keys())
    parser.add_argument("--yaw_axis", default="-x", choices=AXIS_VECTORS.keys())
    parser.add_argument("--wait_sec", type=float, default=10.0)
    args = parser.parse_args()

    import pysurvive

    print("[calibrate] Connecting to libsurvive...")
    ctx = pysurvive.SimpleContext([])
    try:
        print(f"[calibrate] Looking for tracker {args.serial_number}...")
        tracker_object = wait_for_tracker(
            pysurvive, ctx, args.serial_number, args.wait_sec
        )
        print("[calibrate] Tracker found.\n")

        r_ref = capture_rotation(
            tracker_object,
            "Hold the tracker in your normal starting posture (reference point).",
        )

        local_axes = []
        for label, axis, instruction in (
            ("roll", args.roll_axis, "twist it about its own forward axis, like turning a screwdriver"),
            ("pitch", args.pitch_axis, "tilt/nod it, like nodding your head"),
            ("yaw", args.yaw_axis, "turn it side to side, like shaking your head"),
        ):
            r_i = capture_rotation(
                tracker_object,
                f"Now rotate the tracker ~45-90deg for '{label}' (TCP {axis}) -- "
                f"{instruction} -- then hold still.",
            )
            delta_local = r_ref.T @ r_i
            rotvec = pin.log3(delta_local)
            angle = np.linalg.norm(rotvec)
            if np.rad2deg(angle) < 15.0:
                raise RuntimeError(
                    f"[calibrate] Barely any rotation detected for '{label}' "
                    f"({np.rad2deg(angle):.1f} deg) -- rotate it further and retry."
                )
            local_axes.append(rotvec / angle)
            print(f"[calibrate]   -> rotated {np.rad2deg(angle):.1f} deg\n")
    except KeyboardInterrupt:
        print("\n[calibrate] Interrupted by Ctrl+C.")
        return
    finally:
        pysurvive.simple_close(ctx.ptr)

    measured_local = np.stack(local_axes, axis=1)  # (3, 3): columns = roll,pitch,yaw

    # Target axes directly in TCP-local coordinates -- vive_to_eef_frame_rotation's
    # output already *is* the TCP-local frame, no base-frame/FK conversion needed.
    desired_local = np.stack(
        [
            AXIS_VECTORS[args.roll_axis],
            AXIS_VECTORS[args.pitch_axis],
            AXIS_VECTORS[args.yaw_axis],
        ],
        axis=1,
    )

    rotation = fit_rotation(measured_local, desired_local)

    achieved_local = rotation @ measured_local
    print("[calibrate] Fit quality (should be close to 0 deg):")
    for label, i in (("roll", 0), ("pitch", 1), ("yaw", 2)):
        cos_angle = np.clip(np.dot(achieved_local[:, i], desired_local[:, i]), -1.0, 1.0)
        angle_deg = np.rad2deg(np.arccos(cos_angle))
        print(f"    {label}: {angle_deg:.2f} deg error")

    print(
        f"\n[calibrate] Add this under device_params for the {args.side} arm in "
        "teleop/configs/ViveDual.yaml (as a sibling of device_params, NOT nested "
        "inside it):\n"
    )
    print("  vive_to_eef_frame_rotation:")
    for row in rotation:
        print(f"    - [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}]")


if __name__ == "__main__":
    main()
