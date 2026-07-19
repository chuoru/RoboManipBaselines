"""Calibrate ViveInputDevice's vive_world_to_base_frame_rotation.

This governs translation only (see ViveInputDevice.set_command_data()): the raw
Vive-room-frame position delta (current tracker position minus its position when
teleop was enabled) is rotated by this fixed matrix and added directly onto the
EEF's base-frame position. That keeps "move the tracker forward" meaning "move the
tool forward relative to your body" even while the tool itself is tilted, instead of
tracking the tool's own (possibly tilted) local frame.

Unlike vive_to_eef_frame_rotation (which depends on how you happen to hold the
tracker each session, see calibrate_vive_axes.py / calibrate_vive_rotation.py),
this matrix is a fixed physical relationship between where the lighthouses are
mounted and how the robot base is oriented in the room -- it does not depend on
tracker orientation or on how you hold it, and only needs recalibrating if the
lighthouses or the robot are physically moved.

Since it's a physical fact rather than a preference, the three test directions you
move the tracker along should be real, verifiable directions you already know the
robot base axes point in -- e.g. from watching the arm jog +X/+Y/+Z on the teach
pendant in Base coordinates and noting which way it moves in the room.

Usage:
    python ./teleop/calibrate_vive_world_frame.py --serial_number LHR-208C8961 \\
        --x_direction "upper-backward" --y_direction "lower-backward" --z_direction "right"

(The --*_direction strings are just labels echoed back to you while you move the
tracker -- describe them however makes sense to you.)
"""

import argparse
import time

import numpy as np


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
    return np.array(pose.Pos[:3], dtype=np.float64)


def fit_rotation(measured_room, desired_base):
    """Best proper rotation R (measured_room -> desired_base) via Kabsch/SVD."""
    covariance = desired_base @ measured_room.T
    u, _, vt = np.linalg.svd(covariance)
    d = np.sign(np.linalg.det(u @ vt))
    correction = np.diag([1.0, 1.0, d])
    return u @ correction @ vt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--serial_number",
        required=True,
        help="Tracker's hardware serial number (e.g. LHR-904A2704), from "
        "teleop/check_vive_devices.py. Either tracker can be used -- this "
        "calibrates the room-to-base relationship, not anything tracker-specific.",
    )
    parser.add_argument("--x_direction", default="base +X direction")
    parser.add_argument("--y_direction", default="base +Y direction")
    parser.add_argument("--z_direction", default="base +Z direction")
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

        p_ref = capture_position(
            tracker_object, "Hold the tracker still at a reference point."
        )

        local_deltas = []
        for label, direction in (
            ("base +X", args.x_direction),
            ("base +Y", args.y_direction),
            ("base +Z", args.z_direction),
        ):
            p = capture_position(
                tracker_object,
                f"Now move the tracker ~20-30cm toward '{direction}' "
                f"(this is {label}), then hold still.",
            )
            raw_delta = p - p_ref
            norm = np.linalg.norm(raw_delta)
            if norm < 0.03:
                raise RuntimeError(
                    f"[calibrate] Barely any motion detected for '{direction}' "
                    f"({norm * 100:.1f} cm) -- move it further and retry."
                )
            local_deltas.append(raw_delta / norm)
            print(f"[calibrate]   -> moved {norm * 100:.1f} cm\n")
    except KeyboardInterrupt:
        print("\n[calibrate] Interrupted by Ctrl+C.")
        return
    finally:
        pysurvive.simple_close(ctx.ptr)

    measured_room = np.stack(local_deltas, axis=1)  # (3, 3): columns = X, Y, Z tests
    desired_base = np.eye(3)  # by construction, the 3 tests ARE base +X, +Y, +Z

    rotation = fit_rotation(measured_room, desired_base)

    achieved_base = rotation @ measured_room
    print("[calibrate] Fit quality (should be close to 0 deg):")
    for label, i in (("X", 0), ("Y", 1), ("Z", 2)):
        cos_angle = np.clip(achieved_base[i, i], -1.0, 1.0)
        angle_deg = np.rad2deg(np.arccos(cos_angle))
        print(f"    base {label}: {angle_deg:.2f} deg error")

    print(
        "\n[calibrate] Add this at the top level of teleop/configs/ViveDual.yaml "
        "(shared by both arms unless their base frames are oriented "
        "differently in the room, in which case calibrate per-side and add it "
        "under each device_params instead):\n"
    )
    print("    vive_world_to_base_frame_rotation:")
    for row in rotation:
        print(f"      - [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}]")


if __name__ == "__main__":
    main()
