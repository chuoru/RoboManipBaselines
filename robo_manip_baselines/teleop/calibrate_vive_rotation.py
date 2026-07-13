"""Calibrate ViveInputDevice's vive_to_eef_frame_rotation for one arm.

Why this is needed: ViveInputDevice.set_command_data() (see teleop/ViveInputDevice.py)
computes the tracker's motion relative to the tracker's OWN orientation at the moment
teleop was enabled, then rotates that delta by vive_to_eef_frame_rotation before
applying it as a motion relative to the EEF's OWN orientation at that same moment. For
"move the tracker this way" to consistently mean "move the gripper the equivalent way"
regardless of how the tracker happens to be held, the two enable-time orientations
must be related by a fixed rotation R:

    R = (EEF rotation at the arm's init pose)^T @ (tracker rotation as held at the
         start of teleop)

This holds as a constant (rather than something that has to be recalibrated every
session) because the arm always begins teleop from the same init pose (see
MoveToInitPhase in envs/operation/OperationRealFairinoDualDemo.py) and because you
should hold the tracker the same way each time you begin teleop.

This script computes R using the robot's actual forward kinematics (no hardware
connection needed -- it builds the env with dry_run=True) and the tracker's live
orientation from libsurvive, and prints the result ready to paste into
teleop/configs/ViveDual.yaml (or Vive.yaml).

Usage:
    python ./teleop/calibrate_vive_rotation.py --side left --serial_number LHR-904A2704

Caveat: this assumes the robot base frame's axes and the lighthouses' world frame
axes are roughly aligned the way you intuitively expect (e.g. both "facing" the
workspace the same way). If motion still looks off after applying the printed R,
that residual mismatch is between those two frames, not between the tracker and the
EEF, and needs a manual correction on top of this result.
"""

import argparse
import time

import gymnasium as gym
import numpy as np
import pinocchio as pin

import robo_manip_baselines  # noqa: F401  (registers gym envs)
from robo_manip_baselines.common import MotionManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=["left", "right"], required=True)
    parser.add_argument(
        "--serial_number",
        required=True,
        help="Tracker's hardware serial number (e.g. LHR-904A2704), from "
        "teleop/check_vive_devices.py",
    )
    parser.add_argument(
        "--wait_sec",
        type=float,
        default=10.0,
        help="How long to wait for the tracker to be detected after you press Enter",
    )
    args = parser.parse_args()
    side_idx = 0 if args.side == "left" else 1

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
    print(f"[calibrate] EEF rotation at init pose ({args.side} arm):\n{eef_rotation}\n")

    import pysurvive

    print("[calibrate] Connecting to libsurvive...")
    ctx = pysurvive.SimpleContext([])
    try:
        print(
            f"[calibrate] Hold the tracker (serial {args.serial_number}) the SAME "
            "way you will hold it right before starting real teleop -- this is the "
            "posture that gets captured as the anchor pose every time teleop is "
            "enabled.\nPress Enter when you're holding it steady in that posture."
        )
        input()

        known_serials = set()
        tracker_object = None
        start = time.time()
        while time.time() - start < args.wait_sec:
            obj = ctx.NextUpdated()
            if obj is not None:
                serial_number = pysurvive.simple_serial_number(obj.ptr)
                if isinstance(serial_number, bytes):
                    serial_number = serial_number.decode()
                known_serials.add(serial_number)
                if serial_number == args.serial_number:
                    tracker_object = obj
                    break
            time.sleep(0.01)

        if tracker_object is None:
            raise RuntimeError(
                f"[calibrate] Tracker with serial '{args.serial_number}' was not "
                f"detected within {args.wait_sec}s. Detected serials so far: "
                f"{sorted(known_serials)}"
            )

        pose, timecode = tracker_object.Pose()
        if timecode <= 0:
            raise RuntimeError(
                "[calibrate] Tracker was detected but has no valid pose yet -- "
                "make sure it's visible to both lighthouses and retry."
            )

        # libsurvive quaternions are stored as (w, x, y, z), matching
        # MathUtils.py's convention elsewhere in this codebase.
        tracker_rotation = pin.Quaternion(*pose.Rot[:4]).toRotationMatrix()
    except KeyboardInterrupt:
        print("\n[calibrate] Interrupted by Ctrl+C.")
        return
    finally:
        # Must run on every exit path (including Ctrl+C/errors above), or
        # libsurvive's background USB polling thread keeps the process alive.
        pysurvive.simple_close(ctx.ptr)

    print(f"[calibrate] Tracker rotation:\n{tracker_rotation}\n")

    rotation = eef_rotation.T @ tracker_rotation

    print(
        f"[calibrate] Add this under device_params for the {args.side} arm in "
        "teleop/configs/ViveDual.yaml:\n"
    )
    print("    vive_to_eef_frame_rotation:")
    for row in rotation:
        print(f"      - [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}]")


if __name__ == "__main__":
    main()
