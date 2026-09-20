"""Live AprilTag-detection overlay for aiming during a camera-IMU
calibration recording (see README.md's Calibration section).

Unlike capture_calib_frames.py, this doesn't save anything -- the actual
recording is insta360_bridge's own --record (run alongside this, same
--lens), which is what basalt_calibrate/basalt_calibrate_imu will consume
after conversion. This script is purely a live preview so you can see
which tags are actually being detected while aiming/moving the target,
the same way capture_calib_frames.py overlays checkerboard corners.

Run alongside (not instead of) `insta360_bridge --record ...`:
    python3 calibration/live_aprilgrid_preview.py /tmp/insta360_calib.sock
        [front|back]

Uses aprilgrid_2cell_detector.py, a small custom detector matching
gen_aprilgrid.py's 2-cell black border convention -- NOT cv2.aruco, which
(like the modern AprilRobotics/apriltag reference library) only recognizes
the standard 1-cell border and detects nothing on this target at all
(confirmed empirically). See aprilgrid_2cell_detector.py's module
docstring for why a custom detector was needed and how it works; verified
against synthetic renders (all 36 tags, all 4 rotations, and the full
rendered sheet) before wiring in here.

Press 'q' in the preview window to stop.
"""
import os
import socket
import sys

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
# common/utils is four levels up from this calibration/ directory.
sys.path.insert(
    0, os.path.normpath(os.path.join(_HERE, "..", "..", "..", "..", "common", "utils"))
)
from Insta360Protocol import read_message

# aprilgrid_2cell_detector lives next to this script.
sys.path.insert(0, _HERE)
import aprilgrid_2cell_detector as apriltag_detector

sock_path = sys.argv[1]
use_back_lens = len(sys.argv) > 2 and sys.argv[2] == "back"

PREVIEW_SIZE = 700

s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.settimeout(30.0)
s.connect(sock_path)

win_name = f"AprilGrid preview ({'back' if use_back_lens else 'front'} lens)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, PREVIEW_SIZE, PREVIEW_SIZE)

print(f"Live preview open ('{win_name}'). Detected tags are outlined in "
      f"green with their ID labeled -- move the AprilGrid (or the camera) "
      f"so tags are detected across the whole frame, with real rotation on "
      f"all three axes (not just translation -- that's what makes the "
      f"camera-IMU extrinsic observable). This does NOT save anything --"
      f"run insta360_bridge with --record (same --lens) alongside this for "
      f"the actual calibration recording. Press 'q' to stop.")

try:
    while True:
        try:
            msg = read_message(s)
        except socket.timeout:
            print("No frames received for 30s, stopping.")
            break
        if msg["type"] != "frame":
            continue
        frame = msg["frame"]  # (h, w, 3) RGB, full dual-fisheye
        h = frame.shape[0]
        lens_crop = frame[:, -h:] if use_back_lens else frame[:, :h]

        small_bgr = cv2.resize(cv2.cvtColor(lens_crop, cv2.COLOR_RGB2BGR),
                                (PREVIEW_SIZE, PREVIEW_SIZE))
        gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
        detections = apriltag_detector.detect(gray)

        display = small_bgr.copy()
        for tag_id, corners in detections:
            pts = corners.astype(int)
            cv2.polylines(display, [pts], True, (0, 255, 0), 2)
            center = pts.mean(axis=0).astype(int)
            cv2.putText(display, str(tag_id), tuple(center),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        n_detected = len(detections)
        cv2.putText(display, f"{n_detected}/36 tags detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if n_detected >= 4 else (0, 0, 255), 2)
        cv2.imshow(win_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Stopped by user.")
            break
except ConnectionError as e:
    print(f"ConnectionError: {e}")

cv2.destroyAllWindows()
