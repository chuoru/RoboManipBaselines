"""Save calibration frames from insta360_bridge's socket, cropped to the
front lens (matching main.cc's default SLAM crop: left frame.rows-wide
square of the dual-fisheye frame), auto-selecting frames where a
9x6-internal-corner checkerboard is clearly detected, spaced out in time
so the user has room to move the board between saves.

Shows a live cv2.imshow preview (checkerboard corners drawn in green when
found) so the user can see what the camera sees and aim it. Press 'q' in
the preview window to stop early.
"""
import os
import socket
import sys
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# common/utils is four levels up from this calibration/ directory.
sys.path.insert(
    0, os.path.normpath(os.path.join(_HERE, "..", "..", "..", "..", "common", "utils"))
)
from Insta360Protocol import read_message

sock_path = sys.argv[1]
out_dir = sys.argv[2]
duration = float(sys.argv[3]) if len(sys.argv) > 3 else 150.0
target_count = int(sys.argv[4]) if len(sys.argv) > 4 else 30
use_back_lens = len(sys.argv) > 5 and sys.argv[5] == "back"

os.makedirs(out_dir, exist_ok=True)

PATTERN_SIZE = (9, 6)
MIN_SAVE_INTERVAL_SEC = 1.5
PREVIEW_SIZE = 700  # display window side length in pixels
# Radial coverage bands (fraction of max radius from image center, measured
# by the FARTHEST detected corner -- not the centroid, since what actually
# constrains the distortion polynomial at extreme incidence angles is how
# close ANY point gets to the edge) and how many saves each band is
# allowed. Verified against real hardware, twice: (1) centrally-clustered
# captures make cv2.fisheye.calibrate diverge outright (300+ px RMS); (2)
# even with centroids spread to the old center/mid/outer bands (up to 100%
# by centroid, max single-corner radius 93%), the fitted polynomial was
# only monotonic (physically valid) up to ~47 degrees incidence, then
# diverged to nonsense (-4325 degrees at 90 degrees) -- this crashed
# ORB_SLAM3 (segfault in libORB_SLAM3.so) during map init once real
# tracking hit pixels past that angle. This band scheme pushes much harder
# toward the true edge (EXTREME: 78-98% of radius) and away from CENTER,
# to actually constrain the polynomial's high-angle behavior.
BAND_EDGES = [0.0, 0.25, 0.50, 0.78, 0.98]  # center / mid / outer / extreme
BAND_CAPS = [3, 5, 8, 14]  # heavily biased toward extreme

s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.settimeout(duration + 5)
s.connect(sock_path)

win_name = "Calibration preview (front lens)" if not use_back_lens else "Calibration preview (back lens)"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win_name, PREVIEW_SIZE, PREVIEW_SIZE)

saved = 0
band_counts = [0] * (len(BAND_EDGES) - 1)
last_save_t = 0.0
t_end = time.time() + duration
center_xy = np.array([PREVIEW_SIZE / 2, PREVIEW_SIZE / 2])
max_r_px = PREVIEW_SIZE / 2
band_names = ["CENTER", "MID", "OUTER", "EXTREME"]
print(f"Live preview open ('{win_name}'). Move the checkerboard around -- "
      f"the guide rings show center/mid/outer/extreme zones. EXTREME "
      f"(right up near the black border, board still fully visible) is "
      f"heavily prioritized -- push as close to the edge as you can while "
      f"keeping the whole board in view. Saves ~every {MIN_SAVE_INTERVAL_SEC}s "
      f"when detected in a zone that still needs samples (green overlay = "
      f"detected). Press 'q' to stop early.")

try:
    while time.time() < t_end and saved < target_count:
        try:
            msg = read_message(s)
        except socket.timeout:
            break
        if msg["type"] != "frame":
            continue
        frame = msg["frame"]  # (h, w, 3) RGB, full dual-fisheye
        h = frame.shape[0]
        lens_crop = frame[:, -h:] if use_back_lens else frame[:, :h]

        # Detect on a small downscaled copy, not the full 1920x1920 crop --
        # findChessboardCornersSB(EXHAUSTIVE) on the full-res image was the
        # actual FPS bottleneck (verified: camera/bridge deliver frames much
        # faster than this could process them). The full-res crop is only
        # used for the final saved image; detection here just gates *when*
        # to save, so a downscaled, imprecise detection is fine.
        small_bgr = cv2.resize(cv2.cvtColor(lens_crop, cv2.COLOR_RGB2BGR),
                                (PREVIEW_SIZE, PREVIEW_SIZE))
        small_gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)
        # findChessboardCornersSB (not the classic findChessboardCorners) --
        # verified against real hardware that the classic detector fails on
        # this lens's fisheye distortion even when the board is clearly
        # visible, while SB succeeds.
        found, corners = cv2.findChessboardCornersSB(
            small_gray, PATTERN_SIZE, cv2.CALIB_CB_ACCURACY,
        )

        display = small_bgr.copy()
        # Guide rings at each band boundary, so the user can see the
        # center/mid/outer zones directly instead of guessing.
        for frac in BAND_EDGES[1:-1]:
            cv2.circle(display, tuple(center_xy.astype(int)),
                       int(frac * max_r_px), (255, 255, 0), 2)

        band_idx = None
        if found:
            pts = corners.reshape(-1, 2)
            max_r_frac = np.linalg.norm(pts - center_xy, axis=1).max() / max_r_px
            for i in range(len(BAND_EDGES) - 1):
                if BAND_EDGES[i] <= max_r_frac < BAND_EDGES[i + 1]:
                    band_idx = i
                    break
            if band_idx is None:  # beyond the outermost edge -- clamp to EXTREME
                band_idx = len(band_names) - 1
            cv2.drawChessboardCorners(display, PATTERN_SIZE, corners, found)

        counts_str = "/".join(f"{band_names[i]}:{band_counts[i]}/{BAND_CAPS[i]}"
                               for i in range(len(band_counts)))
        cv2.putText(display, f"saved {saved}/{target_count}  {counts_str}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if found else (0, 0, 255), 2)
        cv2.imshow(win_name, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Stopped by user.")
            break

        now = time.time()
        if (not found or now - last_save_t < MIN_SAVE_INTERVAL_SEC
                or band_counts[band_idx] >= BAND_CAPS[band_idx]):
            continue

        out_path = os.path.join(out_dir, f"calib_{saved:03d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(lens_crop, cv2.COLOR_RGB2BGR))
        saved += 1
        band_counts[band_idx] += 1
        last_save_t = now
        print(f"[{saved}/{target_count}] saved {out_path} (zone={band_names[band_idx]})")
except ConnectionError as e:
    print(f"ConnectionError: {e}")

cv2.destroyAllWindows()
print(f"Done. {saved} calibration frames saved to {out_dir}")
