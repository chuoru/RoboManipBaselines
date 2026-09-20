"""Run OpenCV fisheye calibration on the saved calibration frames.

Outputs fx, fy, cx, cy, k1-k4 in the Kannala-Brandt-compatible form
ORB-SLAM3's Insta360_X4.yaml expects (cv2.fisheye uses the same equidistant
model family as KannalaBrandt8, though the distortion polynomial
parameterization isn't numerically identical -- treat these as a much
better starting point than hand-guessed placeholders, not a perfect match).
"""
import glob
import sys

import cv2
import numpy as np

frames_dir = sys.argv[1]
square_size_mm = float(sys.argv[2])
pattern_size = (9, 6)

objp = np.zeros((1, pattern_size[0] * pattern_size[1], 3), np.float64)
objp[0, :, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size_mm

obj_points = []
img_points = []
image_size = None
used_files = []
skipped_files = []

files = sorted(glob.glob(f"{frames_dir}/*.png"))
print(f"Found {len(files)} candidate images")

for f in files:
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if image_size is None:
        image_size = gray.shape[::-1]
    found, corners = cv2.findChessboardCornersSB(
        gray, pattern_size, cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY,
    )
    if not found:
        skipped_files.append(f)
        continue
    obj_points.append(objp)
    img_points.append(np.ascontiguousarray(corners.reshape(1, -1, 2), dtype=np.float64))
    used_files.append(f)

print(f"Usable images (full {pattern_size} pattern found at full res): {len(used_files)}")
print(f"Skipped (pattern not found at full res): {len(skipped_files)}")
for f in skipped_files:
    print(f"  skipped: {f}")

if len(used_files) < 10:
    print("WARNING: fewer than 10 usable images -- calibration may be unreliable.")

# NOTE: passing zero-initialized K/D + CALIB_RECOMPUTE_EXTRINSIC triggers
# cv2's fisheye InitExtrinsics assertion (fabs(norm_u1) > 0) -- verified
# against real hardware. Passing None lets cv2 auto-initialize K/D from the
# point correspondences instead, which works.
calib_flags = cv2.fisheye.CALIB_FIX_SKEW

try:
    rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
        obj_points, img_points, image_size, None, None,
        flags=calib_flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-8),
    )
except cv2.error as e:
    print(f"Calibration FAILED: {e}")
    sys.exit(1)

print("\n=== Calibration result ===")
print(f"RMS reprojection error: {rms:.4f} px "
      f"({'good' if rms < 1.0 else 'high -- results may be unreliable' if rms > 2.0 else 'acceptable'})")
print(f"Image size: {image_size}")
print(f"fx = {K[0,0]:.4f}")
print(f"fy = {K[1,1]:.4f}")
print(f"cx = {K[0,2]:.4f}")
print(f"cy = {K[1,2]:.4f}")
print(f"k1 = {D[0,0]:.6f}")
print(f"k2 = {D[1,0]:.6f}")
print(f"k3 = {D[2,0]:.6f}")
print(f"k4 = {D[3,0]:.6f}")
