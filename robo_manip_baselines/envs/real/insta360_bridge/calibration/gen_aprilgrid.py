"""Generate a printable AprilGrid calibration target (for Basalt's
basalt_calibrate/basalt_calibrate_imu camera-IMU extrinsic calibration) as a
PNG sized for A4 paper (landscape) at 300 DPI, plus the matching
aprilgrid.json config that must be passed to Basalt via --aprilgrid.

Renders tags itself from t36h11_codes.json (the tag36h11 family's 36-bit
codes, copied verbatim from ethz-asl/kalibr's kalibr_create_target_pdf,
which cites "Codes from AprilTags C++ Library
(http://people.csail.mit.edu/kaess/apriltags/)") using Kalibr's own
generateAprilTag bit-layout/rotation/border convention -- NOT the standard
pre-rendered tag36h11 PNGs from github.com/AprilRobotics/apriltag-imgs (a
1-cell-black-border convention) this script used originally.

That distinction matters: Basalt vendors an independent AprilTag
implementation (thirdparty/apriltag/ethz_apriltag2, from
people.csail.mit.edu/kaess/apriltags/, the same lineage as Kalibr's own
codes/rendering) configured with blackTagBorder=2 (see
thirdparty/apriltag/src/apriltag.cpp's ApriltagDetectorData) -- a 2-cell
black border, not the 1-cell border AprilRobotics/apriltag-imgs uses.
Confirmed against real hardware: a target built from apriltag-imgs PNGs
(readable by cv2.aruco's DICT_APRILTAG_36h11 and the modern
AprilRobotics/apriltag reference library -- both found all 36 tags on a
real captured frame) was detected as exactly ZERO corners by
basalt_calibrate's detect_corners on every single frame of two full
recording sweeps, which is what led to tracing this border-width mismatch.

Usage: print at 100% scale (no "fit to page") on A4 paper, and measure the
actual printed tag size with a ruler afterward -- printer scaling drift
means the nominal size below is a starting point, not something to trust
blindly. Update aprilgrid.json's tagSize to the measured value before
running basalt_calibrate/basalt_calibrate_imu.
"""
import json
import os

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODES_PATH = os.path.join(SCRIPT_DIR, "t36h11_codes.json")
OUT_PNG = os.path.join(SCRIPT_DIR, "aprilgrid_a4.png")
OUT_JSON = os.path.join(SCRIPT_DIR, "aprilgrid.json")

DPI = 300
PAGE_WIDTH_IN = 11.69  # A4 landscape, 297mm
PAGE_HEIGHT_IN = 8.27  # A4 landscape, 210mm

TAG_COLS = 6
TAG_ROWS = 6
# Kalibr's generateAprilTag: total cells across one tag = sqrt(bits) +
# 2*borderBits = 6 + 2*2 = 10 (2-cell border + 6x6 data). tagSize below is
# the FULL tag footprint (border included), matching aprilgrid.json's
# tagSize semantics (Basalt/Kalibr both treat tagSize as the outer edge of
# the black border, not just the data payload).
BORDER_BITS = 2
DATA_BITS_SIDE = 6  # sqrt(36)
TOTAL_CELLS = DATA_BITS_SIDE + 2 * BORDER_BITS  # 10
TAG_SIZE_MM = 25  # nominal; measure the actual print to confirm
TAG_SPACING_RATIO = 0.3  # gap between tags, as a fraction of TAG_SIZE_MM

with open(CODES_PATH) as f:
    TAG_CODES = json.load(f)


def render_tag(tag_id, cell_px, rotation=2):
    """Renders one tag36h11 tile (TOTAL_CELLS x TOTAL_CELLS cells) as a
    uint8 grayscale array, following Kalibr's generateAprilTag exactly:
    2-cell black border ring, 6x6 data bits (bit=0 -> black cell, bit=1 ->
    white/background) rotated 180 degrees (rotation=2, Kalibr's default)
    before placement.
    """
    code = TAG_CODES[tag_id]
    size_px = TOTAL_CELLS * cell_px
    img = np.full((size_px, size_px), 255, dtype=np.uint8)

    border_px = BORDER_BITS * cell_px
    img[0:border_px, :] = 0
    img[-border_px:, :] = 0
    img[:, 0:border_px] = 0
    img[:, -border_px:] = 0

    code_matrix = np.zeros((DATA_BITS_SIDE, DATA_BITS_SIDE), dtype=np.uint8)
    for i in range(DATA_BITS_SIDE):
        for j in range(DATA_BITS_SIDE):
            if not (code & (1 << (DATA_BITS_SIDE * i + j))):
                code_matrix[i, j] = 1
    code_matrix = np.rot90(code_matrix, rotation)

    for i in range(DATA_BITS_SIDE):
        for j in range(DATA_BITS_SIDE):
            if code_matrix[i, j]:
                r0 = (BORDER_BITS + i) * cell_px
                c0 = (BORDER_BITS + j) * cell_px
                img[r0:r0 + cell_px, c0:c0 + cell_px] = 0

    return img


px_w = int(PAGE_WIDTH_IN * DPI)
px_h = int(PAGE_HEIGHT_IN * DPI)
tag_px = int(TAG_SIZE_MM / 25.4 * DPI)
cell_px = tag_px // TOTAL_CELLS
tag_px = cell_px * TOTAL_CELLS  # re-snap so cells divide evenly, no rounding seams
pitch_px = int(tag_px * (1 + TAG_SPACING_RATIO))

grid_w = (TAG_COLS - 1) * pitch_px + tag_px
grid_h = (TAG_ROWS - 1) * pitch_px + tag_px
offset_x = (px_w - grid_w) // 2
offset_y = (px_h - grid_h) // 2

img = np.full((px_h, px_w), 255, dtype=np.uint8)

for row in range(TAG_ROWS):
    for col in range(TAG_COLS):
        tag_id = TAG_COLS * row + col
        tag_bitmap = render_tag(tag_id, cell_px)

        # row 0 is the BOTTOM row in Kalibr's PDF (y-up) convention --
        # flip to the top-down row index a raster image uses (row 0 = top).
        raster_row = TAG_ROWS - 1 - row
        y0 = offset_y + raster_row * pitch_px
        x0 = offset_x + col * pitch_px
        img[y0:y0 + tag_px, x0:x0 + tag_px] = tag_bitmap

cv2.imwrite(OUT_PNG, img)

aprilgrid_config = {
    "tagCols": TAG_COLS,
    "tagRows": TAG_ROWS,
    "tagSize": TAG_SIZE_MM / 1000.0,
    "tagSpacing": TAG_SPACING_RATIO,
}
with open(OUT_JSON, "w") as f:
    json.dump(aprilgrid_config, f, indent=2)

print(f"Generated {px_w}x{px_h}px A4 AprilGrid at {OUT_PNG}, "
      f"{TAG_COLS}x{TAG_ROWS} tags, nominal tag size {TAG_SIZE_MM}mm "
      f"({TOTAL_CELLS}x{TOTAL_CELLS} cells incl. 2-cell border), "
      f"spacing {TAG_SPACING_RATIO}x tag size. Config written to {OUT_JSON} "
      f"-- if the printed tag size differs from {TAG_SIZE_MM}mm after "
      f"measuring, update tagSize in that file (meters) to match before "
      f"calibrating. Convert to PDF for printing with e.g. "
      f"`python3 -c \"from PIL import Image; "
      f"Image.open('aprilgrid_a4.png').save('aprilgrid_a4.pdf', "
      f"'PDF', resolution=300.0)\"`")
