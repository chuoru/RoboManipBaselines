"""Generate a printable checkerboard calibration target as a PNG sized for
A3 paper at 300 DPI. 9x6 internal corners (10x7 squares), a common,
well-tested OpenCV calibration pattern size.

Usage: print at 100% scale (no "fit to page") on A3 paper, and measure the
actual printed square size with a ruler afterward -- printer scaling drift
means the nominal size below is a starting point, not something to trust
blindly.
"""
import os

import cv2
import numpy as np

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "checkerboard_a3.png")

DPI = 300
A3_WIDTH_IN = 16.53  # 420mm
A3_HEIGHT_IN = 11.69  # 297mm (landscape)
SQUARES_X = 10
SQUARES_Y = 7
SQUARE_SIZE_MM = 35  # nominal; measure the actual print to confirm

px_w = int(A3_WIDTH_IN * DPI)
px_h = int(A3_HEIGHT_IN * DPI)
square_px = int(SQUARE_SIZE_MM / 25.4 * DPI)

board_w = SQUARES_X * square_px
board_h = SQUARES_Y * square_px
offset_x = (px_w - board_w) // 2
offset_y = (px_h - board_h) // 2

img = np.full((px_h, px_w), 255, dtype=np.uint8)
for i in range(SQUARES_Y):
    for j in range(SQUARES_X):
        if (i + j) % 2 == 0:
            y0 = offset_y + i * square_px
            y1 = y0 + square_px
            x0 = offset_x + j * square_px
            x1 = x0 + square_px
            img[y0:y1, x0:x1] = 0

cv2.imwrite(OUT_PATH, img)
print(f"Generated {px_w}x{px_h}px A3 checkerboard at {OUT_PATH}, "
      f"{SQUARES_X}x{SQUARES_Y} squares ({SQUARES_X-1}x{SQUARES_Y-1}={((SQUARES_X-1)*(SQUARES_Y-1))} internal corners), "
      f"nominal square size {SQUARE_SIZE_MM}mm. Convert to PDF for printing "
      f"with e.g. `python3 -c \"from PIL import Image; "
      f"Image.open('checkerboard_a3.png').save('checkerboard_a3.pdf', "
      f"'PDF', resolution=300.0)\"`")
