"""Minimal AprilTag (tag36h11, 2-cell black border) detector, for live
aiming feedback only -- NOT used by the actual calibration (Basalt's own
vendored ethz_apriltag2 detector does that, see gen_aprilgrid.py's
docstring for why the border width matters). Neither cv2.aruco nor the
modern AprilRobotics/apriltag reference library can read this target's
2-cell-border tags, so this exists purely to give live_aprilgrid_preview.py
something to draw an overlay from.

Approach (deliberately simple -- this only needs to be good enough for
visual aiming feedback, not calibration-grade precision):
1. Threshold + find contours; keep convex, roughly-square, 4-corner
   candidates in a plausible size range (this is the same "quad detection"
   idea real AprilTag detectors use, just via OpenCV's basic
   findContours/approxPolyDP instead of a purpose-built quad detector).
2. Perspective-warp each candidate to a canonical TOTAL_CELLS x TOTAL_CELLS
   grid (10x10 cells, matching gen_aprilgrid.py's BORDER_BITS=2 +
   DATA_BITS_SIDE=6).
3. Reject candidates whose warped 2-cell border ring isn't mostly black
   (filters out non-tag squares).
4. Sample the center of each of the 36 data cells to get a 36-bit pattern.
5. Compare that pattern (and its 3 other 90-degree rotations, since the
   quad's corner order doesn't tell us which corner is "first") against
   t36h11_codes.json by Hamming distance; accept the best match if it's
   comfortably below the family's minimum inter-code distance (11 bits,
   see Tag36h11.h), same error-tolerance idea real AprilTag decoders use.
"""
import json
import os

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, "t36h11_codes.json")) as f:
    TAG_CODES = json.load(f)

BORDER_BITS = 2
DATA_BITS_SIDE = 6
TOTAL_CELLS = DATA_BITS_SIDE + 2 * BORDER_BITS  # 10
CELL_PX = 10  # canonical warp resolution per cell
CANON_PX = TOTAL_CELLS * CELL_PX

# Family's minimum Hamming distance (accounting for rotation) is 11 (see
# Tag36h11.h) -- (11-1)/2 = 5 bit errors are theoretically recoverable, but
# stay well clear of that boundary here since this is a coarse detector
# (no subpixel refinement) prone to noisier bit reads than the real thing.
MAX_HAMMING_DISTANCE = 4

_MIN_QUAD_AREA_PX = 400  # 20x20px minimum, at whatever resolution frames arrive


def _order_corners(pts):
    """Orders 4 points as top-left, top-right, bottom-right, bottom-left,
    by summing/differencing x+y and x-y (standard trick)."""
    pts = pts.reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]  # top-left (smallest x+y)
    ordered[2] = pts[np.argmax(s)]  # bottom-right (largest x+y)
    ordered[1] = pts[np.argmin(d)]  # top-right (smallest y-x)
    ordered[3] = pts[np.argmax(d)]  # bottom-left (largest y-x)
    return ordered


def _find_candidate_quads(gray):
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 5)
    # RETR_EXTERNAL (not RETR_LIST): a tag's black border ring produces both
    # an outer and an inner contour: RETR_LIST returns both (duplicate
    # detections of the same tag, confirmed empirically), RETR_EXTERNAL
    # keeps only the outermost one.
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    quads = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < _MIN_QUAD_AREA_PX:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        x, y, w, h = cv2.boundingRect(approx)
        aspect = w / float(h)
        if not (0.7 < aspect < 1.4):
            continue
        quads.append(_order_corners(approx.astype(np.float32)))
    return quads


def _warp_to_canonical(gray, corners):
    dst = np.array([[0, 0], [CANON_PX - 1, 0], [CANON_PX - 1, CANON_PX - 1],
                     [0, CANON_PX - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    return cv2.warpPerspective(gray, M, (CANON_PX, CANON_PX))


def _border_is_black(canon, threshold=110):
    mask = np.ones((TOTAL_CELLS, TOTAL_CELLS), dtype=bool)
    mask[BORDER_BITS:-BORDER_BITS, BORDER_BITS:-BORDER_BITS] = False
    border_cell_means = []
    for i in range(TOTAL_CELLS):
        for j in range(TOTAL_CELLS):
            if not mask[i, j]:
                continue
            cy, cx = i * CELL_PX + CELL_PX // 2, j * CELL_PX + CELL_PX // 2
            border_cell_means.append(canon[cy, cx])
    return np.mean(border_cell_means) < threshold


def _sample_bits(canon, threshold=128):
    bits = np.zeros((DATA_BITS_SIDE, DATA_BITS_SIDE), dtype=np.uint8)
    for i in range(DATA_BITS_SIDE):
        for j in range(DATA_BITS_SIDE):
            cy = (BORDER_BITS + i) * CELL_PX + CELL_PX // 2
            cx = (BORDER_BITS + j) * CELL_PX + CELL_PX // 2
            # black cell (pixel < threshold) -> code_matrix bit 1 -> original
            # code bit 0 (see gen_aprilgrid.py's render_tag: "if not (code &
            # bit): code_matrix=1").
            bits[i, j] = 1 if canon[cy, cx] < threshold else 0
    return bits


def _bits_to_code(bits):
    code = 0
    for i in range(DATA_BITS_SIDE):
        for j in range(DATA_BITS_SIDE):
            if bits[i, j] == 0:  # white cell -> original bit 1
                code |= 1 << (DATA_BITS_SIDE * i + j)
    return code


def _best_match(bits):
    best_id, best_dist = None, MAX_HAMMING_DISTANCE + 1
    for rot in range(4):
        rotated = np.rot90(bits, rot)
        code = _bits_to_code(rotated)
        for tag_id, ref_code in enumerate(TAG_CODES):
            dist = bin(code ^ ref_code).count("1")
            if dist < best_dist:
                best_dist, best_id = dist, tag_id
    return (best_id, best_dist) if best_id is not None else (None, None)


def detect(gray):
    """Returns a list of (tag_id, corners) for tags found in a grayscale
    image, corners as a (4,2) float32 array in the same order used
    internally (top-left, top-right, bottom-right, bottom-left) -- NOT
    corrected for which corner is tag-space (0,0), since that requires
    knowing the matched rotation, which is applied here for consistency
    with gen_aprilgrid.py's rendering (rotation=2 fixed offset already
    baked into TAG_CODES lookup via _best_match's rotation search)."""
    results = []
    for corners in _find_candidate_quads(gray):
        canon = _warp_to_canonical(gray, corners)
        if not _border_is_black(canon):
            continue
        bits = _sample_bits(canon)
        tag_id, dist = _best_match(bits)
        if tag_id is not None:
            results.append((tag_id, corners))
    return results
