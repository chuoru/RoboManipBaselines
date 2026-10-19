"""Depth-based background removal for a camera's RGB image.

Policies trained on UMI demos recorded in one room learned to read the
BACKGROUND (shelves, monitors, people) instead of the cup: on the robot, whose
cell looks different, they moved in the wrong direction no matter where the cup
was. Blacking out everything farther than `max_depth` from the camera removes
that cue. The same function must be applied to the training data
(misc/MaskRmbBackgroundByDepth.py) and to the live camera image at rollout
time (RealEnvBase, pointcloud_camera_background_max_depth), on the full
1280x800 image (the mask itself is computed on a 640x400 grid; the
morphology kernel sizes below are in pixels at that resolution).

The depth image is expected in meters, pixel-aligned with the RGB image (on
the Orbbec Gemini 305 the depth is computed from the color camera's own
viewpoint, so the plain resize done in RealEnvBase is within a few pixels).
"""

import cv2
import numpy as np


def get_background_mask(depth, max_depth=0.6, min_area_ratio=0.004):
    """Return a boolean foreground mask (True = keep), same size as depth.

    Foreground = valid depth closer than max_depth, plus invalid-depth pixels
    right next to it (specular metal and transparent objects return no depth),
    cleaned up with an opening/closing and by dropping small blobs.

    Computed at half resolution (kernel sizes are for that 640x400 grid when
    the input is 1280x800) and upsampled with nearest-neighbour, which cut the
    cost from ~40 ms to ~15 ms per frame -- it runs inside the rollout loop.
    """
    full_h, full_w = depth.shape[:2]
    depth = cv2.resize(depth, (full_w // 2, full_h // 2), interpolation=cv2.INTER_NEAREST)
    mask = _get_background_mask_half(depth, max_depth, min_area_ratio)
    return cv2.resize(
        mask.astype(np.uint8), (full_w, full_h), interpolation=cv2.INTER_NEAREST
    ).astype(bool)


def _get_background_mask_half(depth, max_depth, min_area_ratio):
    valid = depth > 0
    fg = (valid & (depth < max_depth)).astype(np.uint8)
    near = cv2.dilate(fg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    mask = fg | ((~valid).astype(np.uint8) & near)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    )
    # Fill holes fully enclosed by the foreground, but only their invalid-depth
    # pixels: a large no-return patch (the inside of the metal cup up close) is
    # part of the object, whereas background genuinely visible through a gap
    # (valid, far depth) must stay removed.
    outside = np.pad(1 - mask, 1, constant_values=1)
    flood = outside.copy()
    cv2.floodFill(flood, None, (0, 0), 2)
    enclosed = (flood[1:-1, 1:-1] == 1) & (mask == 0)
    mask = mask | (enclosed & ~valid).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros(num, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] > min_area_ratio * mask.size
    return keep[labels]


def mask_background_by_depth(rgb, depth, max_depth=0.6):
    """Return a copy of rgb with the background (see get_background_mask) set to 0."""
    mask = get_background_mask(depth, max_depth)
    out = rgb.copy()
    out[~mask] = 0
    return out
