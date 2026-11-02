"""Did the grasp / release work? Judged from the finger AprilTag width.

The tool_do gripper reports only what it was told (open / closed); whether
the fingers actually closed ON THE CUP is only visible in the camera: when a
cup is held, the fingers stop at the cup's width; when the grasp missed,
they close all the way.

    width <= width_min_m              -> "empty"     (closed on nothing)
    width_min_m < width < width_max_m -> "grasped"   (stopped by the cup)
    width >= width_max_m              -> "not_closed" (fingers did not close)
    too few frames with both tags     -> "undetected"

The width is the median over several frames, so one misdetected frame
cannot flip the result. Calibrate width_min_m / width_max_m /
width_open_min_m with misc/CalibrateGripperMarkerWidth.py --env
RealFairino5DemoEnv-v0 (empty-closed, closed-on-cup and open widths).
"""

import numpy as np

GRASP_RESULTS = ("grasped", "empty", "not_closed", "undetected")
RELEASE_RESULTS = ("released", "not_released", "undetected")


def median_width(widths, min_detections):
    """(median of the detected widths, number of detections); median is
    None if fewer than min_detections frames had a reading."""
    detected = [w for w in widths if w is not None and np.isfinite(w)]
    if len(detected) < max(int(min_detections), 1):
        return None, len(detected)
    return float(np.median(detected)), len(detected)


def evaluate_grasp(widths, width_min_m, width_max_m, min_detections=3):
    width, num = median_width(widths, min_detections)
    if width is None:
        return "undetected", None, num
    if width <= width_min_m:
        return "empty", width, num
    if width >= width_max_m:
        return "not_closed", width, num
    return "grasped", width, num


def evaluate_release(widths, width_open_min_m, min_detections=3):
    width, num = median_width(widths, min_detections)
    if width is None:
        return "undetected", None, num
    if width >= width_open_min_m:
        return "released", width, num
    return "not_released", width, num
