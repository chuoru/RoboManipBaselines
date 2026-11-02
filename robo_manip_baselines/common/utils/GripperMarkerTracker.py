"""Gripper opening width from the AprilTags on the fingers, for any env.

Same measurement as RealUMIEnvBase (see common/utils/ArucoGripperUtils.py
and envs/configs/RealUMIDemo.yaml for the config keys and how they were
calibrated): detect the two finger tags in the hand camera image, take the
distance between them, reject single-frame jumps, smooth with an EMA and map
to percent-closed with width_open_m / width_closed_m.

In addition it keeps a short history of RAW per-frame readings (including
frames where no tag was found), which the grasp/release checks of the
segment runner take a median over -- see common/utils/GripperCheckUtils.py.
"""

import collections
import threading
import time

import numpy as np

from .ArucoGripperUtils import (
    detect_aruco_tags,
    get_gripper_width,
    parse_aruco_config,
    preprocess_low_light_image,
)


class GripperMarkerTracker:
    # Raw readings kept for the checks (~10 s at 10 fps).
    HISTORY_LEN = 100

    def __init__(self, config, get_frame_func):
        """config: the gripper_marker_config dict (camera_name, aruco_dict,
        marker_size_map, left/right_marker_id, nominal_z, camera_matrix, ...).
        get_frame_func(camera_name) returns the latest RGB frame or None."""
        self.config = config
        self.get_frame_func = get_frame_func
        self.camera_name = config["camera_name"]
        self.aruco = parse_aruco_config(
            {
                "aruco_dict": config["aruco_dict"],
                "marker_size_map": config["marker_size_map"],
            }
        )
        self.camera_matrix = np.array(config["camera_matrix"], dtype=np.float64)
        dist_coeffs = config.get("dist_coeffs")
        self.dist_coeffs = (
            None if dist_coeffs is None else np.array(dist_coeffs, dtype=np.float64)
        )
        self.clahe_clip_limit = config.get("clahe_clip_limit", 0.0)
        self.max_width_jump_m = config.get("max_width_jump_m", 0.03)
        self.width_smoothing_alpha = config.get("width_smoothing_alpha", 0.3)
        roi = config.get("roi")
        self.roi = None if roi is None else tuple(roi)
        # Detection costs tens of ms per frame at 1280x800; in the background
        # it does not slow the control loop (readings are then at most one
        # camera frame old).
        self.async_detection = config.get("async_detection", True)

        self.lock = threading.Lock()
        self.thread = None
        self.stop_event = threading.Event()
        self.reset()

    def reset(self):
        with self.lock:
            self.smoothed_width_m = None
            self.last_percent_closed = None
            self.last_detected = False
            self.last_raw_width_m = None
            self.last_update_time = None
            # (wall time, raw width [m] or None when not detected)
            self.history = collections.deque(maxlen=self.HISTORY_LEN)
            # Detection durations [s], to tell whether detection keeps up
            # with the camera (see get_detect_time_ms).
            self.detect_durations = collections.deque(maxlen=self.HISTORY_LEN)

    def start(self):
        if not self.async_detection or self.thread is not None:
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()

    def stop(self):
        if self.thread is not None:
            self.stop_event.set()
            self.thread.join(timeout=1.0)
            self.thread = None

    def detect_width(self, frame):
        """Raw width [m] of one frame, or None if no usable tag was found."""
        if self.clahe_clip_limit > 0:
            frame = preprocess_low_light_image(frame, clip_limit=self.clahe_clip_limit)
        tag_dict = detect_aruco_tags(
            frame,
            self.aruco["aruco_dict"],
            self.aruco["marker_size_map"],
            self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            corner_refinement=self.aruco["corner_refinement"],
            roi=self.roi,
        )
        return get_gripper_width(
            tag_dict,
            left_id=self.config["left_marker_id"],
            right_id=self.config["right_marker_id"],
            nominal_z=self.config["nominal_z"],
            z_tolerance=self.config.get("z_tolerance", 0.008),
        )

    def update_from_frame(self, frame, now=None):
        start_time = time.time()
        width_m = self.detect_width(frame)
        with self.lock:
            self.detect_durations.append(time.time() - start_time)
        self.update_from_width(width_m, now)

    def get_detect_time_ms(self):
        """Median detection time per frame [ms] (None before any frame).
        Should stay well below the camera period (100 ms at 10 fps)."""
        with self.lock:
            if len(self.detect_durations) == 0:
                return None
            return 1000.0 * float(np.median(self.detect_durations))

    def update_from_width(self, width_m, now=None):
        """Feed one frame's raw width (None = not detected)."""
        if now is None:
            now = time.time()
        with self.lock:
            self.history.append((now, width_m))
            self.last_update_time = now
            self.last_raw_width_m = width_m
            self.last_detected = width_m is not None
            if width_m is None:
                return
            # Same two layers of defence as RealUMIEnvBase: reject a
            # physically implausible single-frame jump, then smooth.
            if (
                self.smoothed_width_m is not None
                and abs(width_m - self.smoothed_width_m) > self.max_width_jump_m
            ):
                width_m = self.smoothed_width_m
            if self.smoothed_width_m is None:
                self.smoothed_width_m = width_m
            else:
                alpha = self.width_smoothing_alpha
                self.smoothed_width_m = (
                    alpha * width_m + (1.0 - alpha) * self.smoothed_width_m
                )
            width_open_m = self.config.get("width_open_m")
            width_closed_m = self.config.get("width_closed_m")
            if width_open_m is not None and width_closed_m is not None:
                percent_closed = (
                    100.0
                    * (width_open_m - self.smoothed_width_m)
                    / (width_open_m - width_closed_m)
                )
                self.last_percent_closed = float(np.clip(percent_closed, 0.0, 100.0))

    def get_reading(self):
        """Latest state: smoothed width [m] (None before any detection),
        percent-closed, whether the LAST frame had the tags, its raw width,
        and when it was taken."""
        with self.lock:
            return {
                "width_m": self.smoothed_width_m,
                "percent_closed": self.last_percent_closed,
                "detected": self.last_detected,
                "raw_width_m": self.last_raw_width_m,
                "time": self.last_update_time,
            }

    def get_raw_widths_since(self, since_time):
        """Raw readings (width [m] or None) of frames taken after since_time."""
        with self.lock:
            return [width for stamp, width in self.history if stamp > since_time]

    def _detection_loop(self):
        last_frame = None
        while not self.stop_event.is_set():
            frame = self.get_frame_func(self.camera_name)
            if frame is None or frame is last_frame:
                time.sleep(0.002)
                continue
            last_frame = frame
            try:
                self.update_from_frame(frame)
            except Exception as e:  # noqa: BLE001 -- keep tracking on a bad frame
                print(
                    f"[{self.__class__.__name__}] Gripper marker detection failed: {e}"
                )
