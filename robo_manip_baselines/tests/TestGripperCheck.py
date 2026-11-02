import unittest

from robo_manip_baselines.common.utils.GripperCheckUtils import (
    evaluate_grasp,
    evaluate_release,
    median_width,
)
from robo_manip_baselines.common.utils.GripperMarkerTracker import GripperMarkerTracker

CONFIG = {
    "camera_name": "hand",
    "aruco_dict": {"predefined": "DICT_APRILTAG_16h5"},
    "marker_size_map": {"default": None, 0: 0.015, 1: 0.015},
    "left_marker_id": 1,
    "right_marker_id": 0,
    "nominal_z": 0.18,
    "camera_matrix": [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
    "width_open_m": 0.12,
    "width_closed_m": 0.02,
    "max_width_jump_m": 0.03,
    "width_smoothing_alpha": 0.5,
    "async_detection": False,
}


class TestGripperCheckUtils(unittest.TestCase):
    def test_median_ignores_missing_frames(self):
        self.assertEqual(median_width([None, 0.05, 0.07, None, 0.06], 3), (0.06, 3))
        self.assertEqual(median_width([None, 0.05, None], 3), (None, 1))

    def test_grasp_outcomes(self):
        self.assertEqual(evaluate_grasp([0.07] * 5, 0.04, 0.10)[0], "grasped")
        self.assertEqual(evaluate_grasp([0.02] * 5, 0.04, 0.10)[0], "empty")
        self.assertEqual(evaluate_grasp([0.11] * 5, 0.04, 0.10)[0], "not_closed")
        self.assertEqual(evaluate_grasp([None] * 5, 0.04, 0.10)[0], "undetected")
        # One misdetected frame cannot flip a held cup into "empty".
        self.assertEqual(
            evaluate_grasp([0.07, 0.07, 0.01, 0.07], 0.04, 0.10)[0], "grasped"
        )

    def test_release_outcomes(self):
        self.assertEqual(evaluate_release([0.12] * 4, 0.10)[0], "released")
        self.assertEqual(evaluate_release([0.07] * 4, 0.10)[0], "not_released")
        self.assertEqual(evaluate_release([0.12, None, None], 0.10)[0], "undetected")


class TestGripperMarkerTracker(unittest.TestCase):
    def test_smoothing_jump_rejection_and_history(self):
        tracker = GripperMarkerTracker(CONFIG, lambda name: None)
        tracker.update_from_width(0.10, now=1.0)
        self.assertAlmostEqual(tracker.get_reading()["width_m"], 0.10)
        self.assertAlmostEqual(tracker.get_reading()["percent_closed"], 20.0)
        tracker.update_from_width(0.08, now=2.0)  # EMA 0.5
        self.assertAlmostEqual(tracker.get_reading()["width_m"], 0.09)
        tracker.update_from_width(0.01, now=3.0)  # implausible jump: held
        self.assertAlmostEqual(tracker.get_reading()["width_m"], 0.09)
        tracker.update_from_width(None, now=4.0)
        reading = tracker.get_reading()
        self.assertFalse(reading["detected"])
        self.assertAlmostEqual(reading["width_m"], 0.09)
        # The checks see RAW readings, misses included.
        self.assertEqual(tracker.get_raw_widths_since(1.5), [0.08, 0.01, None])
        tracker.reset()
        self.assertIsNone(tracker.get_reading()["width_m"])
        self.assertEqual(tracker.get_raw_widths_since(0.0), [])


if __name__ == "__main__":
    unittest.main()
