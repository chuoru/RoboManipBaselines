import threading
import unittest

import gymnasium as gym
import numpy as np

import robo_manip_baselines.envs  # noqa: F401 (registers the gym envs)
from robo_manip_baselines.common import get_aruco_dict
from robo_manip_baselines.tests.TestArucoGripperUtils import render_marker

MARKER_SIZE = 0.012  # [m]
NOMINAL_Z = 0.05  # [m]
CAMERA_MATRIX = [[400.0, 0, 480.0], [0, 400.0, 480.0], [0, 0, 1.0]]
WIDTH_OPEN_M = 0.09
WIDTH_CLOSED_M = 0.0


def make_gripper_marker_config(**overrides):
    config = {
        "camera_name": "hand",
        "aruco_dict": {"predefined": "DICT_4X4_50"},
        "marker_size_map": {"default": None, 0: MARKER_SIZE, 1: MARKER_SIZE},
        "left_marker_id": 0,
        "right_marker_id": 1,
        "nominal_z": NOMINAL_Z,
        "z_tolerance": 0.01,
        "camera_matrix": CAMERA_MATRIX,
        "width_open_m": WIDTH_OPEN_M,
        "width_closed_m": WIDTH_CLOSED_M,
    }
    config.update(overrides)
    return config


class TestGripperMarkerSmoothing(unittest.TestCase):
    """Regression test for a real issue reported against the live rig: the
    displayed/recorded gripper-marker percent-closed occasionally snapped
    to 0% or 100% under changing lighting. Root cause was a single spurious
    per-frame width reading (bad corner localization, etc.) landing outside
    the calibrated range and then being clipped to an extreme -- see
    RealUMIEnvBase.__init__'s comment on max_width_jump_m/
    width_smoothing_alpha. This locks in that the fix actually rejects such
    an outlier while still tracking real, gradual motion.
    """

    def setUp(self):
        self.aruco_dict = get_aruco_dict("DICT_4X4_50")
        self.camera_matrix = np.array(CAMERA_MATRIX, dtype=np.float64)
        self.env = gym.make(
            "robo_manip_baselines/RealUMIDemoEnv-v0",
            gripper_marker_config=make_gripper_marker_config(
                max_width_jump_m=0.03, width_smoothing_alpha=0.3
            ),
        )
        self.unwrapped = self.env.unwrapped

    def tearDown(self):
        # Bypass RealEnvBase.close()'s rgb_cameras cleanup: this test injects
        # frames directly into self.unwrapped.rgb_cameras without going
        # through setup_insta360, so that dict lacks the
        # thread/stop_event/connection keys a real camera setup would have.
        self.unwrapped.rgb_cameras = {}
        self.env.close()

    def _set_frame(self, width_m):
        image = np.full((960, 960, 3), 255, dtype=np.uint8)
        render_marker(
            image, self.aruco_dict, 0, MARKER_SIZE, (-width_m / 2, 0, NOMINAL_Z),
            self.camera_matrix,
        )
        render_marker(
            image, self.aruco_dict, 1, MARKER_SIZE, (width_m / 2, 0, NOMINAL_Z),
            self.camera_matrix,
        )
        self.unwrapped.rgb_cameras["hand"] = {
            "lock": threading.Lock(),
            "latest_frame": image,
        }

    def _read_percent_closed(self):
        return self.unwrapped._get_obs()["joint_pos"][7]

    def test_spurious_outlier_frame_is_rejected(self):
        # Settle at a steady mid-range width (~50% closed).
        for _ in range(10):
            self._set_frame(0.045)
            steady_pct = self._read_percent_closed()
        self.assertAlmostEqual(steady_pct, 50.0, delta=2.0)

        # One spurious frame reporting fully-open width -- must NOT snap
        # the output to 0%.
        self._set_frame(WIDTH_OPEN_M)
        spurious_pct = self._read_percent_closed()
        self.assertAlmostEqual(spurious_pct, steady_pct, delta=2.0)

        # Immediately back to normal -- confirms the outlier didn't
        # permanently corrupt the smoothing filter's state either.
        self._set_frame(0.045)
        recovery_pct = self._read_percent_closed()
        self.assertAlmostEqual(recovery_pct, steady_pct, delta=2.0)

    def test_gradual_real_motion_is_still_tracked(self):
        for _ in range(5):
            self._set_frame(0.045)
            self._read_percent_closed()

        for i in range(30):
            width = 0.045 + i * ((WIDTH_OPEN_M - 0.045) / 30)
            self._set_frame(width)
            pct = self._read_percent_closed()

        # Real, gradual motion to fully open must actually be reflected
        # (allowing for smoothing lag -- not asserting it reaches exactly 0%).
        self.assertLess(pct, 20.0)


if __name__ == "__main__":
    unittest.main()
