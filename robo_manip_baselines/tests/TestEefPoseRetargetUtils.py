import unittest

import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import EpisodeRelativeEefPoseRetargeter


class TestEefPoseRetargetUtils(unittest.TestCase):
    def setUp(self):
        self.ref_se3 = pin.SE3(
            pin.exp3(np.array([0.3, -0.5, 1.2])), np.array([0.4, -0.2, 0.5])
        )

    def test_constant_policy_prediction_moves_target(self):
        # A policy whose first prediction is already offset from identity, and
        # that keeps predicting it while the arm has not moved yet, must still
        # get that offset as its target (regression: the incremental version
        # anchored on the first prediction and left the target at the start).
        retargeter = EpisodeRelativeEefPoseRetargeter(self.ref_se3)
        rel_se3 = pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.02]))
        for _ in range(5):
            abs_se3 = retargeter.to_absolute(rel_se3)
        expected = self.ref_se3 * rel_se3
        np.testing.assert_allclose(abs_se3.translation, expected.translation)
        np.testing.assert_allclose(abs_se3.rotation, expected.rotation)

    def test_round_trip_with_to_episode_relative(self):
        for pos_scale in (1.0, 0.5):
            retargeter = EpisodeRelativeEefPoseRetargeter(self.ref_se3, pos_scale)
            retargeter.to_episode_relative(self.ref_se3)
            rel_se3 = pin.SE3(
                pin.exp3(np.array([0.05, 0.1, -0.08])), np.array([0.05, -0.02, 0.1])
            )
            back = retargeter.to_episode_relative(retargeter.to_absolute(rel_se3))
            np.testing.assert_allclose(back.translation, rel_se3.translation, atol=1e-12)
            np.testing.assert_allclose(back.rotation, rel_se3.rotation, atol=1e-12)

    def test_identity_maps_to_reference(self):
        retargeter = EpisodeRelativeEefPoseRetargeter(self.ref_se3)
        abs_se3 = retargeter.to_absolute(pin.SE3.Identity())
        np.testing.assert_allclose(abs_se3.homogeneous, self.ref_se3.homogeneous)


if __name__ == "__main__":
    unittest.main()
