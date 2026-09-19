import unittest

import numpy as np

from robo_manip_baselines.common import VisualRelativePoseEstimator
from robo_manip_baselines.misc.ValidateVisualPoseEstimatorInMujoco import (
    evaluate,
    handheld_like_trajectory,
    render_camera_trajectory,
)


class TestVisualRelativePoseEstimator(unittest.TestCase):
    def test_first_call_returns_none(self):
        estimator = VisualRelativePoseEstimator(
            camera_matrix=np.eye(3), min_tracked_points=1
        )
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        self.assertIsNone(estimator.update(frame))

    def test_pure_translation_direction_sign_and_accuracy(self):
        """Regression test for a real bug found while validating this class
        against MuJoCo-rendered frames: cv2.recoverPose's raw output was
        measured to point OPPOSITE the camera's actual direction of travel
        (cosine similarity ~-0.999 against ground truth, consistently, in a
        pure-translation scene) -- see VisualRelativePoseEstimator.update()'s
        negation of `translation`, and the module docstring's "SECOND
        LIMITATION" section. This locks that fix in."""
        positions = [[-0.5 + i * (1.0 / 40), 0.0, 1.0] for i in range(40)]
        rpys = [[0.0, 0.9, np.pi / 2] for _ in range(40)]
        frames, gt_positions, gt_rotmats, camera_matrix = render_camera_trajectory(
            positions, rpys
        )
        rot_errs_deg, cos_sims = evaluate(
            frames, gt_positions, gt_rotmats, camera_matrix
        )
        self.assertGreater(len(cos_sims), 10)
        self.assertGreater(np.median(cos_sims), 0.95)
        self.assertLess(np.median(rot_errs_deg), 1.0)

    def test_handheld_like_trajectory_translation_direction_positively_correlated(
        self,
    ):
        """Weaker guarantee than the pure-translation case above (see the
        module docstring's documented rotation/translation-coupling
        degeneracy): a realistic translation-dominant handheld-style
        trajectory should still show a clearly POSITIVE median cosine
        similarity to ground truth, even though it will be well below 1.0.
        This is deliberately NOT tested for a rotation-heavy trajectory --
        that regime is documented as unreliable, not asserted as working.
        """
        positions, rpys = handheld_like_trajectory(n_steps=150)
        frames, gt_positions, gt_rotmats, camera_matrix = render_camera_trajectory(
            positions, rpys
        )
        rot_errs_deg, cos_sims = evaluate(
            frames, gt_positions, gt_rotmats, camera_matrix
        )
        self.assertGreater(len(cos_sims), 30)
        self.assertGreater(np.median(cos_sims), 0.1)
        self.assertLess(np.median(rot_errs_deg), 3.0)


if __name__ == "__main__":
    unittest.main()
