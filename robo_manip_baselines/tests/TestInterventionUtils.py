import unittest

import numpy as np

from robo_manip_baselines.common.utils.InterventionUtils import (
    compute_d8_masks,
    get_window_weight,
)


def masks(intervention, input_active, success=True, **kwargs):
    num_steps = len(intervention)
    time_seq = 0.2 * np.arange(num_steps)
    kwargs.setdefault("boundary_drop_ticks", 1)
    return compute_d8_masks(time_seq, intervention, input_active, success, **kwargs)


class TestD8Masks(unittest.TestCase):
    def test_only_the_human_correction_is_used(self):
        #             (1) autonomous | taken over: (3) still, (4) correcting | (5)
        intervention = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
        input_active = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0]
        usable, weight = masks(intervention, input_active, intervention_weight=2.5)
        # (3) = steps 4-5 (before the first input), last boundary tick (11)
        expected = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
        np.testing.assert_array_equal(usable, expected)
        np.testing.assert_allclose(weight[usable], 2.5)

    def test_failed_attempt_contributes_nothing(self):
        usable, _ = masks([0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 1], success=False)
        self.assertFalse(usable.any())

    def test_correction_until_the_end_is_kept(self):
        # The operator finished the segment: no hand-back boundary at the end.
        usable, _ = masks([0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1])
        np.testing.assert_array_equal(usable, [0, 0, 0, 1, 1, 1])

    def test_operator_never_moved(self):
        usable, _ = masks([0, 1, 1, 1, 0], [0, 0, 0, 0, 0])
        self.assertFalse(usable.any())

    def test_teleop_only_attempt(self):
        usable, _ = masks([1] * 6, [0, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(usable, [0, 1, 1, 1, 1, 1])

    def test_autonomous_weight_drops_pre_intervention_window(self):
        intervention = [0] * 10 + [1] * 5
        input_active = [0] * 10 + [1] * 5
        usable, weight = masks(
            intervention,
            input_active,
            autonomous_weight=0.5,
            pre_intervention_drop_sec=1.0,
            boundary_drop_ticks=1,
        )
        # 1.0 s at 0.2 s/step before step 10 = steps 5..9 dropped
        np.testing.assert_array_equal(usable[:10], [1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(weight[:5], 0.5)
        self.assertTrue(usable[11:].all())

    def test_window_weight(self):
        usable = np.array([0, 1, 1, 1, 0], dtype=bool)
        weight = np.array([0, 2, 2, 2, 0], dtype=float)
        self.assertEqual(get_window_weight(usable, weight, np.array([1, 2, 3])), 2.0)
        self.assertEqual(get_window_weight(usable, weight, np.array([2, 3, 4])), 0.0)


if __name__ == "__main__":
    unittest.main()
