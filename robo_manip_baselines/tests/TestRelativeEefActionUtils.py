import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from robo_manip_baselines.common import get_absolute_eef_pose9, get_relative_eef_pose9
from robo_manip_baselines.common.utils.PoseUtils import get_pose7_from_pose9


def random_pose7(rng, n):
    quat_xyzw = Rotation.random(n, random_state=rng.integers(1 << 31)).as_quat()
    return np.concatenate([rng.uniform(-0.5, 0.5, (n, 3)), quat_xyzw[:, [3, 0, 1, 2]]], axis=1)


class TestRelativeEefActionUtils(unittest.TestCase):
    def test_round_trip(self):
        rng = np.random.default_rng(0)
        poses = random_pose7(rng, 16)
        ref = random_pose7(rng, 1)[0]
        back = get_pose7_from_pose9(get_absolute_eef_pose9(get_relative_eef_pose9(poses, ref), ref))
        np.testing.assert_allclose(back[:, :3], poses[:, :3], atol=1e-9)
        # quaternions may flip sign
        dots = np.abs(np.sum(back[:, 3:] * poses[:, 3:], axis=1))
        np.testing.assert_allclose(dots, 1.0, atol=1e-9)

    def test_ref_maps_to_identity(self):
        rng = np.random.default_rng(1)
        ref = random_pose7(rng, 1)[0]
        rel = get_relative_eef_pose9(ref[None], ref)[0]
        np.testing.assert_allclose(rel[:3], 0.0, atol=1e-12)
        np.testing.assert_allclose(rel[3:], [1, 0, 0, 0, 1, 0], atol=1e-12)  # identity in 6D

    def test_relative_is_in_ref_local_frame(self):
        # ref rotated 90 deg about z; a target 1 m along world x is 1 m along
        # the ref's local -y.
        ref = np.array([0, 0, 0, np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
        target = np.array([[1.0, 0, 0, 1, 0, 0, 0]])
        rel = get_relative_eef_pose9(target, ref)[0]
        np.testing.assert_allclose(rel[:3], [0, -1, 0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
