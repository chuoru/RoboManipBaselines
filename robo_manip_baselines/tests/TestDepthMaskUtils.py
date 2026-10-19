import unittest

import numpy as np

from robo_manip_baselines.common import get_background_mask, mask_background_by_depth


class TestDepthMaskUtils(unittest.TestCase):
    def setUp(self):
        # 1280x800 scene: far background at 2 m, a near object (0.3 m) with an
        # invalid-depth hole inside it (specular metal), and a few isolated
        # invalid pixels in the background.
        self.depth = np.full((800, 1280), 2.0, dtype=np.float32)
        self.depth[400:700, 500:800] = 0.3
        self.depth[500:560, 600:660] = 0.0  # hole inside the object
        self.depth[100:103, 100:103] = 0.0  # stray invalid speck far away
        self.rgb = np.full((800, 1280, 3), 200, dtype=np.uint8)

    def test_keeps_near_object_and_fills_its_hole(self):
        mask = get_background_mask(self.depth, max_depth=0.6)
        self.assertTrue(mask[450:650, 550:750].all())
        self.assertTrue(mask[500:560, 600:660].all())

    def test_background_seen_through_a_gap_stays_removed(self):
        # valid FAR depth enclosed by the object (e.g. seen between fingers)
        depth = self.depth.copy()
        depth[580:640, 700:760] = 2.0
        mask = get_background_mask(depth, max_depth=0.6)
        self.assertFalse(mask[590:630, 710:750].any())

    def test_removes_far_background_and_specks(self):
        mask = get_background_mask(self.depth, max_depth=0.6)
        self.assertFalse(mask[0:300, :].any())
        self.assertFalse(mask[100:103, 100:103].any())

    def test_masked_image(self):
        out = mask_background_by_depth(self.rgb, self.depth, max_depth=0.6)
        self.assertEqual(out[50, 50].sum(), 0)
        self.assertEqual(out[550, 650].tolist(), [200, 200, 200])
        self.assertEqual(self.rgb[50, 50].tolist(), [200, 200, 200])  # input untouched


if __name__ == "__main__":
    unittest.main()
