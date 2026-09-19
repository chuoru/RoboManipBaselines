import unittest

import cv2
import numpy as np

from robo_manip_baselines.common import (
    detect_aruco_tags,
    get_aruco_dict,
    get_gripper_width,
    parse_aruco_config,
)


def render_marker(img, aruco_dict, marker_id, size_m, center_xyz_cam, camera_matrix):
    """Synthetically composite one ArUco marker into img at a known 3D pose
    in the camera frame, via a pinhole projection + perspective warp. Used
    to build ground-truth test scenes without needing real hardware/Mujoco.
    """
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 200)
    half = size_m / 2.0
    # cv2's camera-frame convention is Y-DOWN, so the marker image's
    # top-left pixel (0, 0) maps to object point (-half, -half, 0) -- see
    # ArucoGripperUtils.detect_aruco_tags's identical convention.
    object_points = np.array(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ]
    ) + np.array(center_xyz_cam)
    image_points, _ = cv2.projectPoints(
        object_points, np.zeros(3), np.zeros(3), camera_matrix, None
    )
    image_points = image_points.reshape(4, 2).astype(np.float32)
    source_points = np.array(
        [[0, 0], [200, 0], [200, 200], [0, 200]], dtype=np.float32
    )
    homography = cv2.getPerspectiveTransform(source_points, image_points)
    warped = cv2.warpPerspective(
        cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR),
        homography,
        (img.shape[1], img.shape[0]),
    )
    mask = cv2.warpPerspective(
        np.full((200, 200), 255, dtype=np.uint8),
        homography,
        (img.shape[1], img.shape[0]),
    )
    img[mask > 0] = warped[mask > 0]


class TestArucoGripperUtils(unittest.TestCase):
    def setUp(self):
        self.aruco_dict = get_aruco_dict("DICT_4X4_50")
        self.marker_size = 0.012  # [m]
        self.marker_size_map = {0: self.marker_size, 1: self.marker_size}
        self.nominal_z = 0.15  # [m]
        fx = fy = 800.0
        cx, cy = 320.0, 240.0
        self.camera_matrix = np.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64
        )

    def _render_pair(self, left_x, right_x, left_z=None, right_z=None):
        left_z = self.nominal_z if left_z is None else left_z
        right_z = self.nominal_z if right_z is None else right_z
        img = np.full((480, 640, 3), 255, dtype=np.uint8)
        render_marker(
            img, self.aruco_dict, 0, self.marker_size, (left_x, 0, left_z),
            self.camera_matrix,
        )
        render_marker(
            img, self.aruco_dict, 1, self.marker_size, (right_x, 0, right_z),
            self.camera_matrix,
        )
        return img

    def _estimate_width(self, img):
        tag_dict = detect_aruco_tags(
            img, self.aruco_dict, self.marker_size_map, self.camera_matrix
        )
        return get_gripper_width(
            tag_dict, left_id=0, right_id=1, nominal_z=self.nominal_z,
            z_tolerance=0.02,
        )

    def test_width_across_range(self):
        # Sub-mm accuracy expected: this is a clean synthetic render (no
        # sensor noise/blur), so the tolerance here is about validating the
        # solvePnP + get_gripper_width math, not real-world robustness.
        for true_width in (0.02, 0.04, 0.06, 0.09):
            with self.subTest(true_width=true_width):
                img = self._render_pair(-true_width / 2, true_width / 2)
                width = self._estimate_width(img)
                self.assertIsNotNone(width)
                self.assertLess(abs(width - true_width), 0.003)

    def test_single_marker_fallback(self):
        # Only the left marker rendered -- get_gripper_width should double
        # its absolute offset from the optical axis as the width estimate.
        img = self._render_pair(-0.03, 0.03)
        # Blank out the right half of the image where marker 1 was rendered
        # to simulate it being occluded/out of frame, then re-render only
        # the left marker cleanly.
        img = np.full((480, 640, 3), 255, dtype=np.uint8)
        render_marker(
            img, self.aruco_dict, 0, self.marker_size, (-0.03, 0, self.nominal_z),
            self.camera_matrix,
        )
        width = self._estimate_width(img)
        self.assertIsNotNone(width)
        self.assertLess(abs(width - 0.06), 0.003)

    def test_depth_outlier_rejected(self):
        # Left marker rendered far outside the nominal_z window (e.g. a
        # spurious/misidentified detection) must be ignored, falling back
        # to the single-marker (right-only) estimate.
        img = self._render_pair(-0.03, 0.03, left_z=0.40)
        width = self._estimate_width(img)
        self.assertIsNotNone(width)
        self.assertLess(abs(width - 0.06), 0.003)

    def test_no_markers_returns_none(self):
        img = np.full((480, 640, 3), 255, dtype=np.uint8)
        width = self._estimate_width(img)
        self.assertIsNone(width)


class TestArucoGripperUtilsAprilTag(unittest.TestCase):
    """Same coverage as TestArucoGripperUtils.test_width_across_range, but
    for AprilTag family dictionaries (switched to from DICT_4X4_50 in
    practice, for its generally more robust real-world detection) --
    confirms detect_aruco_tags/get_gripper_width need no AprilTag-specific
    code (OpenCV's ArucoDetector treats it as just another predefined
    dictionary) and that parse_aruco_config's corner-refinement selection
    (see _default_corner_refinement) doesn't break accuracy.

    Covers both 16h5 (what the real rig actually uses -- see
    envs/configs/RealUMIDemo.yaml's comment on why: quicker to detect / works
    at a smaller pixel footprint than 36h11, at the cost of a smaller ID
    space and lower resistance to a bit-flip under noise reading as a
    different valid ID) and 36h11 (the more commonly recommended AprilTag
    family in general, most robust against that failure mode) -- both are
    legitimate choices depending on the marker size/distance/lighting
    trade-off, so both stay covered rather than one replacing the other.
    """

    def setUp(self):
        self.marker_size = 0.012  # [m]
        self.marker_size_map = {0: self.marker_size, 1: self.marker_size}
        self.nominal_z = 0.15  # [m]
        self.camera_matrix = np.array(
            [[800.0, 0, 320.0], [0, 800.0, 240.0], [0, 0, 1]], dtype=np.float64
        )

    def test_width_across_range(self):
        for dict_name in ("DICT_APRILTAG_16h5", "DICT_APRILTAG_36h11"):
            aruco_dict = get_aruco_dict(dict_name)
            for true_width in (0.02, 0.04, 0.06, 0.09):
                with self.subTest(dict_name=dict_name, true_width=true_width):
                    img = np.full((480, 640, 3), 255, dtype=np.uint8)
                    render_marker(
                        img, aruco_dict, 0, self.marker_size,
                        (-true_width / 2, 0, self.nominal_z), self.camera_matrix,
                    )
                    render_marker(
                        img, aruco_dict, 1, self.marker_size,
                        (true_width / 2, 0, self.nominal_z), self.camera_matrix,
                    )
                    tag_dict = detect_aruco_tags(
                        img, aruco_dict, self.marker_size_map, self.camera_matrix,
                        corner_refinement=cv2.aruco.CORNER_REFINE_APRILTAG,
                    )
                    width = get_gripper_width(
                        tag_dict, left_id=0, right_id=1, nominal_z=self.nominal_z,
                        z_tolerance=0.02,
                    )
                    self.assertIsNotNone(width)
                    self.assertLess(abs(width - true_width), 0.003)

    def test_parse_aruco_config_picks_apriltag_refinement(self):
        parsed = parse_aruco_config(
            {
                "aruco_dict": {"predefined": "DICT_APRILTAG_36h11"},
                "marker_size_map": {"default": self.marker_size},
            }
        )
        self.assertEqual(parsed["corner_refinement"], cv2.aruco.CORNER_REFINE_APRILTAG)

    def test_parse_aruco_config_picks_subpix_refinement_for_aruco(self):
        parsed = parse_aruco_config(
            {
                "aruco_dict": {"predefined": "DICT_4X4_50"},
                "marker_size_map": {"default": 0.012},
            }
        )
        self.assertEqual(parsed["corner_refinement"], cv2.aruco.CORNER_REFINE_SUBPIX)


if __name__ == "__main__":
    unittest.main()
