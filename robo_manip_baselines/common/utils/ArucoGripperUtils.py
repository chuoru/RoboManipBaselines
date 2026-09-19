"""ArUco-marker-based gripper width estimation for the UMI handheld gripper
rig, following the same approach as
https://github.com/real-stanford/universal_manipulation_interface (see
umi/common/cv_util.py's detect_localize_aruco_tags/get_gripper_width): a
small marker is mounted on each gripper finger, both visible to the
wrist/hand camera, and the gripper's opening width is recovered from the
markers' estimated 3D poses relative to the camera.

Ported to this repo's OpenCV version (4.11) rather than copied verbatim,
because the free functions UMI's code calls
(cv2.aruco.detectMarkers/cv2.aruco.estimatePoseSingleMarkers) were removed
upstream in OpenCV 4.7+ in favor of the cv2.aruco.ArucoDetector class and
manual solvePnP -- see detect_aruco_tags() below. The math (which corner
axis becomes the marker's local Z after undistortion, and get_gripper_width's
depth-outlier-filter + single-marker-mirror fallback) is unchanged from UMI's
original.
"""

from typing import Dict, Optional

import cv2
import numpy as np


def preprocess_low_light_image(image: np.ndarray, clip_limit: float = 6.0) -> np.ndarray:
    """CLAHE (contrast-limited adaptive histogram equalization) contrast
    boost, applied on the grayscale image before ArUco detection.

    Measured as necessary in practice: a real UMI-rig hand camera frame
    with mean brightness ~9/255 (a dim room) detected 0/2 mounted markers
    as captured, 1/2 with a mild CLAHE (clip_limit=3), and 2/2 with this
    stronger default (clip_limit=6) -- detection itself (corner-pattern
    thresholding), not pose estimation, is what fails first as brightness
    drops, so this must run before detect_aruco_tags, not after.
    clip_limit=0 (or a negative value) disables enhancement and returns the
    input grayscale image unchanged, for scenes already well lit."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if clip_limit <= 0:
        return gray
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(4, 4))
    return clahe.apply(gray)


def get_aruco_dict(predefined: str) -> cv2.aruco.Dictionary:
    """e.g. get_aruco_dict("DICT_4X4_50") or get_aruco_dict("DICT_APRILTAG_36h11")
    -- same convention as UMI's aruco_config.yaml's aruco_dict.predefined
    field. AprilTag families (DICT_APRILTAG_16h5/25h9/36h10/36h11) are
    supported the same way: OpenCV's cv2.aruco module treats them as just
    another predefined dictionary, so ArucoDetector/detect_aruco_tags below
    need no special-casing to detect and localize them -- verified via
    synthetic round-trip (sub-mm accuracy, same as DICT_4X4_50 -- see
    tests/TestArucoGripperUtils.py). The one thing worth choosing per-family
    is corner refinement method -- see _default_corner_refinement below."""
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, predefined))


def _default_corner_refinement(predefined: str) -> int:
    """AprilTag markers have their own dedicated refinement method
    (CORNER_REFINE_APRILTAG, the AprilTag paper's own edge-based corner
    fit) that OpenCV recommends over the generic CORNER_REFINE_SUBPIX for
    that family specifically; ArUco-style dictionaries (DICT_4X4_* etc.)
    still use CORNER_REFINE_SUBPIX."""
    if "APRILTAG" in predefined.upper():
        return cv2.aruco.CORNER_REFINE_APRILTAG
    return cv2.aruco.CORNER_REFINE_SUBPIX


def parse_aruco_config(aruco_config_dict: dict):
    """Same schema/semantics as UMI's parse_aruco_config: a marker_size_map
    with an optional "default" entry, expanded to every marker id in the
    dictionary. Example:
        aruco_dict:
          predefined: DICT_4X4_50
        marker_size_map:
          default: 0.06
          0: 0.016
          1: 0.016
    Also picks a corner-refinement method matched to the dictionary family
    (see _default_corner_refinement); the returned "corner_refinement" is
    meant to be passed straight through to detect_aruco_tags.
    """
    aruco_dict = get_aruco_dict(**aruco_config_dict["aruco_dict"])
    corner_refinement = _default_corner_refinement(
        aruco_config_dict["aruco_dict"]["predefined"]
    )

    n_markers = len(aruco_dict.bytesList)
    marker_size_map = aruco_config_dict["marker_size_map"]
    default_size = marker_size_map.get("default", None)

    out_marker_size_map = {}
    for marker_id in range(n_markers):
        size = marker_size_map.get(marker_id, default_size)
        out_marker_size_map[marker_id] = size

    return {
        "aruco_dict": aruco_dict,
        "marker_size_map": out_marker_size_map,
        "corner_refinement": corner_refinement,
    }


def detect_aruco_tags(
    image: np.ndarray,
    aruco_dict: cv2.aruco.Dictionary,
    marker_size_map: Dict[int, Optional[float]],
    camera_matrix: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None,
    corner_refinement: Optional[int] = None,
) -> Dict[int, dict]:
    """Detect every ArUco/AprilTag marker in `image` with a known size in
    marker_size_map and estimate each one's pose relative to the camera.

    camera_matrix: (3, 3) pinhole intrinsics (already the UNDISTORTED/
    rectified intrinsics if dist_coeffs is None -- e.g. for a fisheye lens,
    undistort the image, or the marker corners, before calling this with
    dist_coeffs=None; see cv2.fisheye.undistortPoints, as UMI's code does).

    corner_refinement: one of cv2.aruco.CORNER_REFINE_*, or None to default
    to CORNER_REFINE_SUBPIX. When calling this with a dictionary/config
    produced by parse_aruco_config, pass its "corner_refinement" value
    instead of leaving this as None -- it already picks
    CORNER_REFINE_APRILTAG for an AprilTag family dictionary, which OpenCV
    recommends over SUBPIX for those markers specifically.

    Returns {marker_id: {"rvec": (3,), "tvec": (3,), "corners": (4, 2)}},
    tvec in the same units as marker_size_map's sizes (meters, by
    convention).
    """
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = (
        cv2.aruco.CORNER_REFINE_SUBPIX if corner_refinement is None else corner_refinement
    )
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(image)

    if ids is None or len(ids) == 0:
        return {}

    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)

    tag_dict = {}
    for marker_id_arr, marker_corners in zip(ids, corners):
        marker_id = int(marker_id_arr[0])
        marker_size_m = marker_size_map.get(marker_id)
        if marker_size_m is None:
            continue

        # Marker-local 3D corner template, matching cv2.aruco's own detected
        # corner ordering (top-left, top-right, bottom-right, bottom-left in
        # IMAGE space) and the convention the old estimatePoseSingleMarkers
        # used: centered at the marker's origin, Z=0 in its own plane. Note
        # OpenCV's camera-frame convention is Y-DOWN (X right, Y down, Z
        # forward), so "top" is the NEGATIVE-Y side, not positive -- easy to
        # get backwards (as an earlier draft of this function did; a Y-sign
        # flip here makes solvePnP's correspondence set inconsistent with
        # the actually-detected corners, which manifests as pose solves
        # silently failing/degenerating, not a clean error).
        half = marker_size_m / 2.0
        object_points = np.array(
            [
                [-half, -half, 0.0],
                [half, -half, 0.0],
                [half, half, 0.0],
                [-half, half, 0.0],
            ],
            dtype=np.float64,
        )

        # SOLVEPNP_IPPE_SQUARE (nominally the right choice for a known-size
        # square marker) was measured to return degenerate near-zero tvecs
        # here despite ok=True; SOLVEPNP_ITERATIVE was verified (synthetic
        # projection round-trip, see the test invoked from this repo) to
        # recover the true pose to sub-mm/sub-degree accuracy for this
        # exact 4-point-square setup, so it is used instead.
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            marker_corners.reshape(4, 2),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            continue

        tag_dict[marker_id] = {
            "rvec": rvec.squeeze(),
            "tvec": tvec.squeeze(),
            "corners": marker_corners.squeeze(),
        }

    return tag_dict


def get_gripper_width(
    tag_dict: Dict[int, dict],
    left_id: int,
    right_id: int,
    nominal_z: float,
    z_tolerance: float = 0.008,
) -> Optional[float]:
    """Recover the gripper opening width [m] from the two finger markers'
    poses, exactly following UMI's umi/common/cv_util.py:get_gripper_width.

    nominal_z: expected marker depth from the camera [m] (fixed by the rig's
    geometry -- how far the camera is mounted from the finger markers when
    the gripper is roughly facing the camera). Markers whose estimated depth
    falls outside [nominal_z - z_tolerance, nominal_z + z_tolerance] are
    rejected as outliers (e.g. a spurious detection, or the marker at a
    steep/unreliable viewing angle).

    Returns None if neither marker is validly detected. If only one is, the
    width is approximated as 2x that marker's absolute X-offset from the
    camera's optical axis (relies on the camera being mounted on the
    gripper's own symmetry axis, i.e. equidistant from both fingers when
    closed) -- this mirrors UMI's identical single-marker fallback.
    """
    zmin = nominal_z - z_tolerance
    zmax = nominal_z + z_tolerance

    left_x = None
    if left_id in tag_dict:
        tvec = tag_dict[left_id]["tvec"]
        if zmin < tvec[-1] < zmax:
            left_x = tvec[0]

    right_x = None
    if right_id in tag_dict:
        tvec = tag_dict[right_id]["tvec"]
        if zmin < tvec[-1] < zmax:
            right_x = tvec[0]

    if (left_x is not None) and (right_x is not None):
        return right_x - left_x
    elif left_x is not None:
        return abs(left_x) * 2
    elif right_x is not None:
        return abs(right_x) * 2
    return None
