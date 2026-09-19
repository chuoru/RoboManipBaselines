"""Frame-to-frame monocular visual relative-pose estimation (sparse optical
flow + the 5-point/essential-matrix algorithm), meant to be fused with
MadgwickAhrsFilter's IMU-derived orientation (see ImuAhrsFilter.py) as the
TRANSLATION half of an interim, camera+IMU pose source for
teleop/Insta360InputDevice -- buildable and testable entirely without the
Insta360 CameraSDK/ORB-SLAM3 (see envs/real/insta360_bridge/), since it only
needs a stream of RGB/gray frames + known camera intrinsics, real or
Mujoco-rendered (see tests/TestVisualRelativePoseEstimator.py).

FUNDAMENTAL LIMITATION -- read before using this for anything beyond
plumbing/qualitative validation: a SINGLE moving camera cannot recover
metric scale. cv2.recoverPose() returns a translation whose DIRECTION is
correct (up to noise) but whose MAGNITUDE is an arbitrary unit vector -- the
same visual motion is equally consistent with "the camera moved 1cm near a
tiny scene" or "10m near a huge one". This estimator does not resolve that
ambiguity: its output translation is direction-only, meant to be scaled by a
hand-tuned pos_scale (the same role ViveInputDevice.pos_scale and
Insta360UMI.yaml's pos_scale already play), NOT treated as metric. Real
metric-scale recovery needs either a second view (stereo), a known object in
frame (this is exactly why the ArUco gripper markers in
ArucoGripperUtils.py, which DO have known physical size, are a natural
future scale reference -- not wired up here), or full VIO/SLAM (ORB-SLAM3's
IMU_MONOCULAR mode resolves it via the accelerometer's known magnitude, if
that turns out to be available -- see the bridge's "Known unresolved risk").

SECOND LIMITATION, measured empirically (see
tests/TestVisualRelativePoseEstimator.py and
misc/ValidateVisualPoseEstimatorInMujoco.py, both against real MuJoCo-
rendered frames): translation DIRECTION accuracy itself depends heavily on
how translation-dominant the frame-to-frame motion is. Near-pure translation
recovers the true direction almost exactly (cosine similarity ~0.999 to
ground truth in testing). But as the rotation-to-translation ratio between
consecutive frames grows, translation direction degrades toward noise (a
realistic "mostly-translating, slowly-rotating" handheld trajectory measured
~0.25-0.4 mean/median cosine similarity; a trajectory with rotation and
translation of comparable magnitude between frames measured ~0.05-0.12,
close to uncorrelated). This is NOT a bug -- it is the classical small-
baseline/rotation-translation coupling degeneracy of two-view epipolar
geometry (well documented in the SFM/VO literature): when a frame pair's
apparent pixel motion is dominated by rotation, the translation component of
the essential matrix becomes poorly constrained by that pair alone. Rotation
recovery itself stayed accurate (sub-1-2 deg typical) across all tested
regimes, including rotation-dominant ones -- only translation direction is
affected. Practical implication: trust this estimator's translation more at
a higher frame rate (smaller inter-frame rotation for the same angular
velocity) or when explicitly rotation is not simultaneously large; consider
it unreliable during fast wrist twists.
"""

from typing import Optional

import cv2
import numpy as np


class VisualRelativePoseEstimator:
    """Tracks sparse features frame-to-frame with pyramidal Lucas-Kanade
    optical flow and recovers the relative camera rotation + (scale-free)
    translation DIRECTION between consecutive frames via the essential
    matrix. Call update(frame) once per new frame; the first call only
    seeds features and returns None (no previous frame to compare against).
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        max_corners: int = 200,
        quality_level: float = 0.01,
        min_distance: float = 8.0,
        min_tracked_points: int = 20,
    ):
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.min_tracked_points = min_tracked_points

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_points: Optional[np.ndarray] = None

    @staticmethod
    def _to_gray(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    def _detect_features(self, gray: np.ndarray) -> np.ndarray:
        points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
        )
        return points

    def reset(self):
        """Drop the previous frame/features -- call after a tracking
        failure or a large expected discontinuity, so the next update()
        reseeds from scratch instead of comparing against a stale frame."""
        self._prev_gray = None
        self._prev_points = None

    def update(self, frame: np.ndarray):
        """Returns None on the first call (or after reset()/tracking
        failure -- no relative pose to report yet), otherwise
        {"rotation": (3,3), "translation_direction": (3,) unit vector,
        "num_inliers": int}. `rotation`/`translation_direction` describe the
        SECOND (this) camera pose relative to the FIRST (previous) one, in
        the previous camera's frame -- same convention cv2.recoverPose
        itself uses.
        """
        gray = self._to_gray(frame)

        if self._prev_gray is None:
            self._prev_gray = gray
            self._prev_points = self._detect_features(gray)
            return None

        if self._prev_points is None or len(self._prev_points) < self.min_tracked_points:
            # Not enough features were available to seed tracking last
            # frame (e.g. a texture-less scene) -- try reseeding on THIS
            # frame instead of failing forever.
            self._prev_gray = gray
            self._prev_points = self._detect_features(gray)
            return None

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_points, None
        )
        status = status.reshape(-1).astype(bool)
        prev_matched = self._prev_points[status]
        next_matched = next_points[status]

        result = None
        if len(prev_matched) >= self.min_tracked_points:
            essential_matrix, inlier_mask = cv2.findEssentialMat(
                prev_matched,
                next_matched,
                self.camera_matrix,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0,
            )
            if essential_matrix is not None and essential_matrix.shape == (3, 3):
                num_inliers, rotation, translation, pose_mask = cv2.recoverPose(
                    essential_matrix,
                    prev_matched,
                    next_matched,
                    self.camera_matrix,
                    mask=inlier_mask,
                )
                if num_inliers >= self.min_tracked_points:
                    result = {
                        "rotation": rotation,
                        # NEGATED: cv2.recoverPose's `t` was empirically
                        # verified (pure-translation Mujoco test, see
                        # tests/TestVisualRelativePoseEstimator.py) to point
                        # OPPOSITE the camera's own direction of travel --
                        # i.e. it satisfies X1 = R @ X2 + t (second camera's
                        # position expressed in frame 1), not this class's
                        # documented X2 = R @ X1 + t. Negating here once
                        # keeps that OpenCV-specific inversion contained
                        # instead of leaking into every caller.
                        "translation_direction": -translation.reshape(3),
                        "num_inliers": int(num_inliers),
                    }

        # Re-seed for next call regardless of this call's outcome, so a
        # single bad essential-matrix solve doesn't permanently wedge
        # tracking on a shrinking/stale point set.
        self._prev_gray = gray
        self._prev_points = self._detect_features(gray)

        return result
