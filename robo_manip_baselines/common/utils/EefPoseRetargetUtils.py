import numpy as np
import pinocchio as pin


class EpisodeRelativeEefPoseRetargeter:
    """Converts between an env's ABSOLUTE end-effector SE3 (as used by
    ArmManager's forward/inverse kinematics for actual robot control) and the
    EPISODE-RELATIVE convention that DataKey.MEASURED_EEF_POSE/COMMAND_EEF_POSE
    carry in UMI-collected training data (see envs/real/umi/RealUMIEnvBase.py):
    every episode's own pose starts at SE3 identity, and motion since then is
    represented as a per-frame EEF/TCP-local composition, not a plain "current
    minus start" delta.

    This reproduces the exact retargeting math validated in
    misc/ReplayUmiOnFairino5.py (see that module's docstring for the full
    derivation and the wrong versions it went through before landing there),
    generalized to run causally/incrementally one tick at a time -- both
    forward (to_absolute: episode-relative predicted pose -> absolute SE3,
    for interpreting a policy's predicted COMMAND_EEF_POSE) and, by algebraic
    inversion of the same formulas, backward (to_episode_relative: absolute
    measured SE3 -> episode-relative pose, for reporting MEASURED_EEF_POSE to
    a policy in the convention it was trained on).

    `ref_se3` is the ABSOLUTE pose that the episode's own convention starts
    from (identity) -- ordinarily the arm's actual current_se3 at the moment
    policy control begins.
    """

    def __init__(self, ref_se3, pos_scale=1.0):
        self.ref_se3 = ref_se3.copy()
        self.pos_scale = pos_scale

        self._inv_prev_measured_se3 = None
        self._inv_translation = np.zeros(3)

    def to_absolute(self, rel_se3):
        """Forward: given the policy's raw predicted COMMAND_EEF_POSE (in
        episode-relative/UMI convention), return the ABSOLUTE SE3 target to
        feed ArmManager.set_command_eef_pose: simply ref_se3 composed with the
        (position-scaled) relative pose.

        This is the exact inverse of to_episode_relative() below (whose
        incremental update reduces algebraically to
        p_rel = ref.R^T (p_abs - p_ref) / pos_scale, R_rel = ref.R^T R_abs).

        It used to be incremental instead -- anchored to the FIRST input and
        accumulating only the changes between successive inputs. That is
        identical when the input sequence starts at identity (a recorded demo
        replayed by misc/ReplayUmiOnFairino5.py), but wrong for a policy: a
        policy's predictions are absolute targets in the episode frame, and
        its very first one is already offset from identity (it predicts the
        next pose). Anchoring on it discarded that offset, and while the arm
        then stood still the policy kept predicting the same pose, so the
        target never left the start pose -- observed on the real FR5 as ACT
        rollouts that did not move at all and Diffusion Policy rollouts that
        only jittered by their own prediction noise.
        """
        target_rotation = self.ref_se3.rotation @ rel_se3.rotation
        target_translation = self.ref_se3.translation + self.pos_scale * (
            self.ref_se3.rotation @ rel_se3.translation
        )
        return pin.SE3(target_rotation, target_translation)

    def to_episode_relative(self, abs_se3):
        """Inverse: given the arm's current ABSOLUTE measured SE3, return the
        episode-relative/UMI-convention pose to report as MEASURED_EEF_POSE --
        the algebraic inverse of to_absolute(), assuming (matching the
        training-data convention) the episode-relative rotation starts at
        identity. The first call establishes the reference and returns
        identity unchanged."""
        input_rotation = self.ref_se3.rotation.T @ abs_se3.rotation

        if self._inv_prev_measured_se3 is None:
            self._inv_prev_measured_se3 = abs_se3.copy()
            return pin.SE3(input_rotation, self._inv_translation.copy())

        raw_absolute_delta = (
            abs_se3.translation - self._inv_prev_measured_se3.translation
        )
        local_increment = (abs_se3.rotation.T @ raw_absolute_delta) / self.pos_scale
        self._inv_translation = self._inv_translation + input_rotation @ local_increment
        self._inv_prev_measured_se3 = abs_se3.copy()

        return pin.SE3(input_rotation, self._inv_translation.copy())
