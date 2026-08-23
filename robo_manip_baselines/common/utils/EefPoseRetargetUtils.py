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

        self._fwd_input_rotation_0 = None
        self._fwd_prev_input_se3 = None
        self._fwd_translation = ref_se3.translation.copy()

        self._inv_prev_measured_se3 = None
        self._inv_translation = np.zeros(3)

    def to_absolute(self, rel_se3):
        """Forward: given the policy's raw predicted COMMAND_EEF_POSE (in
        episode-relative/UMI convention), return the ABSOLUTE SE3 target to
        feed ArmManager.set_command_eef_pose. The first call establishes the
        episode's own rotation reference and returns ref_se3 unchanged."""
        if self._fwd_input_rotation_0 is None:
            self._fwd_input_rotation_0 = rel_se3.rotation.copy()
            self._fwd_prev_input_se3 = rel_se3.copy()

        delta_rotation = self._fwd_input_rotation_0.T @ rel_se3.rotation
        target_rotation = self.ref_se3.rotation @ delta_rotation

        raw_translation_delta = (
            rel_se3.translation - self._fwd_prev_input_se3.translation
        )
        translation_delta_local = rel_se3.rotation.T @ raw_translation_delta
        self._fwd_translation = self._fwd_translation + self.pos_scale * (
            target_rotation @ translation_delta_local
        )
        self._fwd_prev_input_se3 = rel_se3.copy()

        return pin.SE3(target_rotation, self._fwd_translation.copy())

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
