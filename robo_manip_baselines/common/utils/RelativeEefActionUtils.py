"""UMI-style relative end-effector trajectories.

With --relative_eef_action, the COMMAND_EEF_POSE action chunk is expressed
relative to the end-effector pose at the observation (inference) time:

    action_k = T_obs^-1 * T_k        (T = SE3 of the pose)

as in the Universal Manipulation Interface. Compared with per-step relative
motions (command_eef_pose_rel), the targets are larger and smoother (noise
does not dominate a few-millimetre step), errors do not accumulate across
steps (every inference re-anchors on the current pose), and nothing depends
on where the episode started. At rollout time the chunk is turned back into
absolute targets with the robot's current pose at inference time.

Poses are 7D [x, y, z, qw, qx, qy, qz] (RMB/env representation); relative
actions are 9D (position + 6D rotation), the policy representation of
COMMAND_EEF_POSE.
"""

import numpy as np

from .MathUtils import get_pose_from_se3, get_se3_from_pose
from .PoseUtils import get_pose7_from_pose9, get_pose9_from_pose7


def get_relative_eef_pose9(pose7_seq, ref_pose7):
    """(N, 7) absolute poses -> (N, 9) poses relative to ref_pose7."""
    ref_inv = get_se3_from_pose(np.asarray(ref_pose7, dtype=np.float64)).inverse()
    rel_pose7 = np.stack(
        [get_pose_from_se3(ref_inv * get_se3_from_pose(p)) for p in np.asarray(pose7_seq)]
    )
    return get_pose9_from_pose7(rel_pose7)


def get_absolute_eef_pose9(rel_pose9_seq, ref_pose7):
    """(N, 9) poses relative to ref_pose7 -> (N, 9) absolute poses."""
    ref = get_se3_from_pose(np.asarray(ref_pose7, dtype=np.float64))
    rel_pose7 = get_pose7_from_pose9(np.asarray(rel_pose9_seq, dtype=np.float64))
    abs_pose7 = np.stack([get_pose_from_se3(ref * get_se3_from_pose(p)) for p in rel_pose7])
    return get_pose9_from_pose7(abs_pose7)
