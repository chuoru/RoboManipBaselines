import numpy as np

from .RealFairino5EnvBase import RealFairino5EnvBase


class RealFairino5DemoEnv(RealFairino5EnvBase):
    # TODO: This is the FR3 ready pose reused as a placeholder for the FR5 arm.
    # Verify it against the official FR5 specification -- the joint ranges match
    # FR3's, but the link geometry differs -- before running on real hardware.
    # To capture the real FR5 value: move the arm (teach pendant drag/free-drive)
    # to a safe, non-singular ready pose, then read it back with
    # `Robot.RPC(robot_ip).GetActualJointPosRadian()` and replace this constant.
    # Also referenced by misc/CheckUmiFairino5Reachability.py so both stay in sync.
    READY_POSE_DEG = [-65.000, -93.000, -142.000, 56.000, 64.000, -1.000]

    def __init__(
        self,
        **kwargs,
    ):
        RealFairino5EnvBase.__init__(
            self,
            init_qpos=np.concatenate(
                [np.deg2rad(self.READY_POSE_DEG), np.array([0.0])]
            ),
            **kwargs,
        )

    def modify_world(self, world_idx=None, cumulative_idx=None):
        """Modify simulation world depending on world index."""
        # TODO: Automatically set world index according to task variations
        if world_idx is None:
            world_idx = 0
            # world_idx = cumulative_idx % 2
        return world_idx
