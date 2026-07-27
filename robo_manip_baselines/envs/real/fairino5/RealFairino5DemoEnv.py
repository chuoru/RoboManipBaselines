import numpy as np

from .RealFairino5EnvBase import RealFairino5EnvBase


class RealFairino5DemoEnv(RealFairino5EnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        RealFairino5EnvBase.__init__(
            self,
            # TODO: This is the FR3 ready pose reused as a placeholder for the FR5
            # arm. Verify it against the official FR5 specification -- the joint
            # ranges match FR3's, but the link geometry differs -- before running on
            # real hardware.
            init_qpos=np.concatenate(
                [
                    np.deg2rad(
                        [44.088, -55.886, 109.266, -231.54, -89.854, 89.605]
                    ),
                    np.array([0.0]),
                ]
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
