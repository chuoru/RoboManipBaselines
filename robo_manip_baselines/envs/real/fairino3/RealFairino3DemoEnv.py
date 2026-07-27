import numpy as np

from .RealFairino3EnvBase import RealFairino3EnvBase


class RealFairino3DemoEnv(RealFairino3EnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        RealFairino3EnvBase.__init__(
            self,
            # TODO: This is a placeholder ready pose for the FR3 arm. Verify it against
            # the official specification before running on real hardware.
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
