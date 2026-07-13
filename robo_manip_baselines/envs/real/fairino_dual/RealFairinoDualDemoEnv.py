import numpy as np

from .RealFairinoDualEnvBase import RealFairinoDualEnvBase


class RealFairinoDualDemoEnv(RealFairinoDualEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        RealFairinoDualEnvBase.__init__(
            self,
            # Left arm index: 0:6 (+gripper at 6), right arm index: 7:13 (+gripper at
            # 13). This follows the same layout convention as the other Dual envs.
            # TODO: This reuses the single-arm RealFairinoDemoEnv reset pose for both
            # arms as a placeholder; verify/replace with poses appropriate for the
            # dual-arm mount before running on real hardware.
            init_qpos=np.concatenate(
                [
                    np.deg2rad([44.088, -55.886, 109.266, -231.54, -90, 0.000 ]),
                    np.array([0.0]),
                    np.deg2rad([-131, -118, -108, 46, 87, 0.000 ]),
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
