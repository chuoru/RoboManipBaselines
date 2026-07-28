from .RealUMIEnvBase import RealUMIEnvBase


class RealUMIDemoEnv(RealUMIEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        RealUMIEnvBase.__init__(
            self,
            **kwargs,
        )

    def modify_world(self, world_idx=None, cumulative_idx=None):
        """Modify simulation world depending on world index."""
        # TODO: Automatically set world index according to task variations
        if world_idx is None:
            world_idx = 0
        return world_idx
