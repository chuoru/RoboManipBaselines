import gymnasium as gym


class OperationRealUMIDemo:
    """Operation for the UMI (Universal Manipulation Interface)-style
    handheld demo-collection rig (RealUMIEnvBase). Unlike the
    teleoperated-robot operations, there is no physical arm to move to an
    init pose before standby teleop begins, so (unlike e.g.
    OperationRealFairino3Demo) this does not override
    get_pre_motion_phases() -- TeleopBase's default (no pre-motion phases) is
    used as-is.
    """

    def __init__(
        self,
        camera_ids=None,
        pointcloud_camera_ids=None,
        gelsight_ids=None,
    ):
        self.camera_ids = camera_ids
        self.pointcloud_camera_ids = pointcloud_camera_ids
        self.gelsight_ids = gelsight_ids
        super().__init__()

    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/RealUMIDemoEnv-v0",
            camera_ids=self.camera_ids,
            pointcloud_camera_ids=self.pointcloud_camera_ids,
            gelsight_ids=self.gelsight_ids,
        )
