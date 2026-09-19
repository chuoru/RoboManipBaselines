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
        gelsight_ids=None,
        m5stack_ids=None,
        insta360_ids=None,
        pointcloud_camera_ids=None,
        pointcloud_camera_color_resolution=None,
        pointcloud_camera_color_exposure=None,
        pointcloud_camera_color_gain=None,
        gripper_marker_config=None,
    ):
        self.camera_ids = camera_ids
        self.gelsight_ids = gelsight_ids
        self.m5stack_ids = m5stack_ids
        self.insta360_ids = insta360_ids
        self.pointcloud_camera_ids = pointcloud_camera_ids
        self.pointcloud_camera_color_resolution = pointcloud_camera_color_resolution
        self.pointcloud_camera_color_exposure = pointcloud_camera_color_exposure
        self.pointcloud_camera_color_gain = pointcloud_camera_color_gain
        self.gripper_marker_config = gripper_marker_config
        super().__init__()

    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/RealUMIDemoEnv-v0",
            camera_ids=self.camera_ids,
            gelsight_ids=self.gelsight_ids,
            m5stack_ids=self.m5stack_ids,
            insta360_ids=self.insta360_ids,
            pointcloud_camera_ids=self.pointcloud_camera_ids,
            pointcloud_camera_color_resolution=self.pointcloud_camera_color_resolution,
            pointcloud_camera_color_exposure=self.pointcloud_camera_color_exposure,
            pointcloud_camera_color_gain=self.pointcloud_camera_color_gain,
            gripper_marker_config=self.gripper_marker_config,
        )
