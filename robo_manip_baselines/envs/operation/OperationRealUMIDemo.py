import gymnasium as gym
import numpy as np


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
        record_hz=None,
        dataset_dir=None,
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
        self.record_hz = record_hz
        # Where TeleopBase.get_save_filename() puts recorded episodes (None:
        # the repo's dataset/).
        self.dataset_dir = dataset_dir
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
            record_hz=self.record_hz,
        )

    def record_data(self):
        # The gripper is read from the finger AprilTags, not commanded (the
        # Vive button is not used on this rig), so without this the recorded
        # command_gripper_joint_pos -- and the gripper element of
        # command_joint_pos, which policies train on as the action -- would
        # sit at its initial 0 for the whole episode. Mirror UMI's own
        # convention instead: the gripper action is the measured gripper
        # state, just as the Vive pose is both the observation and the
        # action for the arm. Teleop only (input_device_list exists): during
        # a rollout the command must stay whatever the policy output.
        if hasattr(self, "input_device_list"):
            percent_closed = self.env.unwrapped.latest_gripper_marker_percent_closed
            if percent_closed is not None:
                self.motion_manager.body_manager_list[0].set_command_gripper_joint_pos(
                    np.array([percent_closed])
                )
        super().record_data()
