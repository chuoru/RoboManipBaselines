import gymnasium as gym
import numpy as np

from robo_manip_baselines.common import GraspPhaseBase


class GraspPhase(GraspPhaseBase):
    def set_target(self):
        self.gripper_joint_pos = np.array([0.0])
        self.duration = 0.5  # [s]


class OperationRealFairinoDemo:
    def __init__(
        self,
        robot_ip,
        camera_ids=None,
        gelsight_ids=None,
        gripper_hand_type="right",
        gripper_modbus_port="/dev/ttyUSB0",
    ):
        self.robot_ip = robot_ip
        self.camera_ids = camera_ids
        self.gelsight_ids = gelsight_ids
        self.gripper_hand_type = gripper_hand_type
        self.gripper_modbus_port = gripper_modbus_port
        super().__init__()

    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/RealFairinoDemoEnv-v0",
            robot_ip=self.robot_ip,
            camera_ids=self.camera_ids,
            gelsight_ids=self.gelsight_ids,
            gripper_hand_type=self.gripper_hand_type,
            gripper_modbus_port=self.gripper_modbus_port,
        )

    def get_pre_motion_phases(self):
        return [GraspPhase(self)]
