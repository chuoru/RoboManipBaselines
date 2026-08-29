import gymnasium as gym
import numpy as np

from robo_manip_baselines.common import GraspPhaseBase, PhaseBase


class MoveToInitPhase(PhaseBase):
    """Physically moves both arms/grippers to the reset pose and enables streaming
    teleop commands. Runs once, as the first pre-motion phase (i.e. after the
    operator has pressed 'n' past InitialTeleopPhase, and right before
    StandbyTeleopPhase begins), rather than immediately at env.reset() -- so the
    arms do not move on their own before the operator is at the controls."""

    def start(self):
        super().start()
        self.op.env.unwrapped.move_to_init_pose()
        # move_to_init_pose() is a blocking MoveJ that happens OUTSIDE the
        # env.step() loop, so op.obs still describes where the arms were
        # before the move. Anything that re-anchors on the measured position
        # from op.obs -- notably MotionManager.sync_arm_to_measured(), which
        # RolloutBase.run() calls every tick -- would otherwise seed the IK
        # with the PRE-move pose and command the arm straight back to it.
        # Measured on the real FR5: a 13.8 deg step demanded on the very
        # first tick after the move, executed at the full velocity-clamp
        # limit -- a violent lurch the instant the rollout started. Refresh
        # the observation here so the first post-move command starts from
        # where the arms actually are.
        self.op.obs = self.op.env.unwrapped._get_obs()

    def check_transition(self):
        return True  # move_to_init_pose() is blocking, so proceed immediately


class GraspPhase(GraspPhaseBase):
    def set_target(self):
        self.gripper_joint_pos = np.array([0.0, 0.0])
        self.duration = 0.5  # [s]


class OperationRealFairinoDualDemo:
    def __init__(
        self,
        robot_ip_left,
        robot_ip_right,
        camera_ids=None,
        gelsight_ids=None,
        pointcloud_camera_ids=None,
        gripper_hand_type_left="left",
        gripper_hand_type_right="right",
        gripper_modbus_port_left="/dev/ttyUSB0",
        gripper_modbus_port_right="/dev/ttyUSB1",
        dry_run=False,
    ):
        self.robot_ip_left = robot_ip_left
        self.robot_ip_right = robot_ip_right
        self.camera_ids = camera_ids
        self.gelsight_ids = gelsight_ids
        self.pointcloud_camera_ids = pointcloud_camera_ids
        self.gripper_hand_type_left = gripper_hand_type_left
        self.gripper_hand_type_right = gripper_hand_type_right
        self.gripper_modbus_port_left = gripper_modbus_port_left
        self.gripper_modbus_port_right = gripper_modbus_port_right
        self.dry_run = dry_run
        super().__init__()

    def setup_env(self, render_mode="human"):
        self.env = gym.make(
            "robo_manip_baselines/RealFairinoDualDemoEnv-v0",
            robot_ip_left=self.robot_ip_left,
            robot_ip_right=self.robot_ip_right,
            camera_ids=self.camera_ids,
            gelsight_ids=self.gelsight_ids,
            pointcloud_camera_ids=self.pointcloud_camera_ids,
            gripper_hand_type_left=self.gripper_hand_type_left,
            gripper_hand_type_right=self.gripper_hand_type_right,
            gripper_modbus_port_left=self.gripper_modbus_port_left,
            gripper_modbus_port_right=self.gripper_modbus_port_right,
            dry_run=self.dry_run,
        )

    def get_pre_motion_phases(self):
        return [MoveToInitPhase(self), GraspPhase(self)]
