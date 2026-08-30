import gymnasium as gym
import numpy as np

from robo_manip_baselines.common import GraspPhaseBase, PhaseBase


class MoveToInitPhase(PhaseBase):
    """Physically moves the arm/gripper to the reset pose and enables streaming
    teleop commands. Runs once, as the first pre-motion phase (i.e. after the
    operator has pressed 'n' past InitialTeleopPhase, and right before
    StandbyTeleopPhase begins), rather than immediately at env.reset() -- so the
    arm does not move on its own before the operator is at the controls."""

    def start(self):
        super().start()
        self.op.env.unwrapped.move_to_init_pose()
        # move_to_init_pose() is a blocking MoveJ that happens OUTSIDE the
        # env.step() loop, so op.obs still describes where the arm was
        # before the move. Anything that re-anchors on the measured position
        # from op.obs -- notably MotionManager.sync_arm_to_measured(), which
        # RolloutBase.run() calls every tick -- would otherwise seed the IK
        # with the PRE-move pose and command the arm straight back to it.
        # Measured on the real FR5: a 13.8 deg step demanded on the very
        # first tick after the move, executed at the full velocity-clamp
        # limit -- a violent lurch the instant the rollout started. Refresh
        # the observation here so the first post-move command starts from
        # where the arm actually is.
        self.op.obs = self.op.env.unwrapped._get_obs()

    def check_transition(self):
        return True  # move_to_init_pose() is blocking, so proceed immediately


class GraspPhase(GraspPhaseBase):
    def set_target(self):
        self.gripper_joint_pos = np.array([0.0])
        self.duration = 0.5  # [s]


class OperationRealFairino5Demo:
    def __init__(
        self,
        robot_ip,
        camera_ids=None,
        gelsight_ids=None,
        pointcloud_camera_ids=None,
        gripper_hand_type="right",
        gripper_modbus_port="/dev/ttyUSB0",
        gripper_type="tool_do",
        gripper_do_close_id=0,
        gripper_do_open_id=1,
        observe_only=False,
        dry_run=False,
        command_log_path=None,
        # Scales the arm's per-step joint velocity clamp (see
        # RealEnvBase.joint_vel_limit_scale). Lower than the default 2.0 to
        # slow the arm's physical top speed for a rollout, independent of
        # --skip (which instead controls how often the policy
        # observes/re-infers -- see RolloutBase.infer_policy).
        joint_vel_limit_scale=2.0,
        # Stretches the control period, slowing the whole rollout uniformly
        # (2.0 = half speed). This is the knob for slowing a rollout down --
        # see RealEnvBase.time_scale for why joint_vel_limit_scale is not.
        time_scale=1.0,
        # HARD cap [deg] on how far any joint may move between two
        # consecutive commands actually sent to the arm -- see
        # RealFairino5EnvBase.max_joint_pos_delta_deg. None disables it.
        max_joint_pos_delta_deg=None,
        # EMA smoothing applied to the commanded joint position before it is
        # sent (see RealFairino5EnvBase.command_smoothing_alpha). Lower =
        # smoother but laggier; 1.0 = no filtering.
        command_smoothing_alpha=0.3,
    ):
        self.robot_ip = robot_ip
        self.camera_ids = camera_ids
        self.gelsight_ids = gelsight_ids
        self.pointcloud_camera_ids = pointcloud_camera_ids
        self.gripper_hand_type = gripper_hand_type
        self.gripper_modbus_port = gripper_modbus_port
        self.gripper_type = gripper_type
        self.gripper_do_close_id = gripper_do_close_id
        self.gripper_do_open_id = gripper_do_open_id
        self.observe_only = observe_only
        self.dry_run = dry_run
        self.command_log_path = command_log_path
        self.joint_vel_limit_scale = joint_vel_limit_scale
        self.time_scale = time_scale
        self.max_joint_pos_delta_deg = max_joint_pos_delta_deg
        self.command_smoothing_alpha = command_smoothing_alpha
        super().__init__()

    def setup_env(self, render_mode="human"):
        # A --time_scale on the command line wins over the config file, so the
        # rollout speed can be swept without editing the config. getattr()
        # because the teleop entry point's parser has no such argument.
        time_scale = self.time_scale
        time_scale_arg = getattr(getattr(self, "args", None), "time_scale", None)
        if time_scale_arg is not None:
            time_scale = time_scale_arg

        self.env = gym.make(
            "robo_manip_baselines/RealFairino5DemoEnv-v0",
            robot_ip=self.robot_ip,
            camera_ids=self.camera_ids,
            gelsight_ids=self.gelsight_ids,
            pointcloud_camera_ids=self.pointcloud_camera_ids,
            gripper_hand_type=self.gripper_hand_type,
            gripper_modbus_port=self.gripper_modbus_port,
            gripper_type=self.gripper_type,
            gripper_do_close_id=self.gripper_do_close_id,
            gripper_do_open_id=self.gripper_do_open_id,
            observe_only=self.observe_only,
            dry_run=self.dry_run,
            command_log_path=self.command_log_path,
            joint_vel_limit_scale=self.joint_vel_limit_scale,
            time_scale=time_scale,
            max_joint_pos_delta_deg=self.max_joint_pos_delta_deg,
            command_smoothing_alpha=self.command_smoothing_alpha,
        )

    def get_pre_motion_phases(self):
        return [MoveToInitPhase(self), GraspPhase(self)]
