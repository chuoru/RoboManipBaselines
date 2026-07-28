import time
from os import path

import numpy as np
from gymnasium.spaces import Box, Dict

from robo_manip_baselines.common import ArmConfig
from robo_manip_baselines.teleop import (
    KeyboardInputDevice,
    SpacemouseInputDevice,
    ViveInputDevice,
)

from ..RealEnvBase import RealEnvBase


class RealUMIEnvBase(RealEnvBase):
    """Base env for a UMI (Universal Manipulation Interface)-style handheld
    demonstration-collection rig: a camera-equipped gripper with no attached
    robot arm, whose end-effector pose is driven directly by an external 6-DoF
    pose source (an HTC Vive Tracker, via ViveInputDevice) rather than by
    teleoperating a real robot.

    Unlike the other Real*EnvBase classes, step()/_set_action() here do not
    command any motor: this rig has none. The commanded pose/gripper state is
    simply echoed back as the "measured" state on the next _get_obs(), since
    for a passively-tracked handheld device there is no separate physical
    plant to converge toward the command -- the human's hand motion *is* the
    trajectory (mirroring UMI's own design, where the collected pose is
    simultaneously the observation and the action).

    The arm_urdf_path below points at a virtual free-flyer joint (see
    umi.urdf) rather than a real robot's kinematic chain, so that
    this repo's existing ArmManager/MotionManager/DataKey machinery (which
    several call sites -- e.g. MotionManager.get_measured_data's
    isinstance(ArmManager) check -- assume is backed by a URDF and IK, not
    just any BodyManagerBase) can be reused unmodified to record
    MEASURED_EEF_POSE/COMMAND_EEF_POSE for this robot-less body. See the
    caveat in that URDF file: this has not been verified against a real
    Pinocchio install.
    """

    # Percent-closed gripper convention (0 = fully open, 100 = fully closed),
    # matching RealFairino3EnvBase's gripper_percent_closed convention so the
    # existing Vive gripper-scale toggle logic (ViveInputDevice.gripper_scale)
    # works unchanged.
    action_space = Box(
        low=np.array([-2.0, -2.0, -2.0, -1.0, -1.0, -1.0, -1.0, 0.0], dtype=np.float32),
        high=np.array([2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 100.0], dtype=np.float32),
        dtype=np.float32,
    )
    observation_space = Dict(
        {
            "joint_pos": Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64),
            "joint_vel": Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64),
            "wrench": Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
        }
    )

    def __init__(
        self,
        camera_ids=None,
        pointcloud_camera_ids=None,
        gelsight_ids=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Identity free-flyer configuration: (x, y, z, qx, qy, qz, qw).
        init_arm_joint_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        # Percent-closed; 0 = open.
        init_gripper_joint_pos = np.zeros(1)

        self.body_config_list = [
            ArmConfig(
                arm_urdf_path=path.join(
                    path.dirname(__file__),
                    "../../assets/common/robots/umi/umi.urdf",
                ),
                arm_root_pose=None,
                ik_eef_joint_id=1,
                arm_joint_idxes=np.arange(7),
                gripper_joint_idxes=np.array([7]),
                gripper_joint_idxes_in_gripper_joint_pos=np.array([0]),
                eef_idx=0,
                init_arm_joint_pos=init_arm_joint_pos,
                init_gripper_joint_pos=init_gripper_joint_pos,
            )
        ]

        self.arm_joint_pos_actual = init_arm_joint_pos.copy()
        self.gripper_joint_pos_actual = init_gripper_joint_pos.copy()

        # This rig has no force/torque sensor.
        self.joint_vel_limit = np.inf

        self.setup_realsense(camera_ids)
        self.setup_femtobolt(pointcloud_camera_ids)
        self.setup_gelsight(gelsight_ids)

    def setup_input_device(self, input_device_name, motion_manager, overwrite_kwargs):
        if input_device_name == "spacemouse":
            InputDeviceClass = SpacemouseInputDevice
        elif input_device_name == "keyboard":
            InputDeviceClass = KeyboardInputDevice
        elif input_device_name == "vive":
            InputDeviceClass = ViveInputDevice
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid input device key: {input_device_name}"
            )

        default_kwargs = self.get_input_device_kwargs(input_device_name)

        return [
            InputDeviceClass(
                motion_manager.body_manager_list[0],
                **{**default_kwargs, **overwrite_kwargs},
            )
        ]

    def _reset_robot(self):
        self.arm_joint_pos_actual = self.body_config_list[0].init_arm_joint_pos.copy()
        self.gripper_joint_pos_actual = self.body_config_list[
            0
        ].init_gripper_joint_pos.copy()

    def _set_action(self, action, duration=None, joint_vel_limit_scale=0.5, wait=False):
        # There is no motor to command and thus no joint-velocity limit to
        # protect, so (unlike the teleoperated-robot envs) this deliberately
        # skips overwrite_command_for_safety(): its per-component clamp is
        # meant for real revolute-joint velocities, and does not have a
        # physically meaningful interpretation applied directly to a
        # free-flyer's raw (x, y, z, qx, qy, qz, qw) configuration vector.
        start_time = time.time()

        if not np.all(np.isfinite(action)):
            raise RuntimeError(
                f"[{self.__class__.__name__}] Action contains NaN or Inf: {action}"
            )

        arm_joint_pos = action[self.body_config_list[0].arm_joint_idxes]
        gripper_joint_pos = action[self.body_config_list[0].gripper_joint_idxes]

        # The commanded pose is the measured pose: a passively-tracked UMI-
        # style rig has no separate physical plant to converge toward the
        # command.
        self.arm_joint_pos_actual = np.array(arm_joint_pos, dtype=np.float64)
        self.gripper_joint_pos_actual = np.array(gripper_joint_pos, dtype=np.float64)

        # TODO: if the handheld gripper is motorized (rather than a passive
        # spring-loaded trigger design), send gripper_joint_pos to its
        # actuator driver here.

        if wait and duration is not None:
            elapsed_duration = time.time() - start_time
            if elapsed_duration < duration:
                time.sleep(duration - elapsed_duration)

    def _get_obs(self):
        # TODO: if the handheld gripper has a real width sensor (rather than
        # relying on ViveInputDevice's button-toggle gripper_scale
        # integrator), read its sensed width here instead of echoing back the
        # last commanded value.
        return {
            "joint_pos": np.concatenate(
                (self.arm_joint_pos_actual, self.gripper_joint_pos_actual),
                dtype=np.float64,
            ),
            "joint_vel": np.zeros(8, dtype=np.float64),
            "wrench": np.zeros(6, dtype=np.float64),
        }
