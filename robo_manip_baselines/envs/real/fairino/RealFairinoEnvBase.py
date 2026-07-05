import time
from os import path

import numpy as np
from fairino import Robot
from gymnasium.spaces import Box, Dict
from LinkerHand.linker_hand_api import LinkerHandApi

from robo_manip_baselines.common import ArmConfig
from robo_manip_baselines.teleop import (
    GelloInputDevice,
    KeyboardInputDevice,
    SpacemouseInputDevice,
)

from ..RealEnvBase import RealEnvBase


class RealFairinoEnvBase(RealEnvBase):
    # TODO: These are placeholder joint limits for the FR3 arm. Verify them against the
    # official specification before running on real hardware.
    action_space = Box(
        low=np.array(
            [
                np.deg2rad(-175),
                np.deg2rad(-175),
                np.deg2rad(-175),
                np.deg2rad(-175),
                np.deg2rad(-175),
                np.deg2rad(-175),
                0.0,
            ],
            dtype=np.float32,
        ),
        high=np.array(
            [
                np.deg2rad(175),
                np.deg2rad(175),
                np.deg2rad(175),
                np.deg2rad(175),
                np.deg2rad(175),
                np.deg2rad(175),
                100.0,
            ],
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    observation_space = Dict(
        {
            "joint_pos": Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
            "joint_vel": Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
            "wrench": Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
        }
    )

    # This arm has no external axes, so ServoJ's axis position is always zero
    # (see the ServoJ usage in fairino-python-sdk's example/servo.py)
    EXAXIS_POS = [0.0, 0.0, 0.0, 0.0]

    # The LinkerHand's thumb abduction axis ("thumb_cmc_yaw") is held at this fixed
    # close percentage regardless of the commanded gripper open/close position.
    GRIPPER_THUMB_ABDUCTION_CLOSE_PERCENT = 50.0

    def __init__(
        self,
        robot_ip,
        camera_ids,
        gelsight_ids,
        init_qpos,
        gripper_hand_type="right",
        gripper_modbus_port="/dev/ttyUSB0",
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Setup robot
        self.init_qpos = init_qpos
        # TODO: Verify against the official FR3 max joint speed before running on real hardware.
        self.joint_vel_limit = np.deg2rad(180)  # [rad/s]
        self.body_config_list = [
            ArmConfig(
                # TODO: Add the FR3 URDF asset at this path before running on real hardware.
                arm_urdf_path=path.join(
                    path.dirname(__file__),
                    "../../assets/common/robots/fairino/fr3.urdf",
                ),
                arm_root_pose=None,
                ik_eef_joint_id=6,
                arm_joint_idxes=np.arange(6),
                gripper_joint_idxes=np.array([6]),
                gripper_joint_idxes_in_gripper_joint_pos=np.array([0]),
                eef_idx=0,
                init_arm_joint_pos=self.init_qpos[0:6],
                init_gripper_joint_pos=np.zeros(1),
            )
        ]

        # Connect to Fairino arm
        print(f"[{self.__class__.__name__}] Start connecting the Fairino arm.")
        self.robot_ip = robot_ip
        self.robot = Robot.RPC(self.robot_ip)
        self._check_fr_code(self.robot.RobotEnable(1))
        self._check_fr_code(self.robot.Mode(0))
        self.robot.ResetAllError()
        self._check_fr_code(self.robot.ServoMoveStart())
        fr_code, arm_joint_pos = self.robot.GetActualJointPosRadian()
        self._check_fr_code(fr_code)
        self.arm_joint_pos_actual = np.array(arm_joint_pos)
        print(f"[{self.__class__.__name__}] Finish connecting the Fairino arm.")

        # Connect to LinkerHand gripper
        print(f"[{self.__class__.__name__}] Start connecting the LinkerHand gripper.")
        self.gripper = LinkerHandApi(
            hand_type=gripper_hand_type,
            hand_joint="O6",
            modbus=gripper_modbus_port,
        )
        gripper_finger_order = self.gripper.get_finger_order()
        self.gripper_num_joints = len(gripper_finger_order)
        self.gripper_thumb_abduction_idx = gripper_finger_order.index("thumb_cmc_yaw")
        print(f"[{self.__class__.__name__}] Finish connecting the LinkerHand gripper.")

        # Connect to RealSense
        self.setup_realsense(camera_ids)
        self.setup_gelsight(gelsight_ids)

    def _check_fr_code(self, fr_code):
        if fr_code != 0:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Invalid Fairino API code: {fr_code}"
            )

    def close(self):
        self.robot.ServoMoveEnd()
        self.robot.CloseRPC()

    def _gripper_pose(self, percent_closed):
        finger_raw = self._gripper_percent_closed_to_raw(percent_closed)
        thumb_abduction_raw = self._gripper_percent_closed_to_raw(
            self.GRIPPER_THUMB_ABDUCTION_CLOSE_PERCENT
        )
        pose = [finger_raw] * self.gripper_num_joints
        pose[self.gripper_thumb_abduction_idx] = thumb_abduction_raw
        return pose

    @staticmethod
    def _gripper_percent_closed_to_raw(percent_closed):
        # LinkerHand joint values are 0 (fully closed) to 255 (fully open)
        return int(round(np.clip(255.0 * (1.0 - percent_closed / 100.0), 0, 255)))

    def setup_input_device(self, input_device_name, motion_manager, overwrite_kwargs):
        if input_device_name == "spacemouse":
            InputDeviceClass = SpacemouseInputDevice
        elif input_device_name == "gello":
            InputDeviceClass = GelloInputDevice
        elif input_device_name == "keyboard":
            InputDeviceClass = KeyboardInputDevice
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

    def get_input_device_kwargs(self, input_device_name):
        if input_device_name == "spacemouse":
            return {"pos_scale": 1.5e-2, "rpy_scale": 1e-2, "gripper_scale": 10.0}
        else:
            return super().get_input_device_kwargs(input_device_name)

    def _reset_robot(self):
        print(
            f"[{self.__class__.__name__}] Start moving the robot to the reset position."
        )
        self._set_action(
            self.init_qpos, duration=None, joint_vel_limit_scale=0.3, wait=True
        )
        print(
            f"[{self.__class__.__name__}] Finish moving the robot to the reset position."
        )

    def _set_action(self, action, duration=None, joint_vel_limit_scale=0.5, wait=False):
        start_time = time.time()

        # Overwrite duration or joint_pos for safety
        action, duration = self.overwrite_command_for_safety(
            action, duration, joint_vel_limit_scale
        )

        # Send command to Fairino arm
        arm_joint_pos_command_deg = np.rad2deg(
            action[self.body_config_list[0].arm_joint_idxes]
        )
        self._check_fr_code(
            self.robot.ServoJ(list(arm_joint_pos_command_deg), self.EXAXIS_POS)
        )

        # Send command to LinkerHand gripper
        gripper_percent_closed = float(
            action[self.body_config_list[0].gripper_joint_idxes][0]
        )
        self.gripper.finger_move(pose=self._gripper_pose(gripper_percent_closed))

        # Wait
        elapsed_duration = time.time() - start_time
        if wait and elapsed_duration < duration:
            time.sleep(duration - elapsed_duration)

    def _get_obs(self):
        # Get state from Fairino arm
        fr_code, arm_joint_pos = self.robot.GetActualJointPosRadian()
        self._check_fr_code(fr_code)
        arm_joint_pos = np.array(arm_joint_pos, dtype=np.float64)
        self.arm_joint_pos_actual = arm_joint_pos.copy()

        fr_code, arm_joint_vel_deg = self.robot.GetActualJointSpeedsDegree()
        self._check_fr_code(fr_code)
        arm_joint_vel = np.deg2rad(np.array(arm_joint_vel_deg, dtype=np.float64))

        # Get state from LinkerHand gripper
        gripper_pose_raw = np.array(self.gripper.get_state(), dtype=np.float64)
        finger_raw = np.delete(gripper_pose_raw, self.gripper_thumb_abduction_idx)
        gripper_percent_closed = 100.0 * (1.0 - finger_raw.mean() / 255.0)
        gripper_joint_pos = np.array([gripper_percent_closed], dtype=np.float64)
        gripper_joint_vel = np.zeros(1)

        # This arm has no force/torque sensor
        wrench = np.zeros(6, dtype=np.float64)

        return {
            "joint_pos": np.concatenate(
                (arm_joint_pos, gripper_joint_pos), dtype=np.float64
            ),
            "joint_vel": np.concatenate(
                (arm_joint_vel, gripper_joint_vel), dtype=np.float64
            ),
            "wrench": wrench,
        }
