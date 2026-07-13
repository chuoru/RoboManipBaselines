import socket
import time
from os import path

import numpy as np
from fairino import Robot
from gymnasium.spaces import Box, Dict

from robo_manip_baselines.common import ArmConfig
from robo_manip_baselines.teleop import (
    GelloInputDevice,
    KeyboardInputDevice,
    SpacemouseInputDevice,
    ViveInputDevice,
)

from ..RealEnvBase import RealEnvBase

# Official FR3 joint ranges: J1/J5/J6 are +-175 deg; J2/J4 are +85 to -265 deg;
# J3 is +-150 deg.
_ARM_LOW_DEG = [-175, -265, -150, -265, -175, -175]
_ARM_HIGH_DEG = [175, 85, 150, 85, 175, 175]

# This arm has no external axes, so ServoJ's axis position is always zero
# (see the ServoJ usage in fairino-python-sdk's example/servo.py)
EXAXIS_POS = [0.0, 0.0, 0.0, 0.0]

# The LinkerHand's thumb abduction axis ("thumb_cmc_yaw") is held at this fixed
# close percentage regardless of the commanded gripper open/close position.
GRIPPER_THUMB_ABDUCTION_CLOSE_PERCENT = 50.0

# Split-key bindings so ONE physical keyboard can drive both arms at once: the left
# arm keeps KeyboardInputDevice's default WASDQE/IJKLUO/ZX layout, and the right arm
# is remapped to a disjoint set of keys (numpad-style digits for translation, an
# unused letter cluster for rotation/gripper) so neither arm reacts to the other's
# keys.
LEFT_ARM_KEYBOARD_BINDINGS = {
    "x_pos": "w",
    "x_neg": "s",
    "y_pos": "a",
    "y_neg": "d",
    "z_pos": "q",
    "z_neg": "e",
    "roll_neg": "j",
    "roll_pos": "l",
    "pitch_pos": "i",
    "pitch_neg": "k",
    "yaw_pos": "u",
    "yaw_neg": "o",
    "gripper_close": "z",
    "gripper_open": "x",
}
RIGHT_ARM_KEYBOARD_BINDINGS = {
    "x_pos": "8",
    "x_neg": "2",
    "y_pos": "4",
    "y_neg": "6",
    "z_pos": "9",
    "z_neg": "3",
    "roll_neg": "f",
    "roll_pos": "h",
    "pitch_pos": "t",
    "pitch_neg": "g",
    "yaw_pos": "y",
    "yaw_neg": "b",
    "gripper_close": "n",
    "gripper_open": "m",
}


class RealFairinoDualEnvBase(RealEnvBase):
    action_space = Box(
        low=np.array(
            [np.deg2rad(v) for v in _ARM_LOW_DEG] + [0.0]
            + [np.deg2rad(v) for v in _ARM_LOW_DEG] + [0.0],
            dtype=np.float32,
        ),
        high=np.array(
            [np.deg2rad(v) for v in _ARM_HIGH_DEG] + [100.0]
            + [np.deg2rad(v) for v in _ARM_HIGH_DEG] + [100.0],
            dtype=np.float32,
        ),
        dtype=np.float32,
    )
    observation_space = Dict(
        {
            "joint_pos": Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float64),
            "joint_vel": Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float64),
            "wrench": Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64),
        }
    )

    def __init__(
        self,
        robot_ip_left,
        robot_ip_right,
        camera_ids,
        gelsight_ids,
        init_qpos,
        gripper_hand_type_left="left",
        gripper_hand_type_right="right",
        gripper_modbus_port_left="/dev/ttyUSB0",
        gripper_modbus_port_right="/dev/ttyUSB1",
        dry_run=False,
        command_smoothing_alpha=0.3,
        pointcloud_camera_ids=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.init_qpos = init_qpos
        self.dry_run = dry_run
        self.command_smoothing_alpha = command_smoothing_alpha
        # TODO: Verify against the official FR3 max joint speed before running on real hardware.
        self.joint_vel_limit = np.deg2rad(90)  # [rad/s]

        # Per-arm state, indexed [0]=left, [1]=right, mirroring the layout used by
        # action/observation (arm1(0:6), gripper1(6), arm2(7:13), gripper2(13)).
        self.robot_ips = [robot_ip_left, robot_ip_right]
        self.gripper_hand_types = [gripper_hand_type_left, gripper_hand_type_right]
        self.gripper_modbus_ports = [
            gripper_modbus_port_left,
            gripper_modbus_port_right,
        ]
        self.robots = [None, None]
        self.grippers = [None, None]
        self._last_gripper_percent_closed = [50.0, 50.0]
        self.gripper_num_joints = [None, None]
        self.gripper_thumb_abduction_idx = [None, None]
        self._last_servoj_time = [None, None]
        self._filtered_arm_joint_pos_command = [None, None]
        # Gates whether _set_action actually transmits ServoJ/gripper commands to
        # hardware. Starts False so the arms stay still (even though env.step() is
        # called continuously by the teleop loop from the very first frame) until
        # move_to_init_pose() explicitly enables it, right before StandbyTeleopPhase.
        self._motion_enabled = False

        self.body_config_list = [
            ArmConfig(
                arm_urdf_path=path.join(
                    path.dirname(__file__),
                    "../../assets/common/robots/fairinofr3/fairino3_v6.urdf",
                ),
                arm_root_pose=None,
                ik_eef_joint_id=6,
                arm_joint_idxes=np.arange(6),
                gripper_joint_idxes=np.array([6]),
                gripper_joint_idxes_in_gripper_joint_pos=np.array([0]),
                eef_idx=0,
                init_arm_joint_pos=self.init_qpos[0:6],
                init_gripper_joint_pos=np.zeros(1),
            ),
            ArmConfig(
                arm_urdf_path=path.join(
                    path.dirname(__file__),
                    "../../assets/common/robots/fairinofr3/fairino3_v6.urdf",
                ),
                arm_root_pose=None,
                ik_eef_joint_id=6,
                arm_joint_idxes=np.arange(7, 13),
                gripper_joint_idxes=np.array([13]),
                gripper_joint_idxes_in_gripper_joint_pos=np.array([1]),
                eef_idx=1,
                init_arm_joint_pos=self.init_qpos[7:13],
                init_gripper_joint_pos=np.zeros(1),
            ),
        ]

        self.arm_joint_pos_actual = np.zeros(12)

        if self.dry_run:
            print(
                f"[{self.__class__.__name__}] DRY RUN MODE: Skipping robot connection. "
                "Commands will be printed instead of executed."
            )
            for side in (0, 1):
                self.arm_joint_pos_actual[
                    self._arm_slice(side)
                ] = self.init_qpos[self.body_config_list[side].arm_joint_idxes]
        else:
            for side, side_name in ((0, "left"), (1, "right")):
                print(
                    f"[{self.__class__.__name__}] Start connecting the {side_name} Fairino arm."
                )
                self.robots[side] = self._connect_robot(self.robot_ips[side])
                self._enable_robot(side)
                self._start_servo_mode(side)
                fr_code, arm_joint_pos = self.robots[side].GetActualJointPosRadian()
                self._check_fr_code(fr_code)
                self.arm_joint_pos_actual[self._arm_slice(side)] = np.array(
                    arm_joint_pos
                )
                print(
                    f"[{self.__class__.__name__}] Finish connecting the {side_name} Fairino arm."
                )

                print(
                    f"[{self.__class__.__name__}] Start connecting the {side_name} LinkerHand gripper."
                )
                try:
                    from LinkerHand.linker_hand_api import LinkerHandApi
                except ImportError as e:
                    raise RuntimeError(
                        f"[{self.__class__.__name__}] Failed to import LinkerHand. "
                        f"Please ensure the LinkerHand package is installed. Error: {e}"
                    )
                self.grippers[side] = LinkerHandApi(
                    hand_type=self.gripper_hand_types[side],
                    hand_joint="O6",
                    modbus=self.gripper_modbus_ports[side],
                )
                gripper_finger_order = self.grippers[side].get_finger_order()
                self.gripper_num_joints[side] = len(gripper_finger_order)
                self.gripper_thumb_abduction_idx[side] = gripper_finger_order.index(
                    "thumb_cmc_yaw"
                )
                print(
                    f"[{self.__class__.__name__}] Finish connecting the {side_name} LinkerHand gripper."
                )

        # Connect to RealSense, Orbbec (femtobolt), and GelSight (only in
        # non-dry-run mode)
        if not self.dry_run:
            self.setup_realsense(camera_ids)
            self.setup_femtobolt(pointcloud_camera_ids)
            self.setup_gelsight(gelsight_ids)
        else:
            self.cameras = {}
            self.pointcloud_cameras = {}
            self.rgb_tactiles = {}
            self.intensity_tactiles = {}

    @staticmethod
    def _arm_slice(side):
        return slice(6 * side, 6 * (side + 1))

    def _check_fr_code(self, fr_code):
        if fr_code != 0:
            error_messages = {
                14: "Motion command failed - robot may not be in servo mode or motion is blocked",
                -1: "Command execution error",
                -2: "Parameter error",
                -3: "Robot not enabled",
                -4: "Communication error",
            }
            error_desc = error_messages.get(fr_code, "Unknown error")
            raise RuntimeError(
                f"[{self.__class__.__name__}] Fairino API error code {fr_code}: {error_desc}"
            )

    def _connect_robot(self, robot_ip):
        print(f"[{self.__class__.__name__}] Connecting to the Fairino arm at {robot_ip}.")
        robot = Robot.RPC(robot_ip)

        if not Robot.RPC.is_connect:
            # The vendored SDK only proceeds with XML-RPC calls (RobotEnable, ServoJ,
            # GetActualJointPosRadian, ...) once both the CNDE state-feed channel (port
            # 20005) and the XML-RPC command channel (port 20003) have connected. Some
            # controllers only expose the XML-RPC channel, so verify that channel
            # directly before giving up (mirrors the workaround in
            # fairino_keyboard_teleop/devices/fairino/interface.py, which was validated
            # against real hardware).
            print(
                f"[{self.__class__.__name__}] CNDE state channel (port 20005) is "
                "unreachable. Verifying the XML-RPC command channel (port 20003) "
                "independently."
            )
            xmlrpc_ok = False
            try:
                socket.setdefaulttimeout(1)
                robot.robot.GetControllerIP()
                xmlrpc_ok = True
            except Exception as e:
                print(f"[{self.__class__.__name__}] XML-RPC verification failed: {e}")
            finally:
                socket.setdefaulttimeout(None)

            if not xmlrpc_ok:
                raise RuntimeError(
                    f"[{self.__class__.__name__}] Failed to connect to the Fairino arm "
                    f"at {robot_ip} (neither the CNDE nor the XML-RPC channel is "
                    "reachable)."
                )
            print(
                f"[{self.__class__.__name__}] XML-RPC is reachable; proceeding without "
                "the CNDE state feed. Note: any SDK call that relies on the CNDE feed "
                "(e.g. GetRobotErrorCode/GetRobotMotionDone) may not be reliable in "
                "this mode."
            )
            Robot.RPC.is_connect = True

        return robot

    def _enable_robot(self, side):
        robot = self.robots[side]
        self._check_fr_code(robot.RobotEnable(1))
        self._check_fr_code(robot.Mode(0))
        self._check_fr_code(robot.ResetAllError())

    def _start_servo_mode(self, side):
        self._check_fr_code(self.robots[side].ServoMoveStart())
        time.sleep(1.5)
        self._last_servoj_time[side] = None

    def close(self):
        for robot in self.robots:
            if robot is None:
                continue
            try:
                robot.ServoMoveEnd()
            except Exception as e:
                print(f"[{self.__class__.__name__}] Error during ServoMoveEnd: {e}")
            try:
                robot.CloseRPC()
            except Exception as e:
                print(f"[{self.__class__.__name__}] Error during CloseRPC: {e}")

        super().close()

    def _gripper_pose(self, side, percent_closed):
        finger_raw = self._gripper_percent_closed_to_raw(percent_closed)
        thumb_abduction_raw = self._gripper_percent_closed_to_raw(
            GRIPPER_THUMB_ABDUCTION_CLOSE_PERCENT
        )
        pose = [finger_raw] * self.gripper_num_joints[side]
        pose[self.gripper_thumb_abduction_idx[side]] = thumb_abduction_raw
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
        elif input_device_name == "vive":
            InputDeviceClass = ViveInputDevice
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid input device key: {input_device_name}"
            )

        default_kwargs = self.get_input_device_kwargs(input_device_name)

        return [
            InputDeviceClass(
                body_manager,
                **{
                    **default_kwargs.get(device_idx, {}),
                    **overwrite_kwargs.get(device_idx, {}),
                },
            )
            for device_idx, body_manager in enumerate(motion_manager.body_manager_list)
        ]

    def get_input_device_kwargs(self, input_device_name):
        if input_device_name == "keyboard":
            common_kwargs = {
                "pos_scale": 1.5e-2,
                "rpy_scale": 1e-2,
                "gripper_scale": 10.0,
            }
            return {
                0: {**common_kwargs, "key_bindings": LEFT_ARM_KEYBOARD_BINDINGS},
                1: {**common_kwargs, "key_bindings": RIGHT_ARM_KEYBOARD_BINDINGS},
            }
        elif input_device_name == "spacemouse":
            return {
                0: {"pos_scale": 1.5e-2, "rpy_scale": 1e-2, "gripper_scale": 10.0},
                1: {"pos_scale": 1.5e-2, "rpy_scale": 1e-2, "gripper_scale": 10.0},
            }
        elif input_device_name == "vive":
            # gripper_key_bindings is intentionally omitted: ViveInputDevice
            # defaults to toggling the gripper open/closed via the tracker's own
            # button when no keyboard bindings are given.
            #
            # Unlike keyboard/spacemouse's pos_scale (a small per-frame increment
            # added while a key/axis is held), Vive's pos_scale multiplies the
            # tracker's *absolute* displacement from its anchor pose every frame,
            # so leaving it at ViveInputDevice's default of 1.0 makes the arm
            # track the tracker 1:1 at full speed. Start conservative (tracker
            # motion attenuated to 30%) and raise it once you've confirmed the
            # motion direction/scale feels safe.
            pos_scale = 0.1
            gripper_scale = 10.0
            return {
                0: {"pos_scale": pos_scale, "gripper_scale": gripper_scale},
                1: {"pos_scale": pos_scale, "gripper_scale": gripper_scale},
            }
        else:
            return {}

    def _reset_robot(self):
        # env.step() is called continuously by the teleop loop starting from the very
        # first frame (InitialTeleopPhase), before the operator has connected input
        # devices or is necessarily at the controls. So the physical move to the reset
        # pose is NOT performed here: this only re-syncs internal state so the safety
        # clamp in overwrite_command_for_safety compares against where the arms truly
        # are. The actual move happens in move_to_init_pose(), which
        # OperationRealFairinoDualDemo's MoveToInitPhase calls as the last pre-motion
        # phase, right before StandbyTeleopPhase begins.
        self._motion_enabled = False
        self._filtered_arm_joint_pos_command = [None, None]

        for side in (0, 1):
            if self.dry_run:
                self.arm_joint_pos_actual[self._arm_slice(side)] = self.init_qpos[
                    self.body_config_list[side].arm_joint_idxes
                ]
            else:
                fr_code, arm_joint_pos = self.robots[side].GetActualJointPosRadian()
                self._check_fr_code(fr_code)
                self.arm_joint_pos_actual[self._arm_slice(side)] = np.array(
                    arm_joint_pos
                )

    def move_to_init_pose(self):
        """Physically move both arms/grippers to the reset pose and enable streaming
        ServoJ commands from the teleop loop. Intended to be called once, right
        before standby teleop begins (see MoveToInitPhase in
        OperationRealFairinoDualDemo.py)."""
        print(
            f"[{self.__class__.__name__}] Start moving the robots to the reset position."
        )

        self._filtered_arm_joint_pos_command = [None, None]

        for side, side_name in ((0, "left"), (1, "right")):
            arm_joint_pos_command_deg = [
                float(x)
                for x in np.rad2deg(
                    self.init_qpos[self.body_config_list[side].arm_joint_idxes]
                )
            ]
            gripper_percent_closed = float(
                self.init_qpos[self.body_config_list[side].gripper_joint_idxes][0]
            )

            if self.dry_run:
                print(
                    f"[{self.__class__.__name__}] [DRY RUN] {side_name} arm would move "
                    f"to reset pose: {arm_joint_pos_command_deg} deg"
                )
                self.arm_joint_pos_actual[self._arm_slice(side)] = self.init_qpos[
                    self.body_config_list[side].arm_joint_idxes
                ]
            else:
                # ServoJ is a high-frequency streaming command: it expects to be
                # called repeatedly with small incremental targets, and interprets a
                # single large joint delta as needing to be reached almost instantly,
                # which trips the controller's axis speed limit. For the (potentially
                # large) one-shot move to the reset pose, use the blocking
                # point-to-point MoveJ command instead, then re-enter servo mode for
                # the streaming teleop control loop.
                robot = self.robots[side]
                self._check_fr_code(robot.ServoMoveEnd())
                self._check_fr_code(
                    robot.MoveJ(arm_joint_pos_command_deg, tool=0, user=0, vel=10.0)
                )
                self._start_servo_mode(side)

                fr_code, arm_joint_pos = robot.GetActualJointPosRadian()
                self._check_fr_code(fr_code)
                self.arm_joint_pos_actual[self._arm_slice(side)] = np.array(
                    arm_joint_pos
                )

                self.grippers[side].finger_move(
                    pose=self._gripper_pose(side, gripper_percent_closed)
                )

        self._motion_enabled = True

        print(
            f"[{self.__class__.__name__}] Finish moving the robots to the reset position."
        )

    def _set_action(self, action, duration=None, joint_vel_limit_scale=0.5, wait=False):
        start_time = time.time()

        if not self._motion_enabled:
            # Motion is gated until move_to_init_pose() runs (right before standby
            # teleop begins); until then, silently ignore commands instead of
            # actuating the arms/grippers. See _reset_robot()/move_to_init_pose().
            return

        # Overwrite duration or joint_pos for safety
        action, duration = self.overwrite_command_for_safety(
            action, duration, joint_vel_limit_scale
        )

        for side in (0, 1):
            arm_joint_pos_command_rad = action[
                self.body_config_list[side].arm_joint_idxes
            ]
            gripper_percent_closed = float(
                action[self.body_config_list[side].gripper_joint_idxes][0]
            )

            # Smooth the commanded arm joint position with an exponential moving
            # average to reduce jerk from the input device's per-frame
            # discretization and from jitter in the teleop loop's actual call rate.
            if self._filtered_arm_joint_pos_command[side] is None:
                self._filtered_arm_joint_pos_command[side] = (
                    arm_joint_pos_command_rad.copy()
                )
            else:
                alpha = self.command_smoothing_alpha
                self._filtered_arm_joint_pos_command[side] = (
                    alpha * arm_joint_pos_command_rad
                    + (1.0 - alpha) * self._filtered_arm_joint_pos_command[side]
                )
            arm_joint_pos_command_deg = np.rad2deg(
                self._filtered_arm_joint_pos_command[side]
            )

            joint_pos_list = [float(x) for x in arm_joint_pos_command_deg]
            exaxis_list = [float(x) for x in EXAXIS_POS]

            if self.dry_run:
                print(f"[{self.__class__.__name__}] [DRY RUN] side={side} ServoJ command:")
                print(f"  Joint positions [deg]: {joint_pos_list}")
                print(f"  Gripper: {gripper_percent_closed:.1f}% closed")
                self.arm_joint_pos_actual[self._arm_slice(side)] = np.deg2rad(
                    arm_joint_pos_command_deg
                )
            else:
                # cmdT tells the controller's interpolator how much time it has to
                # reach the target. The teleop loop's actual call rate is NOT a
                # stable self.dt: camera reads, cv2 drawing, and the synchronous
                # GetActualJointPosRadian/GetActualJointSpeedsDegree/gripper XML-RPC
                # round trips in _get_obs all add jittery, often-larger-than-dt
                # latency on top of it. Passing a fixed cmdT that's shorter than the
                # real interval makes the arm rush to each target and idle until the
                # next command arrives, producing jerky motion. Instead, measure the
                # actual elapsed time since the previous ServoJ call for this arm and
                # use that as cmdT so the interpolator's pacing matches reality.
                now = time.time()
                if self._last_servoj_time[side] is None:
                    cmd_t = self.dt
                else:
                    cmd_t = np.clip(now - self._last_servoj_time[side], 0.004, 0.5)
                self._last_servoj_time[side] = now

                # Send command to Fairino arm using raw XML-RPC proxy. The SDK
                # wrapper sends too many arguments; we call the raw proxy directly,
                # similar to fairino_keyboard_teleop/devices/fairino/interface.py for
                # ServoCart. The controller's XML-RPC endpoint expects exactly 7
                # parameters for ServoJ. All values must be Python floats (not
                # numpy.float64) for XML-RPC serialization.
                fr_code = self.robots[side].robot.ServoJ(
                    joint_pos_list,
                    exaxis_list,
                    0.0,  # acc
                    0.0,  # vel
                    float(cmd_t),  # cmdT
                    0.0,  # filterT
                    0.0,  # gain
                )
                self._check_fr_code(fr_code)

                self.grippers[side].finger_move(
                    pose=self._gripper_pose(side, gripper_percent_closed)
                )

        # Wait
        elapsed_duration = time.time() - start_time
        if wait and elapsed_duration < duration:
            time.sleep(duration - elapsed_duration)

    # Bound on GetActualJointPosRadian/GetActualJointSpeedsDegree calls in _get_obs.
    # These use the XML-RPC ServerProxy, which -- unlike the timeout applied briefly
    # during _connect_robot -- otherwise runs with no timeout at all (blocks
    # forever if the link degrades under load), with no console output while stuck.
    XMLRPC_READ_TIMEOUT_SEC = 2.0

    def _read_arm_state(self, side, robot):
        """Read one arm's actual joint position/velocity with a bounded timeout.
        Falls back to the last known-good joint position (and zero velocity) with a
        warning instead of hanging or crashing _get_obs() if the robot's XML-RPC
        link stalls."""
        try:
            socket.setdefaulttimeout(self.XMLRPC_READ_TIMEOUT_SEC)
            fr_code, arm_joint_pos = robot.GetActualJointPosRadian()
            self._check_fr_code(fr_code)
            arm_joint_pos = np.array(arm_joint_pos, dtype=np.float64)

            fr_code, arm_joint_vel_deg = robot.GetActualJointSpeedsDegree()
            self._check_fr_code(fr_code)
            arm_joint_vel = np.deg2rad(np.array(arm_joint_vel_deg, dtype=np.float64))
        except Exception as e:
            print(
                f"[{self.__class__.__name__}] Timed out or failed reading arm state "
                f"(side={side}): {e}. Using last known joint position."
            )
            arm_joint_pos = self.arm_joint_pos_actual[self._arm_slice(side)].astype(
                np.float64
            )
            arm_joint_vel = np.zeros(6, dtype=np.float64)
        finally:
            socket.setdefaulttimeout(None)

        self.arm_joint_pos_actual[self._arm_slice(side)] = arm_joint_pos.copy()
        return arm_joint_pos, arm_joint_vel

    def _get_obs(self):
        arm_joint_pos_list = []
        arm_joint_vel_list = []
        gripper_joint_pos_list = []
        gripper_joint_vel_list = []
        wrench_list = []

        for side in (0, 1):
            if self.dry_run:
                arm_joint_pos = self.arm_joint_pos_actual[
                    self._arm_slice(side)
                ].astype(np.float64)
                arm_joint_vel = np.zeros(6, dtype=np.float64)
                gripper_joint_pos = np.array([50.0], dtype=np.float64)  # Mock
                gripper_joint_vel = np.zeros(1)
            else:
                robot = self.robots[side]
                arm_joint_pos, arm_joint_vel = self._read_arm_state(side, robot)

                try:
                    gripper_pose_raw = np.array(
                        self.grippers[side].get_state(), dtype=np.float64
                    )
                    finger_raw = np.delete(
                        gripper_pose_raw, self.gripper_thumb_abduction_idx[side]
                    )
                    gripper_percent_closed = 100.0 * (1.0 - finger_raw.mean() / 255.0)
                    self._last_gripper_percent_closed[side] = gripper_percent_closed
                except Exception as e:
                    print(
                        f"[{self.__class__.__name__}] Failed to read gripper state "
                        f"(side={side}): {e}. Using last known gripper position."
                    )
                    gripper_percent_closed = self._last_gripper_percent_closed[side]
                gripper_joint_pos = np.array(
                    [gripper_percent_closed], dtype=np.float64
                )
                gripper_joint_vel = np.zeros(1)

            arm_joint_pos_list.append(arm_joint_pos)
            arm_joint_vel_list.append(arm_joint_vel)
            gripper_joint_pos_list.append(gripper_joint_pos)
            gripper_joint_vel_list.append(gripper_joint_vel)
            # Neither arm has a force/torque sensor
            wrench_list.append(np.zeros(6, dtype=np.float64))

        return {
            "joint_pos": np.concatenate(
                (
                    arm_joint_pos_list[0],
                    gripper_joint_pos_list[0],
                    arm_joint_pos_list[1],
                    gripper_joint_pos_list[1],
                ),
                dtype=np.float64,
            ),
            "joint_vel": np.concatenate(
                (
                    arm_joint_vel_list[0],
                    gripper_joint_vel_list[0],
                    arm_joint_vel_list[1],
                    gripper_joint_vel_list[1],
                ),
                dtype=np.float64,
            ),
            "wrench": np.concatenate(wrench_list, dtype=np.float64),
        }
