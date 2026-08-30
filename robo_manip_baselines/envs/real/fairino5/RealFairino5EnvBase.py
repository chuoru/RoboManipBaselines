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


class RealFairino5EnvBase(RealEnvBase):
    # Official Fairino joint ranges (shared by FR3/FR5): J1/J5/J6 are +-175 deg;
    # J2/J4 are +85 to -265 deg; J3 is +-150 deg.
    action_space = Box(
        low=np.array(
            [
                np.deg2rad(-133),
                np.deg2rad(-136),
                np.deg2rad(-160),
                np.deg2rad(-130),
                np.deg2rad(-90),
                np.deg2rad(-101),
                0.0,
            ],
            dtype=np.float32,
        ),
        high=np.array(
            [
                np.deg2rad(0),
                np.deg2rad(0),
                np.deg2rad(-75),
                np.deg2rad(84),
                np.deg2rad(130),
                np.deg2rad(100),
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
        # "linker_hand": LinkerHand RS485 gripper with continuous 0-100% width
        # control (see _gripper_pose). "tool_do": a simple binary gripper
        # wired to the Fairino's own tool digital outputs (SetToolDO) -- no
        # continuous width. Default is "tool_do" since the currently-mounted
        # gripper is the IAI one.
        gripper_type="tool_do",
        # The two tool DO lines driving the binary gripper: one closes it, the
        # other opens it, each acting on its rising edge (see
        # _send_gripper_command). Measured on the real arm with
        # misc/TestGripperToolDO.py: DO0=1 closes, DO1=1 opens, and driving
        # either line to 0 does nothing.
        gripper_do_close_id=0,
        gripper_do_open_id=1,
        dry_run=False,
        # Connect to the arm and open the cameras as normal, but never
        # transmit a motion command: _motion_enabled stays False, so
        # _set_action() returns before reaching ServoJ/SetToolDO. Use it to
        # check that the observations a policy will consume (camera images,
        # measured EEF pose, gripper state) are correct BEFORE letting that
        # policy drive the hardware. Distinct from dry_run, which skips the
        # robot connection AND the cameras, so it cannot validate either.
        observe_only=False,
        command_smoothing_alpha=0.3,
        # ServoJ's own tuning knobs, exposed so they can be swept against real
        # hardware (see misc/TestFr5ServoJResponse.py) rather than being
        # buried as literals at the call site. Defaults reproduce what the
        # SDK's own examples pass. filterT/gain are the controller-side
        # smoothing and tracking-gain terms; the vendor examples leave both
        # at 0.0, but the real arm was measured tracking a slow commanded
        # trajectory at only 26-45% amplitude on the wrist joints, so these
        # are the first thing to test when tracking is that poor.
        servoj_filter_t=0.0,
        servoj_gain=0.0,
        # HARD per-command joint motion limit [deg], enforced on the exact
        # values sent to ServoJ (see _set_action). None disables it. This is
        # an absolute bound: it is not scaled by the loop period, so timing
        # jitter cannot widen it, and it sits after every other limiter so it
        # holds regardless of what they do. Use it to guarantee the arm can
        # never lurch far enough in one command to damage what it is holding.
        max_joint_pos_delta_deg=None,
        pointcloud_camera_ids=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Setup robot
        self.init_qpos = init_qpos
        self.dry_run = dry_run
        self.observe_only = observe_only
        self.servoj_filter_t = servoj_filter_t
        self.servoj_gain = servoj_gain
        self.max_joint_pos_delta_deg = max_joint_pos_delta_deg
        # Previously transmitted arm joint command [deg], the anchor for
        # max_joint_pos_delta_deg. None = nothing sent since the last reset or
        # out-of-stream move, so the next command anchors on the measurement.
        self._last_sent_arm_joint_pos_deg = None
        self.gripper_type = gripper_type
        self.gripper_do_close_id = gripper_do_close_id
        self.gripper_do_open_id = gripper_do_open_id
        # None until this process has commanded the gripper at least once.
        self._last_gripper_closing = None
        self._last_servoj_time = None
        self._last_gripper_percent_closed = 50.0
        # Gates whether _set_action actually transmits ServoJ/gripper commands to
        # hardware. Starts False so the arm stays still (even though env.step() is
        # called continuously by the teleop loop from the very first frame) until
        # move_to_init_pose() explicitly enables it, right before StandbyTeleopPhase.
        self._motion_enabled = False
        # Exponential-moving-average filter applied to the commanded arm joint
        # position before it is sent to ServoJ, to smooth out the per-frame
        # discretization of the input device and jitter in the teleop loop's actual
        # call rate. Lower alpha = smoother but more lag; 1.0 = no filtering.
        self.command_smoothing_alpha = command_smoothing_alpha
        self._filtered_arm_joint_pos_command = None
        # TODO: Verify against the official FR5 max joint speed before running on real hardware.
        self.joint_vel_limit = np.deg2rad(30) # [rad/s]
        self.body_config_list = [
            ArmConfig(
                arm_urdf_path=path.join(
                    path.dirname(__file__),
                    "../../assets/common/robots/fairino5_v6/fairino5_v6.urdf",
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
        self.robot_ip = robot_ip
        if self.dry_run:
            print(
                f"[{self.__class__.__name__}] DRY RUN MODE: Skipping robot connection. "
                "Commands will be printed instead of executed."
            )
            self.robot = None
            self.arm_joint_pos_actual = np.array(self.init_qpos[0:6])
            self.gripper = None
        else:
            print(f"[{self.__class__.__name__}] Start connecting the Fairino arm.")
            self.robot = self._connect_robot(self.robot_ip)
            self._enable_robot()
            self._start_servo_mode()
            fr_code, arm_joint_pos = self.robot.GetActualJointPosRadian()
            self._check_fr_code(fr_code)
            self.arm_joint_pos_actual = np.array(arm_joint_pos)
            print(f"[{self.__class__.__name__}] Finish connecting the Fairino arm.")

            if self.gripper_type == "linker_hand":
                # Connect to LinkerHand gripper
                print(f"[{self.__class__.__name__}] Start connecting the LinkerHand gripper.")
                try:
                    from LinkerHand.linker_hand_api import LinkerHandApi
                except ImportError as e:
                    raise RuntimeError(
                        f"[{self.__class__.__name__}] Failed to import LinkerHand. "
                        f"Please ensure the LinkerHand package is installed. Error: {e}"
                    )
                self.gripper = LinkerHandApi(
                    hand_type=gripper_hand_type,
                    hand_joint="O6",
                    modbus=gripper_modbus_port,
                )
                gripper_finger_order = self.gripper.get_finger_order()
                self.gripper_num_joints = len(gripper_finger_order)
                self.gripper_thumb_abduction_idx = gripper_finger_order.index(
                    "thumb_cmc_yaw"
                )
                print(f"[{self.__class__.__name__}] Finish connecting the LinkerHand gripper.")
            elif self.gripper_type == "tool_do":
                # The IAI gripper is wired to the Fairino's own tool DO -- no
                # separate connection is needed, self.robot.SetToolDO/GetToolDO
                # (already connected above) is used directly in _set_action/_get_obs.
                self.gripper = None
                print(
                    f"[{self.__class__.__name__}] Using IAI gripper via tool "
                    f"DO{self.gripper_do_close_id} (close) / "
                    f"DO{self.gripper_do_open_id} (open) "
                    "(no separate gripper connection)."
                )
            else:
                raise ValueError(
                    f"[{self.__class__.__name__}] Invalid gripper_type: {self.gripper_type}"
                )

        # Connect to RealSense, Orbbec (femtobolt), and GelSight (only in
        # non-dry-run mode)
        if not self.dry_run:
            self.setup_realsense(camera_ids)
            self.setup_femtobolt(pointcloud_camera_ids)
            self.setup_gelsight(gelsight_ids)
        else:
            # In dry-run mode, initialize empty sensor dicts
            self.cameras = {}
            self.pointcloud_cameras = {}
            self.rgb_tactiles = {}
            self.intensity_tactiles = {}

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

    def _enable_robot(self):
        self._check_fr_code(self.robot.RobotEnable(1))
        self._check_fr_code(self.robot.Mode(0))
        self._check_fr_code(self.robot.ResetAllError())

    def _start_servo_mode(self):
        self._check_fr_code(self.robot.ServoMoveStart())
        # Wait for servo to be ready
        time.sleep(1.5)
        # Reset the ServoJ timing reference: the next ServoJ call has no real
        # "previous call" to measure an interval from.
        self._last_servoj_time = None

    def close(self):
        if self.robot is not None:
            try:
                self.robot.ServoMoveEnd()
            except Exception as e:
                print(f"[{self.__class__.__name__}] Error during ServoMoveEnd: {e}")
            try:
                self.robot.CloseRPC()
            except Exception as e:
                print(f"[{self.__class__.__name__}] Error during CloseRPC: {e}")

        super().close()

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

    def _send_gripper_command(self, gripper_percent_closed):
        if self.gripper_type == "linker_hand":
            self.gripper.finger_move(pose=self._gripper_pose(gripper_percent_closed))
        else:
            # tool_do: no continuous width, just threshold to binary
            # open/close.
            #
            # This IAI gripper is a DOUBLE-ACTING pair: one tool DO line
            # drives "close", a DIFFERENT one drives "open", and each acts on
            # its rising edge -- driving a line to 0 does nothing at all.
            # Measured with misc/TestGripperToolDO.py on the real arm:
            #   DO0=1 -> closes      DO0=0 -> nothing
            #   DO1=1 -> opens       DO1=0 -> nothing
            # An earlier single-line reading of this ("DO1=1 opens, DO1=0
            # closes") was wrong about the closing half, and the resulting
            # one-line scheme could only ever open the gripper: closing sent
            # DO1=0, which does nothing. That is exactly what a UMI replay
            # showed -- a demo whose commands crossed the close threshold for
            # 12.8 continuous seconds never actuated the gripper.
            closing = gripper_percent_closed >= 50.0
            if closing == self._last_gripper_closing:
                # SetToolDO is a blocking XML-RPC call; only send on a state
                # change, not every control-loop tick (unlike LinkerHand's
                # finger_move, which is cheap enough to call every frame).
                return

            active_do_id = (
                self.gripper_do_close_id if closing else self.gripper_do_open_id
            )
            inactive_do_id = (
                self.gripper_do_open_id if closing else self.gripper_do_close_id
            )
            # Release the opposite line before asserting this one, so the two
            # halves are never driven at once.
            self._check_fr_code(self.robot.SetToolDO(inactive_do_id, 0))
            self._check_fr_code(self.robot.SetToolDO(active_do_id, 1))
            self._last_gripper_closing = closing

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
                motion_manager.body_manager_list[0],
                **{**default_kwargs, **overwrite_kwargs},
            )
        ]

    def get_input_device_kwargs(self, input_device_name):
        if input_device_name == "spacemouse":
            return {"pos_scale": 1.5e-2, "rpy_scale": 1e-2, "gripper_scale": 10.0}
        elif input_device_name == "keyboard":
            # KeyboardInputDevice's defaults (pos_scale=1e-2, rpy_scale=5e-2, with yaw
            # at 2x rpy_scale) work out to ~143 deg/s roll/pitch and ~286 deg/s yaw
            # while a key is held at this env's 50 Hz control rate, which is too
            # aggressive for real hardware and reads as jerky/violent motion.
            return {"pos_scale": 1.5e-2, "rpy_scale": 1e-2, "gripper_scale": 10.0}
        else:
            return super().get_input_device_kwargs(input_device_name)

    def _reset_robot(self):
        # env.step() is called continuously by the teleop loop starting from the very
        # first frame (InitialTeleopPhase), before the operator has connected input
        # devices or is necessarily at the controls. So the physical move to the reset
        # pose is NOT performed here: this only re-syncs internal state so the safety
        # clamp in overwrite_command_for_safety compares against where the arm truly
        # is. The actual move happens in move_to_init_pose(), which
        # OperationRealFairino5Demo's MoveToInitPhase calls as the last pre-motion
        # phase, right before StandbyTeleopPhase begins.
        self._motion_enabled = False
        self._filtered_arm_joint_pos_command = None
        self._last_sent_arm_joint_pos_deg = None

        if self.dry_run:
            self.arm_joint_pos_actual = np.array(
                self.init_qpos[self.body_config_list[0].arm_joint_idxes]
            )
        else:
            fr_code, arm_joint_pos = self.robot.GetActualJointPosRadian()
            self._check_fr_code(fr_code)
            self.arm_joint_pos_actual = np.array(arm_joint_pos)

    def move_to_init_pose(self):
        """Physically move the arm/gripper to the reset pose and enable streaming
        ServoJ commands from the teleop loop. Intended to be called once, right
        before standby teleop begins (see MoveToInitPhase in
        OperationRealFairino5Demo.py)."""
        if self.observe_only:
            # Leave _motion_enabled False, so _set_action() returns before
            # transmitting anything (see the gate there). Everything that only
            # READS stays live: cameras, GetActualJointPosRadian, gripper DO
            # state. That is the difference from dry_run, which never connects
            # to the arm at all and never opens the cameras -- useless for
            # checking that the observations a policy will see are correct.
            print(
                f"[{self.__class__.__name__}] OBSERVE-ONLY MODE: the arm will "
                "NOT move and the gripper will NOT actuate. Commands are "
                "computed and discarded; cameras and state readback are live."
            )
            return

        print(
            f"[{self.__class__.__name__}] Start moving the robot to the reset position."
        )

        self._filtered_arm_joint_pos_command = None
        # The arm is about to be moved by MoveJ, outside the ServoJ command
        # stream, so any previous anchor is stale -- see _reset_robot().
        self._prev_arm_joint_pos_command = None
        # The blocking MoveJ below takes seconds; without clearing this the
        # first post-move step() would measure that whole move as its control
        # period and authorize a correspondingly large jump.
        self._last_step_time = None
        self._last_servoj_time = None
        # MoveJ below repositions the arm outside the ServoJ stream, so the
        # previous sent-command anchor is stale.
        self._last_sent_arm_joint_pos_deg = None

        # Open the gripper BEFORE moving the arm, not after: if the gripper was
        # left closed around something from a previous session, moving the arm
        # first would drag/crush whatever it's holding. Independent of
        # init_qpos's gripper value (which is also sent again after the move,
        # below) so this happens even if init_qpos's gripper component is ever
        # non-zero for some env variant.
        print(f"[{self.__class__.__name__}] Opening gripper before moving the arm.")
        if self.dry_run:
            print(f"[{self.__class__.__name__}] [DRY RUN] Would open gripper.")
        else:
            self._send_gripper_command(0.0)

        if self.dry_run:
            print(
                f"[{self.__class__.__name__}] [DRY RUN] Would move to reset pose: "
                f"{list(np.rad2deg(self.init_qpos[self.body_config_list[0].arm_joint_idxes]))} deg"
            )
            self.arm_joint_pos_actual = np.array(
                self.init_qpos[self.body_config_list[0].arm_joint_idxes]
            )
        else:
            # ServoJ is a high-frequency streaming command: it expects to be called
            # repeatedly with small incremental targets, and interprets a single
            # large joint delta as needing to be reached almost instantly, which
            # trips the controller's axis speed limit ("The command speed in the
            # joint space of axis 1 exceeds the limit"). For the (potentially large)
            # one-shot move to the reset pose, use the blocking point-to-point MoveJ
            # command instead, then re-enter servo mode for the streaming teleop
            # control loop.
            arm_joint_pos_command_deg = [
                float(x)
                for x in np.rad2deg(
                    self.init_qpos[self.body_config_list[0].arm_joint_idxes]
                )
            ]
            self._check_fr_code(self.robot.ServoMoveEnd())
            self._check_fr_code(
                self.robot.MoveJ(arm_joint_pos_command_deg, tool=0, user=0, vel=10.0)
            )
            self._start_servo_mode()

            fr_code, arm_joint_pos = self.robot.GetActualJointPosRadian()
            self._check_fr_code(fr_code)
            self.arm_joint_pos_actual = np.array(arm_joint_pos)

            gripper_percent_closed = float(
                self.init_qpos[self.body_config_list[0].gripper_joint_idxes][0]
            )
            self._send_gripper_command(gripper_percent_closed)

        self._motion_enabled = True

        print(
            f"[{self.__class__.__name__}] Finish moving the robot to the reset position."
        )

    def _set_action(self, action, duration=None, joint_vel_limit_scale=0.5, wait=False):
        start_time = time.time()

        if not self._motion_enabled:
            # Motion is gated until move_to_init_pose() runs (right before standby
            # teleop begins); until then, silently ignore commands instead of
            # actuating the arm/gripper. See _reset_robot()/move_to_init_pose().
            return

        raw_command_deg_for_log = np.rad2deg(
            action[self.body_config_list[0].arm_joint_idxes]
        ).copy()
        measured_deg_for_log = np.rad2deg(self.arm_joint_pos_actual).copy()

        # Overwrite duration or joint_pos for safety
        action, duration = self.overwrite_command_for_safety(
            action, duration, joint_vel_limit_scale
        )

        # Extract joint positions
        arm_joint_pos_command_rad = action[self.body_config_list[0].arm_joint_idxes]
        gripper_percent_closed = float(
            action[self.body_config_list[0].gripper_joint_idxes][0]
        )
        safety_command_deg_for_log = np.rad2deg(arm_joint_pos_command_rad).copy()

        # Smooth the commanded arm joint position with an exponential moving average
        # to reduce jerk from the input device's per-frame discretization and from
        # jitter in the teleop loop's actual call rate (see command_smoothing_alpha).
        if self._filtered_arm_joint_pos_command is None:
            self._filtered_arm_joint_pos_command = arm_joint_pos_command_rad.copy()
        else:
            alpha = self.command_smoothing_alpha
            self._filtered_arm_joint_pos_command = (
                alpha * arm_joint_pos_command_rad
                + (1.0 - alpha) * self._filtered_arm_joint_pos_command
            )
        arm_joint_pos_command_deg = np.rad2deg(self._filtered_arm_joint_pos_command)

        # LAST LINE OF DEFENCE: hard cap on how far any joint may move between
        # two consecutive ServoJ commands. Applied here, on the exact values
        # about to be transmitted, deliberately AFTER the velocity clamp and
        # the EMA -- so it holds no matter what those did, and no matter how
        # the loop period jittered. Unlike overwrite_command_for_safety's
        # clamp it is not scaled by any measured duration, so a slow tick can
        # never widen it.
        #
        # This exists because the policy's raw demand is not uniform: measured
        # at time_scale 2.0 with the clamp disabled, 95% of ticks asked for
        # <0.2 deg, but rare action-chunk boundaries asked for up to 13.7 deg
        # in a single tick (~340 deg/s). Those spikes are what slam the arm.
        # Capping spreads such a jump over consecutive commands instead of
        # letting it through in one, while leaving normal motion untouched.
        if self.max_joint_pos_delta_deg is not None:
            if self._last_sent_arm_joint_pos_deg is None:
                # Nothing sent yet this episode: anchor on where the arm
                # actually is, so the first command cannot jump either.
                self._last_sent_arm_joint_pos_deg = np.rad2deg(
                    self.arm_joint_pos_actual
                ).copy()
            arm_joint_pos_command_deg = self._last_sent_arm_joint_pos_deg + np.clip(
                arm_joint_pos_command_deg - self._last_sent_arm_joint_pos_deg,
                -self.max_joint_pos_delta_deg,
                self.max_joint_pos_delta_deg,
            )
            # Keep the EMA state consistent with what was actually sent,
            # otherwise it keeps integrating toward the uncapped command and
            # the cap silently turns into a permanent offset.
            self._filtered_arm_joint_pos_command = np.deg2rad(
                arm_joint_pos_command_deg
            )
        self._last_sent_arm_joint_pos_deg = arm_joint_pos_command_deg.copy()

        self.log_command_for_safety_debug(
            duration_arg=duration,
            wait=wait,
            measured_deg=measured_deg_for_log,
            raw_command_deg=raw_command_deg_for_log,
            safety_command_deg=safety_command_deg_for_log,
            sent_deg=arm_joint_pos_command_deg,
            gripper_percent_closed=gripper_percent_closed,
        )

        # Convert numpy arrays to Python floats for XML-RPC compatibility
        joint_pos_list = [float(x) for x in arm_joint_pos_command_deg]
        exaxis_list = [float(x) for x in self.EXAXIS_POS]

        if self.dry_run:
            # Dry-run mode: print commands instead of executing
            print(f"[{self.__class__.__name__}] [DRY RUN] ServoJ command:")
            print(f"  Joint positions [deg]: {joint_pos_list}")
            print(f"  External axes: {exaxis_list}")
            print(f"  Gripper: {gripper_percent_closed:.1f}% closed")
            # Update arm position in dry-run mode for FK
            self.arm_joint_pos_actual = np.deg2rad(arm_joint_pos_command_deg)
        else:
            # cmdT tells the controller's interpolator how much time it has to reach
            # the target. The teleop loop's actual call rate is NOT a stable self.dt:
            # camera reads, cv2 drawing, and the synchronous GetActualJointPosRadian/
            # GetActualJointSpeedsDegree/gripper XML-RPC round trips in _get_obs all add
            # jittery, often-larger-than-dt latency on top of it. Passing a fixed cmdT
            # that's shorter than the real interval makes the arm rush to each target
            # and idle until the next command arrives, producing jerky motion. Instead,
            # measure the actual elapsed time since the previous ServoJ call and use
            # that as cmdT so the interpolator's pacing matches reality.
            now = time.time()
            if self._last_servoj_time is None:
                cmd_t = self.dt
            else:
                cmd_t = np.clip(now - self._last_servoj_time, 0.004, 0.5)
            self._last_servoj_time = now

            # Send command to Fairino arm using raw XML-RPC proxy
            # The SDK wrapper sends too many arguments. We call the raw proxy directly,
            # similar to fairino_keyboard_teleop/devices/fairino/interface.py for ServoCart.
            # The controller's XML-RPC endpoint expects exactly 7 parameters for ServoJ.
            # All values must be Python floats (not numpy.float64) for XML-RPC serialization.
            # EXACTLY 7 arguments -- do NOT add the SDK wrapper's 8th `id`
            # parameter. Robot.ServoJ passes 8 (…, gain, id), but this
            # controller's XML-RPC endpoint rejects that outright:
            #   Fault -502: "Format string requests exactly 7 items from
            #   array, but array has 8 items."
            # which is why this calls the raw proxy directly instead of going
            # through the SDK wrapper at all. Verified against the real arm.
            fr_code = self.robot.robot.ServoJ(
                joint_pos_list,
                exaxis_list,
                0.0,                            # acc
                0.0,                            # vel
                float(cmd_t),                   # cmdT
                float(self.servoj_filter_t),    # filterT
                float(self.servoj_gain),        # gain
            )
            self._check_fr_code(fr_code)

            self._send_gripper_command(gripper_percent_closed)

        # Wait
        elapsed_duration = time.time() - start_time
        if wait and elapsed_duration < duration:
            time.sleep(duration - elapsed_duration)

    # Bound on GetActualJointPosRadian/GetActualJointSpeedsDegree calls in _get_obs.
    # These use the XML-RPC ServerProxy, which -- unlike the timeout applied briefly
    # during _connect_robot -- otherwise runs with no timeout at all (blocks
    # forever if the link degrades under load), with no console output while stuck.
    XMLRPC_READ_TIMEOUT_SEC = 2.0

    def _get_obs(self):
        if self.dry_run:
            # In dry-run mode, return mock observation based on last commanded position
            arm_joint_pos = self.arm_joint_pos_actual.astype(np.float64)
            arm_joint_vel = np.zeros(6, dtype=np.float64)
            gripper_joint_pos = np.array([50.0], dtype=np.float64)  # Mock: 50% closed
            gripper_joint_vel = np.zeros(1)
        else:
            # Get state from Fairino arm, falling back to the last known joint
            # position (and zero velocity) with a warning instead of hanging or
            # crashing _get_obs() if the robot's XML-RPC link stalls.
            try:
                socket.setdefaulttimeout(self.XMLRPC_READ_TIMEOUT_SEC)
                fr_code, arm_joint_pos = self.robot.GetActualJointPosRadian()
                self._check_fr_code(fr_code)
                arm_joint_pos = np.array(arm_joint_pos, dtype=np.float64)

                fr_code, arm_joint_vel_deg = self.robot.GetActualJointSpeedsDegree()
                self._check_fr_code(fr_code)
                arm_joint_vel = np.deg2rad(
                    np.array(arm_joint_vel_deg, dtype=np.float64)
                )
            except Exception as e:
                print(
                    f"[{self.__class__.__name__}] Timed out or failed reading arm "
                    f"state: {e}. Using last known joint position."
                )
                arm_joint_pos = self.arm_joint_pos_actual.astype(np.float64)
                arm_joint_vel = np.zeros(6, dtype=np.float64)
            finally:
                socket.setdefaulttimeout(None)
            self.arm_joint_pos_actual = arm_joint_pos.copy()

            if self.gripper_type == "linker_hand":
                # Get state from LinkerHand gripper, falling back to the last known
                # gripper position if the Modbus read fails.
                try:
                    gripper_pose_raw = np.array(
                        self.gripper.get_state(), dtype=np.float64
                    )
                    finger_raw = np.delete(
                        gripper_pose_raw, self.gripper_thumb_abduction_idx
                    )
                    gripper_percent_closed = 100.0 * (1.0 - finger_raw.mean() / 255.0)
                    self._last_gripper_percent_closed = gripper_percent_closed
                except Exception as e:
                    print(
                        f"[{self.__class__.__name__}] Failed to read gripper state: {e}. "
                        "Using last known gripper position."
                    )
                    gripper_percent_closed = self._last_gripper_percent_closed
            else:
                # tool_do: no width sensor, only the binary DO state. Trust
                # _last_gripper_closing (updated whenever we command it, see
                # _send_gripper_command) instead of polling GetToolDO() here --
                # that was an extra synchronous XML-RPC round trip on every
                # single control-loop tick (in addition to the
                # GetActualJointPosRadian/GetActualJointSpeedsDegree calls
                # above), and ServoJ's cmdT pacing (see _set_action) is
                # sensitive to exactly this kind of added loop-time jitter --
                # traced to visibly jerky teleop motion. This does mean an
                # out-of-band DO toggle (e.g. from a teach pendant) won't be
                # reflected here, only DO changes this process itself sent.
                if self._last_gripper_closing is None:
                    # Nothing has been commanded yet this process. That is the
                    # normal startup state -- env.step() runs from the very
                    # first frame, while the first gripper command is not sent
                    # until move_to_init_pose()/GraspPhase -- not a failure, so
                    # do not report it. (It used to raise and print every tick,
                    # flooding the console before the operator pressed 'n'.)
                    gripper_percent_closed = self._last_gripper_percent_closed
                else:
                    gripper_percent_closed = (
                        100.0 if self._last_gripper_closing else 0.0
                    )
                    self._last_gripper_percent_closed = gripper_percent_closed
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
