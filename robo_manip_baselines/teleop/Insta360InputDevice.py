import socket
import threading
import time

import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import read_insta360_message

from .InputDeviceBase import InputDeviceBase


class Insta360InputDevice(InputDeviceBase):
    """Insta360-camera-mounted teleoperation input device for the UMI rig.

    Unlike ViveInputDevice (which reads an HTC Vive Tracker's lighthouse-
    anchored absolute pose directly over USB/wireless via libsurvive), this
    device does not talk to the camera itself. It connects over a Unix domain
    socket to the insta360_bridge helper process (see
    envs/real/insta360_bridge/), a separate C++ process that pulls video +
    gyro from the Insta360 CameraSDK, feeds them into ORB-SLAM3
    (Monocular-Inertial), and streams the resulting 6-DoF camera pose back as
    "pose" messages (see common/utils/Insta360Protocol.py for the wire
    format). RealEnvBase.setup_insta360() separately connects to the SAME
    socket path to consume "frame" messages for the recorded camera images;
    the two connections are independent.

    ORB-SLAM3 continuously corrects its own drift (loop closure, IMU bias
    estimation) and reports an absolute pose each frame, similar in spirit to
    the Vive Tracker's lighthouse-anchored measurement -- so this device
    reuses ViveInputDevice's "enable-time anchor + closed-form rotation delta
    + per-frame accumulated translation delta" scheme (see
    ViveInputDevice.set_command_data()) rather than a raw IMU dead-reckoning
    delta, which would drift unboundedly. Two things Vive does not need are
    added on top of that scheme, because they are specific failure modes of a
    SLAM pose source rather than a lighthouse measurement:

      - tracking_state (from the bridge's "pose" message) gates teleop the
        same way TRACKING_TIMEOUT gates a lost Vive tracker: anything other
        than "OK" (i.e. "INIT" or "LOST") is treated as no pose available.
      - A large frame-to-frame pose jump (e.g. from ORB-SLAM3
        re-localizing after tracking loss) is detected and treated like a
        tracking dropout -- teleop is disabled and must re-settle and
        re-anchor -- rather than either being applied directly (which could
        send the arm flying) or silently absorbed (which would desync the
        commanded pose from the operator's actual hand motion).

    The camera has no physical trigger/button of its own, so unlike
    ViveInputDevice there is no button-driven gripper path -- gripper control
    requires gripper_key_bindings (keyboard fallback).
    """

    # If no new pose has arrived from the bridge within this many seconds,
    # the pose is treated as stale/lost. Same role as ViveInputDevice's
    # TRACKING_TIMEOUT.
    TRACKING_TIMEOUT = 0.5  # [s]

    # ORB-SLAM3's own initialization (map init, IMU bias/scale convergence)
    # can take a few seconds, similar in spirit to libsurvive's pose-solver
    # convergence transient -- see ViveInputDevice.MIN_ANCHOR_DELAY. Reusing
    # the same settle-and-anchor scheme and starting from the same magnitude
    # defaults; retune empirically once real tracking data is available.
    POSE_SETTLE_TIME = 0.5  # [s]
    POSE_SETTLE_POS_TOLERANCE = 0.02  # [m]
    POSE_SETTLE_ROT_TOLERANCE = np.deg2rad(5.0)  # [rad]
    MIN_ANCHOR_DELAY = 10.0  # [s]

    # A single-frame position/rotation change beyond this is not plausible
    # human hand motion at the ~50 Hz control loop rate, and is treated as a
    # SLAM re-localization jump rather than real motion -- see class
    # docstring. Tunable; no real-data tuning has been done yet.
    POSE_JUMP_POS_THRESHOLD = 0.15  # [m]
    POSE_JUMP_ROT_THRESHOLD = np.deg2rad(30.0)  # [rad]

    def __init__(
        self,
        arm_manager,
        device_params,
        pos_scale=1.0,
        gripper_scale=5.0,
        gripper_toggle=False,
        insta360_to_eef_frame_rotation=None,
        insta360_to_eef_translation=None,
        gripper_key_bindings=None,
    ):
        super().__init__()

        self.arm_manager = arm_manager
        self.name = device_params["name"]
        self.socket_path = device_params["socket_path"]
        self.pos_scale = pos_scale
        self.gripper_scale = gripper_scale
        self.gripper_toggle = gripper_toggle

        # Same role as ViveInputDevice.vive_to_eef_frame_rotation /
        # vive_to_eef_translation -- see that class for the full derivation.
        # Requires an equivalent calibration procedure for this camera's
        # actual mounting (calibrate_vive_rotation.py's approach can be
        # adapted; not yet done for Insta360).
        if insta360_to_eef_frame_rotation is None:
            self.insta360_to_eef_frame_rotation = np.eye(3)
        else:
            self.insta360_to_eef_frame_rotation = np.array(
                insta360_to_eef_frame_rotation, dtype=np.float64
            )
        assert self.insta360_to_eef_frame_rotation.shape == (3, 3)

        if insta360_to_eef_translation is None:
            self.insta360_to_eef_translation = np.zeros(3)
        else:
            self.insta360_to_eef_translation = np.array(
                insta360_to_eef_translation, dtype=np.float64
            )
        assert self.insta360_to_eef_translation.shape == (3,)

        self.gripper_key_bindings = gripper_key_bindings
        self.keyboard_state = None
        self.listener = None
        self.listener_thread = None

        self._connection = None
        self._reader_thread = None
        self._reader_stop_event = None
        self._pose_lock = threading.Lock()
        self._latest_pose_message = None

    def connect(self):
        self.enabled_teleop = False
        self.insta360_se3_at_enable = None
        self.eef_se3_at_enable = None
        self.has_announced_ready = False
        self._last_update_wall_time = None
        self._settle_start_se3 = None
        self._settle_start_wall_time = None
        self._first_tracked_wall_time = None
        # Raw (room-frame) camera pose as of the previous read() call, used
        # ONLY for the pose-jump check in read() -- updated every call,
        # regardless of enabled_teleop.
        self._prev_raw_translation = None
        self._prev_raw_rotation = None
        # TCP-space translation as of the previous set_command_data() call,
        # used ONLY for the incremental translation delta there -- same role
        # as ViveInputDevice._prev_vive_translation. Deliberately a separate
        # variable from _prev_raw_translation above: read() and
        # set_command_data() both run once per control step (read() first),
        # so if this reused _prev_raw_translation, read() would have already
        # overwritten it with the CURRENT frame's pose before
        # set_command_data() got a chance to diff against the previous one.
        self._prev_command_translation = None

        if self.connected:
            return

        self._connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._connection.connect(self.socket_path)

        self._reader_stop_event = threading.Event()
        self._reader_thread = threading.Thread(
            target=self._read_bridge_loop, daemon=True
        )
        self._reader_thread.start()

        if self.gripper_key_bindings is not None:
            from pynput import keyboard

            self.keyboard_state = {
                key: False for key in self.gripper_key_bindings.values()
            }
            self.listener = keyboard.Listener(
                on_press=self._on_key_press, on_release=self._on_key_release
            )
            self.listener_thread = threading.Thread(target=self._start_listener)
            self.listener_thread.daemon = True
            self.listener_thread.start()

        self.connected = True

    def _read_bridge_loop(self):
        """Runs in a background daemon thread (see connect()). Mirrors
        RealEnvBase._read_insta360_frame_loop, but keeps "pose" messages
        instead of "frame" ones -- read() below just consumes the latest one
        non-blockingly, so a stalled bridge/socket read cannot add latency to
        the control loop."""
        while not self._reader_stop_event.is_set():
            try:
                message = read_insta360_message(self._connection)
            except Exception as e:
                if not self._reader_stop_event.is_set():
                    print(
                        f"[{self.__class__.__name__}] Error reading Insta360 "
                        f"bridge '{self.name}': {e}"
                    )
                break

            if message["type"] != "pose":
                continue

            with self._pose_lock:
                self._latest_pose_message = message

    def _start_listener(self):
        self.listener.start()
        self.listener.join()

    def _on_key_press(self, key):
        try:
            k = key.char.lower()
            if k in self.keyboard_state:
                self.keyboard_state[k] = True
        except AttributeError:
            pass

    def _on_key_release(self, key):
        try:
            k = key.char.lower()
            if k in self.keyboard_state:
                self.keyboard_state[k] = False
        except AttributeError:
            pass

    def close(self):
        if not self.connected:
            return

        if self.listener:
            self.listener.stop()

        if self._reader_stop_event is not None:
            self._reader_stop_event.set()
        if self._connection is not None:
            try:
                self._connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=1.0)
        if self._connection is not None:
            self._connection.close()

        self.connected = False

    def read(self):
        if not self.connected:
            raise RuntimeError(f"[{self.__class__.__name__}] Device is not connected.")

        self._read_latest_pose()

        if self.state is None:
            self.enabled_teleop = False
            self.has_announced_ready = False
            self._settle_start_se3 = None
            self._settle_start_wall_time = None
            self._first_tracked_wall_time = None
            self._prev_raw_translation = None
            self._prev_raw_rotation = None
            self._prev_command_translation = None
            return

        current_se3 = self.state["se3"]

        # Detect a SLAM re-localization jump and treat it exactly like a
        # tracking dropout -- see class docstring. Checked before the
        # settle/anchor state machine below so a jump during an already-
        # enabled session forces a fresh settle-and-anchor instead of
        # applying (or silently absorbing) the discontinuity.
        if (
            self.enabled_teleop
            and self._prev_raw_translation is not None
            and self._prev_raw_rotation is not None
        ):
            pos_jump = np.linalg.norm(
                current_se3.translation - self._prev_raw_translation
            )
            rot_jump = np.linalg.norm(
                pin.log3(self._prev_raw_rotation.T @ current_se3.rotation)
            )
            if (
                pos_jump > self.POSE_JUMP_POS_THRESHOLD
                or rot_jump > self.POSE_JUMP_ROT_THRESHOLD
            ):
                print(
                    f"[{self.__class__.__name__}] Insta360 '{self.name}' pose "
                    f"jumped (pos {pos_jump:.3f} m, rot {np.rad2deg(rot_jump):.1f} "
                    "deg) -- likely a SLAM re-localization. Disabling teleop "
                    "until the pose re-settles."
                )
                self.enabled_teleop = False
                self.has_announced_ready = False
                self._settle_start_se3 = None
                self._settle_start_wall_time = None
                self._first_tracked_wall_time = None
                self._prev_raw_translation = None
                self._prev_raw_rotation = None
                self._prev_command_translation = None
                return

        self._prev_raw_translation = current_se3.translation.copy()
        self._prev_raw_rotation = current_se3.rotation.copy()

        # Same settle-and-anchor scheme as ViveInputDevice.read() -- see
        # POSE_SETTLE_TIME/MIN_ANCHOR_DELAY above for why.
        if not self.enabled_teleop:
            now = time.time()
            if self._first_tracked_wall_time is None:
                self._first_tracked_wall_time = now
            if self._settle_start_se3 is None:
                self._settle_start_se3 = current_se3.copy()
                self._settle_start_wall_time = now
            else:
                pos_diff = np.linalg.norm(
                    current_se3.translation - self._settle_start_se3.translation
                )
                rot_diff = np.linalg.norm(
                    pin.log3(self._settle_start_se3.rotation.T @ current_se3.rotation)
                )
                if (
                    pos_diff > self.POSE_SETTLE_POS_TOLERANCE
                    or rot_diff > self.POSE_SETTLE_ROT_TOLERANCE
                ):
                    self._settle_start_se3 = current_se3.copy()
                    self._settle_start_wall_time = now
                elif (
                    (now - self._settle_start_wall_time) >= self.POSE_SETTLE_TIME
                    and (now - self._first_tracked_wall_time)
                    >= self.MIN_ANCHOR_DELAY
                ):
                    self.enabled_teleop = True
                    self.insta360_se3_at_enable = current_se3.copy()
                    self.eef_se3_at_enable = self.arm_manager.current_se3.copy()
                    # Seed with the TCP position (matching what
                    # set_command_data()'s incremental delta diffs against),
                    # not the raw camera position -- see
                    # ViveInputDevice.read()'s identical seeding for why.
                    self._prev_command_translation = current_se3.translation - (
                        current_se3.rotation
                        @ self.insta360_to_eef_frame_rotation.T
                        @ self.insta360_to_eef_translation
                    )
                    print(
                        f"[{self.__class__.__name__}] Teleoperation enabled for "
                        f"Insta360 '{self.name}'."
                    )

    def _read_latest_pose(self):
        with self._pose_lock:
            message = self._latest_pose_message

        if message is None:
            self.state = None
            return

        now = time.time()
        if (now - message["t"]) > self.TRACKING_TIMEOUT:
            self.state = None
            return

        if message["tracking_state"] != "OK":
            self.state = None
            return

        pos = np.array(message["pos"], dtype=np.float64)
        # Wire format is (w, x, y, z), matching pin.Quaternion's constructor.
        rot = pin.Quaternion(*message["quat"]).toRotationMatrix()
        self.state = {"se3": pin.SE3(rot, pos)}

        if not self.has_announced_ready:
            print(f"[{self.__class__.__name__}] Insta360 '{self.name}' is ready.")
            self.has_announced_ready = True

    def is_ready(self):
        return self.state is not None

    def _gripper_limit(self, closed):
        """See ViveInputDevice._gripper_limit."""
        idxes = self.arm_manager.body_config.gripper_joint_idxes_for_limit
        space = self.arm_manager.env.action_space
        return (space.high if closed else space.low)[idxes].astype(np.float64)

    def set_command_data(self):
        if (not self.enabled_teleop) or (self.state is None):
            return

        # Rotation: closed-form delta from enable time. See
        # ViveInputDevice.set_command_data() for the full derivation --
        # identical math, with the Insta360-derived pose in place of the
        # tracker's.
        delta_insta360_rotation = (
            self.insta360_se3_at_enable.rotation.T @ self.state["se3"].rotation
        )
        adjusted_rotation_delta = (
            self.insta360_to_eef_frame_rotation
            @ delta_insta360_rotation
            @ self.insta360_to_eef_frame_rotation.T
        )
        target_rotation = self.eef_se3_at_enable.rotation @ adjusted_rotation_delta

        # Translation: accumulated incremental delta, same as Vive. Unlike
        # Vive there is no separate lever-arm correction here -- the camera
        # (not a tracker offset from the TCP) IS the pose source, so any
        # lever-arm-like offset between the camera and the TCP is exactly
        # what insta360_to_eef_translation is for, applied the same way
        # vive_to_eef_translation is.
        tcp_translation = self.state["se3"].translation - (
            self.state["se3"].rotation
            @ self.insta360_to_eef_frame_rotation.T
            @ self.insta360_to_eef_translation
        )
        # _prev_command_translation holds the TCP-space translation as of the
        # PREVIOUS set_command_data() call (seeded at enable time in read(),
        # updated below) -- kept separate from read()'s _prev_raw_translation
        # (which is already this frame's pose by the time this method runs;
        # see the field's docstring in connect()).
        raw_translation_delta_incremental = (
            tcp_translation - self._prev_command_translation
        )
        translation_delta_camera_local = (
            self.state["se3"].rotation.T @ raw_translation_delta_incremental
        )
        translation_delta_eef_local = (
            self.insta360_to_eef_frame_rotation @ translation_delta_camera_local
        )
        self._prev_command_translation = tcp_translation.copy()

        target_translation = self.arm_manager.target_se3.translation + self.pos_scale * (
            target_rotation @ translation_delta_eef_local
        )

        target_se3 = pin.SE3(target_rotation, target_translation)

        self.arm_manager.set_command_eef_pose(target_se3)

        # Set gripper command. Unlike ViveInputDevice, the Insta360 camera has
        # no physical trigger/button of its own, so this only supports the
        # keyboard-fallback path.
        if self.gripper_key_bindings is not None:
            gripper_joint_pos = self.arm_manager.get_command_gripper_joint_pos().copy()
            kb = self.gripper_key_bindings
            closing = self.keyboard_state[kb["gripper_close"]] and not (
                self.keyboard_state[kb["gripper_open"]]
            )
            opening = self.keyboard_state[kb["gripper_open"]] and not (
                self.keyboard_state[kb["gripper_close"]]
            )
            if self.gripper_toggle:
                if closing:
                    gripper_joint_pos = self._gripper_limit(closed=True)
                elif opening:
                    gripper_joint_pos = self._gripper_limit(closed=False)
            elif closing:
                gripper_joint_pos += self.gripper_scale
            elif opening:
                gripper_joint_pos -= self.gripper_scale
            self.arm_manager.set_command_gripper_joint_pos(gripper_joint_pos)
