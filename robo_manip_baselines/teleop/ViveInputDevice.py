import threading
import time

import numpy as np
import pinocchio as pin

from .InputDeviceBase import InputDeviceBase


class ViveInputDevice(InputDeviceBase):
    """HTC Vive Tracker for teleoperation input device.

    Uses pysurvive (libsurvive) rather than OpenVR/SteamVR, so no Steam
    installation is required -- trackers and lighthouses are read directly over
    USB/wireless. Run teleop/check_vive_devices.py first to confirm the
    lighthouses and trackers are detected and to look up each tracker's stable
    hardware serial number (e.g. "LHR-904A2704") for the
    device_params.serial_number config field. Note this is the TRACKER's
    serial, not the lighthouse's (lighthouse serials also start with "LH" but
    read "LHB-...").

    Vive Trackers have no trigger to gate teleop on, so teleop is enabled as
    soon as the tracker is visible. The gripper is toggled open/closed by the
    tracker's single physical button (any bit in libsurvive's button mask, since
    which exact SurviveButton id it reports can vary by firmware/pairing). If
    gripper_key_bindings is given instead, dedicated keyboard keys drive the
    gripper as a fallback (e.g. for testing without a working button).
    """

    # If no new pose has arrived from libsurvive within this many seconds, the
    # tracker is treated as lost (out of lighthouse view, powered off, etc.).
    TRACKING_TIMEOUT = 0.5  # [s]

    # A single libsurvive context talks to all USB-connected lighthouses/trackers,
    # so it must be created once and shared across all ViveInputDevice instances
    # (e.g. the two trackers of a dual-arm setup) rather than once per instance.
    _shared_context = None
    _shared_context_refcount = 0
    _shared_context_lock = threading.Lock()
    _pysurvive = None

    # SimpleContext.Objects() only returns the devices seen at the moment the
    # context was constructed -- trackers that finish USB/wireless negotiation
    # afterwards (which is normal; it takes a bit longer than the lighthouses)
    # never show up in it. NextUpdated() is the only way to discover them, so
    # every ViveInputDevice instance drains it into this shared-by-serial-number
    # registry instead of using Objects().
    _object_registry = {}
    _object_registry_lock = threading.Lock()

    def __init__(
        self,
        arm_manager,
        device_params,
        pos_scale=1.0,
        gripper_scale=5.0,
        vive_to_eef_frame_rotation=None,
        gripper_key_bindings=None,
    ):
        super().__init__()

        self.arm_manager = arm_manager
        self.name = device_params["name"]
        self.serial_number = device_params["serial_number"]
        self.pos_scale = pos_scale
        self.gripper_scale = gripper_scale
        if vive_to_eef_frame_rotation is None:
            self.vive_to_eef_frame_rotation = np.eye(3)
        else:
            self.vive_to_eef_frame_rotation = np.array(
                vive_to_eef_frame_rotation, dtype=np.float64
            )
        assert self.vive_to_eef_frame_rotation.shape == (3, 3)

        self.gripper_key_bindings = gripper_key_bindings
        self.keyboard_state = None
        self.listener = None
        self.listener_thread = None

        self._last_timecode = None
        self._last_update_wall_time = None

        # Gripper toggle state driven by the tracker's button. Not reset on
        # connect()/tracking loss so a brief dropout doesn't forget which way the
        # gripper was last commanded.
        self._gripper_button_pressed = False
        self._gripper_closing = False
        self._gripper_direction_initialized = False

    def connect(self):
        self.enabled_teleop = False
        self.vive_se3_at_enable = None
        self.eef_se3_at_enable = None
        self.has_announced_ready = False
        self._last_timecode = None
        self._last_update_wall_time = None

        if self.connected:
            return

        with ViveInputDevice._shared_context_lock:
            if ViveInputDevice._shared_context is None:
                import pysurvive

                ViveInputDevice._pysurvive = pysurvive
                ViveInputDevice._shared_context = pysurvive.SimpleContext([])
            ViveInputDevice._shared_context_refcount += 1
        self.pysurvive = ViveInputDevice._pysurvive
        self.ctx = ViveInputDevice._shared_context

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

        with ViveInputDevice._shared_context_lock:
            ViveInputDevice._shared_context_refcount -= 1
            if ViveInputDevice._shared_context_refcount <= 0:
                self.pysurvive.simple_close(ViveInputDevice._shared_context.ptr)
                ViveInputDevice._shared_context = None
                ViveInputDevice._shared_context_refcount = 0
                ViveInputDevice._object_registry = {}

        self.connected = False

    def read(self):
        if not self.connected:
            raise RuntimeError(f"[{self.__class__.__name__}] Device is not connected.")

        self._read_survive()

        if self.state is None:
            self.enabled_teleop = False
            self.has_announced_ready = False
            return

        # Vive Trackers have no trigger to gate teleop on, so enable teleop as soon
        # as the tracker is visible to the lighthouses and anchor on that pose.
        if not self.enabled_teleop:
            self.enabled_teleop = True
            self.vive_se3_at_enable = self.state["se3"].copy()
            self.eef_se3_at_enable = self.arm_manager.current_se3.copy()
            print(
                f"[{self.__class__.__name__}] Teleoperation enabled for Vive '{self.name}'."
            )

    def _read_survive(self):
        with ViveInputDevice._object_registry_lock:
            while True:
                survive_object = self.ctx.NextUpdated()
                if survive_object is None:
                    break
                serial_number = self.pysurvive.simple_serial_number(survive_object.ptr)
                if isinstance(serial_number, bytes):
                    serial_number = serial_number.decode()
                ViveInputDevice._object_registry[serial_number] = survive_object
            matched_object = ViveInputDevice._object_registry.get(self.serial_number)

        state = None
        if matched_object is not None:
            # Button state is independent of pose tracking quality, so poll it
            # even if the pose below turns out to be stale.
            if self.gripper_key_bindings is None:
                button_mask = self.pysurvive.simple_object_get_button_mask(
                    matched_object.ptr
                )
                button_pressed = button_mask != 0
                if button_pressed and not self._gripper_button_pressed:
                    self._gripper_closing = not self._gripper_closing
                    self._gripper_direction_initialized = True
                    print(
                        f"[{self.__class__.__name__}] Vive '{self.name}' gripper "
                        f"{'closing' if self._gripper_closing else 'opening'}."
                    )
                self._gripper_button_pressed = button_pressed

            pose, timecode = matched_object.Pose()
            if timecode > 0:
                now = time.time()
                if timecode != self._last_timecode:
                    self._last_timecode = timecode
                    self._last_update_wall_time = now
                if (now - self._last_update_wall_time) <= self.TRACKING_TIMEOUT:
                    pos = np.array(pose.Pos[:3], dtype=np.float64)
                    # libsurvive quaternions are stored as (w, x, y, z)
                    rot = pin.Quaternion(*pose.Rot[:4]).toRotationMatrix()
                    state = {"se3": pin.SE3(rot, pos)}

        self.state = state
        if (state is not None) and (not self.has_announced_ready):
            print(f"[{self.__class__.__name__}] Vive '{self.name}' is ready.")
            self.has_announced_ready = True

    def is_ready(self):
        return self.state is not None

    def set_command_data(self):
        if (not self.enabled_teleop) or (self.state is None):
            return

        # Set arm command
        delta_vive_se3 = self.vive_se3_at_enable.inverse() * self.state["se3"]
        adjusted_delta_vive_se3 = pin.SE3(
            self.vive_to_eef_frame_rotation
            @ delta_vive_se3.rotation
            @ self.vive_to_eef_frame_rotation.T,
            self.vive_to_eef_frame_rotation @ delta_vive_se3.translation,
        )
        scaled_delta_vive_se3 = pin.SE3(
            adjusted_delta_vive_se3.rotation,
            self.pos_scale * adjusted_delta_vive_se3.translation,
        )
        target_se3 = self.eef_se3_at_enable * scaled_delta_vive_se3

        self.arm_manager.set_command_eef_pose(target_se3)

        # Set gripper command: either the tracker's button (toggles open/closed)
        # or, if configured, dedicated keyboard keys as a fallback.
        if self.gripper_key_bindings is not None:
            gripper_joint_pos = self.arm_manager.get_command_gripper_joint_pos().copy()
            kb = self.gripper_key_bindings
            if self.keyboard_state[kb["gripper_close"]] and not self.keyboard_state[
                kb["gripper_open"]
            ]:
                gripper_joint_pos += self.gripper_scale
            elif self.keyboard_state[kb["gripper_open"]] and not self.keyboard_state[
                kb["gripper_close"]
            ]:
                gripper_joint_pos -= self.gripper_scale
            self.arm_manager.set_command_gripper_joint_pos(gripper_joint_pos)
        elif self._gripper_direction_initialized:
            gripper_joint_pos = self.arm_manager.get_command_gripper_joint_pos().copy()
            if self._gripper_closing:
                gripper_joint_pos += self.gripper_scale
            else:
                gripper_joint_pos -= self.gripper_scale
            self.arm_manager.set_command_gripper_joint_pos(gripper_joint_pos)
