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

    # A tracker's pose is noisy for a brief moment right after it's first seen
    # (libsurvive's MPFIT pose solver hasn't converged yet -- this is worse right
    # after the lighthouses/tracker have been physically moved). Anchoring
    # enabled_teleop on that first noisy pose, then having it snap to the true
    # pose a few frames later, looks exactly like a startup jump even though the
    # tracker never physically moved -- so we require the pose to stay settled
    # (near-stationary) for this long before anchoring.
    POSE_SETTLE_TIME = 0.5  # [s]
    POSE_SETTLE_POS_TOLERANCE = 0.02  # [m]
    POSE_SETTLE_ROT_TOLERANCE = np.deg2rad(5.0)  # [rad]

    # Minimum time since the tracker was FIRST seen (independent of the
    # settle-window check above) before anchoring is allowed at all.
    # misc/MeasureViveDrift.py measured a stationary tracker's raw pose
    # directly: position climbed ~2cm and orientation ~7deg over the first
    # ~5-10s after acquisition (visible in libsurvive's own log as
    # successive "Global solve with N scenes" refinements -- i.e. the
    # multi-lighthouse pose solver is still actively converging, not just
    # settling from tracker motion), then stayed flat within ~1-2mm/~1deg
    # noise for the rest of a 60s hold. POSE_SETTLE_POS_TOLERANCE/
    # POSE_SETTLE_ROT_TOLERANCE (2cm/5deg) are comparable in size to that
    # ENTIRE convergence transient, so the settle-window check above can
    # (and in practice does) lock in an anchor pose mid-convergence, any
    # time the instantaneous rate of change dips below tolerance for
    # POSE_SETTLE_TIME even though the solver hasn't finished refining --
    # confirmed to be the root cause of a UMI recording that appeared to
    # "drift" ~5cm/~5.7deg over its own first 10.5s despite the tracker
    # never having been intentionally moved. This is a hard floor
    # independent of that check, with margin above the ~5-10s observed.
    MIN_ANCHOR_DELAY = 10.0  # [s]

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
        vive_world_to_base_frame_rotation=None,
        gripper_key_bindings=None,
    ):
        super().__init__()

        self.arm_manager = arm_manager
        self.name = device_params["name"]
        self.serial_number = device_params["serial_number"]
        self.pos_scale = pos_scale
        self.gripper_scale = gripper_scale
        # Rotates a tracker-local *rotation* delta into the EEF's own local (TCP)
        # frame -- see set_command_data(). Depends on how the tracker happens to
        # be held each session; calibrate with calibrate_vive_axes.py.
        if vive_to_eef_frame_rotation is None:
            self.vive_to_eef_frame_rotation = np.eye(3)
        else:
            self.vive_to_eef_frame_rotation = np.array(
                vive_to_eef_frame_rotation, dtype=np.float64
            )
        assert self.vive_to_eef_frame_rotation.shape == (3, 3)
        # No longer used by set_command_data(): translation is now computed
        # TCP-locally (via vive_to_eef_frame_rotation, same as rotation) instead of
        # as a room-to-base-frame absolute delta, so a room<->base calibration is no
        # longer needed for translation. Kept as an accepted (but unused) constructor
        # kwarg only so existing configs (Vive.yaml/ViveDual.yaml/ViveUMI.yaml) that
        # still set this key don't fail to load; harmless to leave unset going
        # forward.
        if vive_world_to_base_frame_rotation is None:
            self.vive_world_to_base_frame_rotation = np.eye(3)
        else:
            self.vive_world_to_base_frame_rotation = np.array(
                vive_world_to_base_frame_rotation, dtype=np.float64
            )
        assert self.vive_world_to_base_frame_rotation.shape == (3, 3)

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
        self._settle_start_se3 = None
        self._settle_start_wall_time = None
        self._first_tracked_wall_time = None
        # Room-frame tracker position as of the *previous* frame, used to compute
        # this frame's incremental translation delta in set_command_data(). Reset
        # here and re-anchored (alongside vive_se3_at_enable) whenever teleop
        # (re-)enables, so a tracking dropout can't leave a stale reference that
        # would otherwise show up as one large spurious jump on the next frame.
        self._prev_vive_translation = None

        if self.connected:
            return

        with ViveInputDevice._shared_context_lock:
            if ViveInputDevice._shared_context is None:
                import pysurvive

                ViveInputDevice._pysurvive = pysurvive
                # Force the BaryCentricSVD poser instead of libsurvive's
                # default (MPFIT) for per-object pose fitting, and disable
                # the separate lighthouse-position calibration solve (which
                # *always* runs through GlobalSceneSolver's own internal
                # MPFIT-based optimizer, regardless of "-p" -- confirmed by
                # "MPFIT success"/"Global solve" log lines still appearing
                # even with the poser above forced to BaryCentricSVD).
                # Windows Event Log repeatedly showed a native stack overflow
                # (0xc00000fd) inside libsurvive.dll/poser_mpfit.dll at the
                # exact same offset during long teleop recording sessions;
                # an isolated 5-minute continuous-tracking stress test only
                # stopped reproducing it once calibration was disabled here.
                # This relies on a valid calibration already being cached in
                # ~/.config/libsurvive/config.json (run once with calibration
                # enabled -- or via calibrate_vive_rotation.py -- to produce
                # it) since --disable-calibrate skips solving it fresh.
                ViveInputDevice._shared_context = pysurvive.SimpleContext(
                    ["-p", "PoserBaryCentricSVD", "--disable-calibrate", "1"]
                )
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
            self._settle_start_se3 = None
            self._settle_start_wall_time = None
            self._first_tracked_wall_time = None
            self._prev_vive_translation = None
            return

        # Vive Trackers have no trigger to gate teleop on, so enable teleop as soon
        # as the tracker's pose has settled (see POSE_SETTLE_TIME) and anchor on
        # that settled pose.
        if not self.enabled_teleop:
            current_se3 = self.state["se3"]
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
                    # Pose moved (or is still converging) -- restart the settle
                    # window from this new pose.
                    self._settle_start_se3 = current_se3.copy()
                    self._settle_start_wall_time = now
                elif (
                    (now - self._settle_start_wall_time) >= self.POSE_SETTLE_TIME
                    and (now - self._first_tracked_wall_time)
                    >= self.MIN_ANCHOR_DELAY
                ):
                    self.enabled_teleop = True
                    self.vive_se3_at_enable = current_se3.copy()
                    self.eef_se3_at_enable = self.arm_manager.current_se3.copy()
                    self._prev_vive_translation = current_se3.translation.copy()
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

        # Set arm command.
        #
        # Rotation tracks the tool's own local frame, and is path-independent (SO(3)
        # composition only depends on start/end orientation, not the path taken in
        # between) so it's computed as a single closed-form delta from enable time:
        # a tracker rotation delta (relative to how it was held at enable time) is
        # applied as an EEF-local rotation delta relative to the tool's enable-time
        # orientation.
        delta_vive_rotation = (
            self.vive_se3_at_enable.rotation.T @ self.state["se3"].rotation
        )
        adjusted_rotation_delta = (
            self.vive_to_eef_frame_rotation
            @ delta_vive_rotation
            @ self.vive_to_eef_frame_rotation.T
        )
        target_rotation = self.eef_se3_at_enable.rotation @ adjusted_rotation_delta

        # Translation follows the tool's own local frame too (so "push the tracker
        # forward" always means "push the TCP forward along its own current Z", even
        # while simultaneously rotating -- holonomic-style combined motion). Unlike
        # rotation, this is genuinely path-dependent (moving forward while turning
        # traces a curve, not a straight line -- there's no single start/end-only
        # formula for it, same as a robot integrating body-frame velocity commands).
        # So instead of one big delta from enable time, we accumulate a fresh
        # incremental delta every frame: this frame's tiny room-frame motion,
        # reinterpreted through *this frame's* tracker/TCP orientation, added onto
        # the running target position (self.arm_manager.target_se3, not a snapshot).
        #
        # This does not IMU-style dead-reckoning drift: state["se3"] comes from
        # libsurvive's lighthouse-anchored pose solve, an absolute measurement each
        # frame (not itself an integrated/drifting quantity), so accumulated error
        # here is bounded by that measurement's noise, not an unbounded bias.
        raw_translation_delta_incremental = (
            self.state["se3"].translation - self._prev_vive_translation
        )
        translation_delta_tracker_local = (
            self.state["se3"].rotation.T @ raw_translation_delta_incremental
        )
        translation_delta_eef_local = (
            self.vive_to_eef_frame_rotation @ translation_delta_tracker_local
        )
        self._prev_vive_translation = self.state["se3"].translation.copy()

        target_translation = self.arm_manager.target_se3.translation + self.pos_scale * (
            target_rotation @ translation_delta_eef_local
        )

        target_se3 = pin.SE3(target_rotation, target_translation)

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
