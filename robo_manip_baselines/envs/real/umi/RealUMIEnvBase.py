import threading
import time
from os import path

import numpy as np
from gymnasium.spaces import Box, Dict

from robo_manip_baselines.common import (
    ArmConfig,
    DataKey,
    detect_aruco_tags,
    get_gripper_width,
    parse_aruco_config,
    preprocess_low_light_image,
)
from robo_manip_baselines.teleop import (
    Insta360InputDevice,
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
            # [g] from an M5Stack + load-cell scale mounted on/near the
            # handheld gripper; see RealEnvBase.setup_m5stack_scale. Zero
            # when no m5stack_ids is configured (no scale attached), same
            # convention as `wrench` above for a rig with no F/T sensor.
            "weight": Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
        }
    )

    def __init__(
        self,
        camera_ids=None,
        gelsight_ids=None,
        m5stack_ids=None,
        insta360_ids=None,
        pointcloud_camera_ids=None,
        pointcloud_camera_color_resolution=None,
        pointcloud_camera_color_exposure=None,
        pointcloud_camera_color_gain=None,
        gripper_marker_config=None,
        record_hz=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Recording rate [Hz]. TeleopBase paces its loop to env.dt, and with
        # no motor to command, nothing else here depends on dt -- so this is
        # simply the rate at which steps (and thus dataset samples) are
        # produced. None keeps RealEnvBase's 50 Hz nominal dt (in practice
        # the loop then just runs as fast as it can, at an irregular rate).
        if record_hz is not None:
            self.dt = 1.0 / record_hz

        # ArUco gripper-marker width tracking (see
        # common/utils/ArucoGripperUtils.py), following
        # https://github.com/real-stanford/universal_manipulation_interface's
        # approach: a small marker on each finger, tracked by the same
        # "hand" camera used for the recorded video. None disables it (the
        # gripper width then falls back to echoing the last commanded value,
        # see _get_obs()). See envs/configs/RealUMIDemo.yaml for the config
        # schema and calibration notes.
        self._gripper_marker_config = gripper_marker_config
        if gripper_marker_config is not None:
            self._gripper_marker_aruco = parse_aruco_config(
                {
                    "aruco_dict": gripper_marker_config["aruco_dict"],
                    "marker_size_map": gripper_marker_config["marker_size_map"],
                }
            )
            self._gripper_marker_camera_matrix = np.array(
                gripper_marker_config["camera_matrix"], dtype=np.float64
            )
            # Both optional -- default to "no distortion correction" / "no
            # contrast enhancement" respectively, matching the previous
            # (pre-real-hardware-test) behavior. Real testing against the
            # UMI rig's actual Orbbec Gemini camera found the raw color
            # frame's own distortion sizable enough to be worth correcting,
            # and the room dim enough that CLAHE made the difference between
            # detecting 0 and 2 of the two mounted markers -- see
            # preprocess_low_light_image's docstring for the measurement.
            dist_coeffs = gripper_marker_config.get("dist_coeffs")
            self._gripper_marker_dist_coeffs = (
                None if dist_coeffs is None else np.array(dist_coeffs, dtype=np.float64)
            )
            self._gripper_marker_clahe_clip_limit = gripper_marker_config.get(
                "clahe_clip_limit", 0.0
            )
            # Outlier rejection + smoothing -- added after real-hardware
            # testing showed the raw per-frame width occasionally jumping to
            # the extremes (reported as the displayed percent-closed
            # snapping to 0% or 100%). Root cause: a single spurious/
            # misdetected frame (bad corner localization under changing
            # light, an ID misread, etc.) can produce a width far outside
            # the calibrated [width_closed_m, width_open_m] range, and
            # _estimate_gripper_percent_closed_from_markers's final
            # np.clip(..., 0, 100) then snaps that one bad frame straight to
            # an extreme -- clipping was never the bug, it was faithfully
            # reporting a genuinely bad single-frame measurement. Two
            # independent layers of defense, both keyed on the WIDTH in
            # meters (before the open/closed->percent mapping, so their
            # behavior doesn't depend on how that calibration is tuned):
            #   1. max_width_jump_m: reject a reading that differs from the
            #      last ACCEPTED one by more than this in one step outright
            #      (physically implausible for a human hand at typical
            #      camera frame rates) -- holds the last smoothed value
            #      instead of applying it.
            #   2. width_smoothing_alpha: exponential moving average over
            #      accepted readings, so even in-range per-frame noise
            #      doesn't drive the displayed/recorded value directly.
            self._gripper_marker_max_width_jump_m = gripper_marker_config.get(
                "max_width_jump_m", 0.03
            )
            self._gripper_marker_width_smoothing_alpha = gripper_marker_config.get(
                "width_smoothing_alpha", 0.3
            )
            self._gripper_marker_smoothed_width_m = None
            # Last successfully estimated percent-closed, held while the
            # markers are momentarily undetected (occlusion, blur). The
            # commanded value is a poor fallback on this rig: the gripper is
            # read from the markers, not driven by the Vive button, so the
            # command just sits at its initial 0 (= fully open).
            self._gripper_marker_last_percent_closed = None
            # Optional (x0, y0, x1, y1) search box -- the finger markers sit
            # at a fixed image region (the camera is rigidly mounted on the
            # gripper), and AprilTag corner refinement cost scales with the
            # searched area. See detect_aruco_tags's roi argument.
            roi = gripper_marker_config.get("roi")
            self._gripper_marker_roi = None if roi is None else tuple(roi)
            # Detect in a background thread (one detection per new camera
            # frame) instead of inline in _get_obs(): even with the ROI,
            # AprilTag detection at 1280x800 takes ~70 ms, which otherwise
            # caps the whole teleop/recording loop at a few Hz. _get_obs()
            # then just reads the latest estimate (at most one frame stale).
            self._gripper_marker_async = gripper_marker_config.get(
                "async_detection", False
            )
            self._gripper_marker_lock = threading.Lock()
            self._gripper_marker_thread = None
            self._gripper_marker_stop_event = threading.Event()

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
        self.setup_gelsight(gelsight_ids)
        self.setup_m5stack_scale(m5stack_ids)
        self.setup_insta360(insta360_ids)
        # Orbbec Gemini, re-enabled as an interim "hand" camera / gripper-
        # marker source while insta360_bridge is not yet buildable (SDK
        # application pending, see envs/real/insta360_bridge/README.md).
        # Coexists with insta360_ids above -- both populate independent
        # camera-name buckets (pointcloud_camera_names vs rgb_camera_names),
        # so gripper_marker_config's camera_name can point at whichever one
        # is actually connected.
        self.setup_femtobolt(
            pointcloud_camera_ids,
            pointcloud_camera_color_resolution,
            pointcloud_camera_color_exposure,
            pointcloud_camera_color_gain,
        )

        if self._gripper_marker_config is not None and self._gripper_marker_async:
            self._gripper_marker_thread = threading.Thread(
                target=self._gripper_marker_detection_loop, daemon=True
            )
            self._gripper_marker_thread.start()

    def close(self):
        if getattr(self, "_gripper_marker_thread", None) is not None:
            self._gripper_marker_stop_event.set()
            self._gripper_marker_thread.join(timeout=1.0)
        super().close()

    @property
    def measured_keys_to_save(self):
        return super().measured_keys_to_save + [DataKey.MEASURED_WEIGHT]

    @property
    def has_gripper_marker_tracking(self):
        """Whether gripper_marker_config was configured (see __init__) --
        lets generic callers like TeleopBase's --plot_gripper_marker gate on
        this without reaching into the "_gripper_marker_config"
        implementation detail directly."""
        return self._gripper_marker_config is not None

    @property
    def latest_gripper_marker_percent_closed(self):
        """Most recent marker-based percent-closed estimate, without running a
        new detection; None if marker tracking is off or nothing has been
        detected since the last reset."""
        if self._gripper_marker_config is None:
            return None
        with self._gripper_marker_lock:
            return self._gripper_marker_last_percent_closed

    def setup_input_device(self, input_device_name, motion_manager, overwrite_kwargs):
        if input_device_name == "spacemouse":
            InputDeviceClass = SpacemouseInputDevice
        elif input_device_name == "keyboard":
            InputDeviceClass = KeyboardInputDevice
        elif input_device_name == "vive":
            InputDeviceClass = ViveInputDevice
        elif input_device_name == "insta360":
            InputDeviceClass = Insta360InputDevice
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
        if self._gripper_marker_config is not None:
            # Drop the smoothing filter's state so a new episode doesn't
            # start by holding over a stale width from before the reset.
            with self._gripper_marker_lock:
                self._gripper_marker_smoothed_width_m = None
                self._gripper_marker_last_percent_closed = None

    def _set_action(self, action, duration=None, joint_vel_limit_scale=0.5, wait=False):
        # There is no motor to command and thus no joint-velocity limit to
        # protect, so (unlike the teleoperated-robot envs) this deliberately
        # skips overwrite_command_for_safety(): its per-component clamp is
        # meant for real revolute-joint velocities, and does not have a
        # physically meaningful interpretation applied directly to a
        # free-flyer's raw (x, y, z, qx, qy, qz, qw) configuration vector.
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

        # `wait` is deliberately ignored: with no motor there is nothing to
        # wait for, and sleeping out `duration` here would come on top of
        # TeleopBase's own pacing of the loop to env.dt (adding a full dt of
        # dead time to every recorded step).

    def _estimate_gripper_percent_closed_from_markers(self):
        """Vision-based measured gripper width, from the two ArUco markers
        mounted on the gripper fingers (see __init__'s gripper_marker_config
        and common/utils/ArucoGripperUtils.py). While the markers are
        momentarily undetected, holds the last valid estimate. Returns None
        (not 0.0) only when there is no estimate at all yet -- no config, or
        no frame with detected markers since the last reset -- so the caller
        can fall back to echoing the last commanded value instead of
        reporting a bogus "fully open" reading.
        """
        if self._gripper_marker_config is None:
            return None

        if self._gripper_marker_async:
            with self._gripper_marker_lock:
                return self._gripper_marker_last_percent_closed

        frame = self.get_latest_rgb_camera_frame(
            self._gripper_marker_config["camera_name"]
        )
        if frame is None:
            return self._gripper_marker_last_percent_closed
        with self._gripper_marker_lock:
            return self._update_gripper_marker_estimate(frame)

    def _gripper_marker_detection_loop(self):
        """Background worker for async_detection: run one detection per new
        frame of the marker camera and publish the result for _get_obs()."""
        camera_name = self._gripper_marker_config["camera_name"]
        last_frame = None
        while not self._gripper_marker_stop_event.is_set():
            frame = self.get_latest_rgb_camera_frame(camera_name)
            if frame is None or frame is last_frame:
                time.sleep(0.002)
                continue
            last_frame = frame
            try:
                tag_dict = self._detect_gripper_markers(frame)
            except Exception as e:
                print(f"[{self.__class__.__name__}] Gripper marker detection failed: {e}")
                continue
            with self._gripper_marker_lock:
                self._update_gripper_marker_estimate(frame, tag_dict)

    def _detect_gripper_markers(self, frame):
        if self._gripper_marker_clahe_clip_limit > 0:
            frame = preprocess_low_light_image(
                frame, clip_limit=self._gripper_marker_clahe_clip_limit
            )

        tag_dict = detect_aruco_tags(
            frame,
            self._gripper_marker_aruco["aruco_dict"],
            self._gripper_marker_aruco["marker_size_map"],
            self._gripper_marker_camera_matrix,
            dist_coeffs=self._gripper_marker_dist_coeffs,
            corner_refinement=self._gripper_marker_aruco["corner_refinement"],
            roi=self._gripper_marker_roi,
        )
        return tag_dict

    def _update_gripper_marker_estimate(self, frame, tag_dict=None):
        """Turn one frame's marker detections into a filtered percent-closed
        and remember it. Caller holds _gripper_marker_lock. Returns the last
        valid estimate (or None if there has never been one) when the
        markers are not detected in this frame."""
        if tag_dict is None:
            tag_dict = self._detect_gripper_markers(frame)
        config = self._gripper_marker_config
        width_m = get_gripper_width(
            tag_dict,
            left_id=config["left_marker_id"],
            right_id=config["right_marker_id"],
            nominal_z=config["nominal_z"],
            z_tolerance=config.get("z_tolerance", 0.008),
        )
        if width_m is None:
            return self._gripper_marker_last_percent_closed

        # Outlier rejection: a single spurious frame (misdetected corner,
        # ID misread, etc.) can report a width far from reality -- discard
        # it rather than let it (via the clip below) snap the displayed/
        # recorded value straight to 0% or 100%. See __init__'s comment for
        # the full reasoning. Holds at the last SMOOTHED value (not None),
        # so a lone bad frame doesn't fall all the way back to echoing the
        # commanded value either.
        if (
            self._gripper_marker_smoothed_width_m is not None
            and abs(width_m - self._gripper_marker_smoothed_width_m)
            > self._gripper_marker_max_width_jump_m
        ):
            width_m = self._gripper_marker_smoothed_width_m

        # Exponential moving average over accepted readings, so even
        # in-range per-frame noise doesn't drive the reported value
        # directly.
        if self._gripper_marker_smoothed_width_m is None:
            self._gripper_marker_smoothed_width_m = width_m
        else:
            alpha = self._gripper_marker_width_smoothing_alpha
            self._gripper_marker_smoothed_width_m = (
                alpha * width_m + (1.0 - alpha) * self._gripper_marker_smoothed_width_m
            )

        # Percent-closed convention (0 = fully open, 100 = fully closed),
        # matching this env's action_space/gripper_joint_pos_actual.
        width_open_m = config["width_open_m"]
        width_closed_m = config["width_closed_m"]
        percent_closed = (
            100.0
            * (width_open_m - self._gripper_marker_smoothed_width_m)
            / (width_open_m - width_closed_m)
        )
        self._gripper_marker_last_percent_closed = float(
            np.clip(percent_closed, 0.0, 100.0)
        )
        return self._gripper_marker_last_percent_closed

    def _get_obs(self):
        # Sum across scales (multiple load cells reporting one combined
        # weight) rather than assuming exactly one; 0.0 when no
        # m5stack_ids was configured, matching `wrench`'s all-zero
        # convention for a rig with no sensor attached.
        weight = sum(
            self.get_m5stack_scale_data(scale_name)
            for scale_name in self.m5stack_scale_names
        )

        measured_gripper_percent_closed = (
            self._estimate_gripper_percent_closed_from_markers()
        )
        if measured_gripper_percent_closed is None:
            # No marker-based reading available this step (not configured,
            # or markers not currently visible/decodable) -- fall back to
            # echoing the last commanded value, same as before
            # gripper_marker_config existed.
            measured_gripper_joint_pos = self.gripper_joint_pos_actual
        else:
            measured_gripper_joint_pos = np.array([measured_gripper_percent_closed])

        return {
            "joint_pos": np.concatenate(
                (self.arm_joint_pos_actual, measured_gripper_joint_pos),
                dtype=np.float64,
            ),
            "joint_vel": np.zeros(8, dtype=np.float64),
            "wrench": np.zeros(6, dtype=np.float64),
            "weight": np.array([weight], dtype=np.float64),
        }
