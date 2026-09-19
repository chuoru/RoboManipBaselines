import concurrent.futures
import csv
import os
import re
import socket
import sys
import threading
import time
from abc import ABC, abstractmethod
from queue import Empty, Queue

import cv2
import gymnasium as gym
import numpy as np

from robo_manip_baselines.common import ArmConfig, DataKey, EnvDataMixin
from robo_manip_baselines.common import read_insta360_message


class RealEnvBase(EnvDataMixin, gym.Env, ABC):
    metadata = {
        "render_modes": [],
    }

    def __init__(
        self,
        # Diagnostic CSV of every commanded joint position, at each stage of
        # overwrite_command_for_safety/EMA smoothing, alongside the measured
        # position -- see log_command_for_safety_debug(). None disables it.
        # A timestamp is appended to the filename so repeated runs with the
        # same config don't clobber each other's log.
        command_log_path=None,
        # Scales the per-step joint velocity clamp applied in step() (see
        # overwrite_command_for_safety): effective limit is
        # joint_vel_limit_scale * self.joint_vel_limit. Lower this to slow
        # the arm's physical top speed WITHOUT changing --skip -- unlike
        # --skip, this does not touch how often the policy observes/re-
        # infers, so it does not distort the observation cadence the policy
        # was trained on (see RolloutBase.infer_policy/model_meta_info's
        # "skip"). 2.0 (the long-standing default before this was
        # configurable) matches teleop's responsiveness; e.g. 0.5 caps the
        # arm at 1/4 of that.
        joint_vel_limit_scale=2.0,
        # Stretches the control period (and with it the whole rollout) by
        # this factor: 2.0 runs everything at half speed. This -- NOT
        # joint_vel_limit_scale -- is the correct way to slow a closed-loop
        # policy down.
        #
        # Throttling with the velocity clamp instead breaks the loop: the
        # policy commands motion the clamp will not pass, the arm falls
        # behind, the policy then observes that lagging state and replans a
        # different trajectory, and the target reverses. Measured on the real
        # FR5 with a 15 deg/s clamp against a policy demanding ~75 deg/s: the
        # clamp fired on 43% of ticks, command-vs-measured reached 20 deg, and
        # the commanded joint angle flipped direction on ~50% of ticks -- fast,
        # jerky, oscillating motion.
        #
        # Stretching the period keeps the loop consistent instead. Both the
        # observation spacing and the arm's speed scale together, so the
        # SPATIAL change between the policy's n_obs_steps observations is
        # unchanged from training (spacing s*dt at speed v/s gives the same
        # v*dt displacement), which is what the policy actually conditions
        # on. Same idea as ReplayUmiOnFairino5.py's --time_scale.
        time_scale=1.0,
        **kwargs,
    ):
        # Setup environment parameters
        self.init_time = time.time()
        if time_scale <= 0.0:
            raise ValueError(
                f"[{self.__class__.__name__}] time_scale must be positive: {time_scale}"
            )
        self.time_scale = time_scale
        self.dt = 0.02 * time_scale  # [s]
        self.joint_vel_limit_scale = joint_vel_limit_scale
        # Wall-clock time of the previous step(), used to measure the real
        # control period for the velocity clamp. None means "no step since
        # the last reset", in which case the nominal dt is used for that
        # first step.
        self._last_step_time = None
        # Real elapsed control period used to scale the velocity clamp; see
        # step() and overwrite_command_for_safety().
        self._clamp_duration = None
        # Gates command transmission. Subclasses that stream to hardware
        # (e.g. RealFairino5EnvBase) own the normal enable/disable flow; it is
        # defined here so overwrite_command_for_safety's abort path can shut
        # motion off on any Real*Env, not just those that happen to define it.
        self._motion_enabled = False
        self.world_random_scale = None
        # Anchor for the velocity clamp in overwrite_command_for_safety. None
        # means "no command issued since the last reset", in which case the
        # clamp anchors on the measured position for that first command.
        self._prev_arm_joint_pos_command = None

        self._command_log_file = None
        self._command_log_writer = None
        if command_log_path is not None:
            root, ext = os.path.splitext(command_log_path)
            timestamped_path = f"{root}_{time.strftime('%Y%m%d_%H%M%S')}{ext or '.csv'}"
            log_dir = os.path.dirname(timestamped_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            self._command_log_file = open(timestamped_path, "w", newline="")
            self._command_log_writer = csv.writer(self._command_log_file)
            self._command_log_writer.writerow(
                [
                    "t",
                    "duration",
                    "wait",
                    *[f"measured_deg_{i}" for i in range(6)],
                    *[f"raw_command_deg_{i}" for i in range(6)],
                    *[f"safety_command_deg_{i}" for i in range(6)],
                    *[f"sent_deg_{i}" for i in range(6)],
                    "gripper_percent_closed",
                    "hit_hard_clip",
                    "max_abs_sent_vs_measured_deg",
                ]
            )
            self._command_log_file.flush()
            print(
                f"[{self.__class__.__name__}] Logging commanded/measured joint "
                f"positions to {timestamped_path}"
            )

        # Setup device variables
        self.cameras = {}
        self.pointcloud_cameras = {}
        self.rgb_tactiles = {}
        self.intensity_tactiles = {}
        self.m5stack_scales = {}
        self.rgb_cameras = {}

    def log_command_for_safety_debug(
        self,
        duration_arg,
        wait,
        measured_deg,
        raw_command_deg,
        safety_command_deg,
        sent_deg,
        gripper_percent_closed,
    ):
        if self._command_log_writer is None:
            return

        hit_hard_clip = bool(
            not np.allclose(raw_command_deg, safety_command_deg, atol=1e-6)
        )
        max_abs_sent_vs_measured_deg = float(np.max(np.abs(sent_deg - measured_deg)))
        self._command_log_writer.writerow(
            [
                time.time() - self.init_time,
                duration_arg,
                wait,
                *[f"{v:.4f}" for v in measured_deg],
                *[f"{v:.4f}" for v in raw_command_deg],
                *[f"{v:.4f}" for v in safety_command_deg],
                *[f"{v:.4f}" for v in sent_deg],
                f"{gripper_percent_closed:.2f}",
                hit_hard_clip,
                f"{max_abs_sent_vs_measured_deg:.4f}",
            ]
        )
        self._command_log_file.flush()

    def close_command_log(self):
        if self._command_log_file is not None:
            self._command_log_file.close()
            self._command_log_file = None
            self._command_log_writer = None

    def setup_realsense(self, camera_ids):
        if camera_ids is None:
            return

        from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids

        detected_camera_ids = get_device_ids()
        for camera_name, camera_id in camera_ids.items():
            if camera_id not in detected_camera_ids:
                raise RuntimeError(
                    f"[{self.__class__.__name__}] Specified camera (name: {camera_name}, ID: {camera_id}) not detected. Detected camera IDs: {detected_camera_ids}"
                )

            camera = RealSenseCamera(device_id=camera_id, flip=False)
            frames = camera._pipeline.wait_for_frames()
            color_intrinsics = (
                frames.get_color_frame().profile.as_video_stream_profile().intrinsics
            )
            camera.color_fovy = np.rad2deg(
                2 * np.arctan(color_intrinsics.height / (2 * color_intrinsics.fy))
            )
            depth_intrinsics = (
                frames.get_depth_frame().profile.as_video_stream_profile().intrinsics
            )
            camera.depth_fovy = np.rad2deg(
                2 * np.arctan(depth_intrinsics.height / (2 * depth_intrinsics.fy))
            )

            self.cameras[camera_name] = camera

    def setup_femtobolt(
        self,
        pointcloud_camera_ids,
        color_resolution=None,
        color_exposure=None,
        color_gain=None,
    ):
        """color_resolution: None (default) keeps the previous behavior --
        request the sensor's default color profile and resize/deliver at
        (640, 480), same as always. Pass (width, height, fps) -- e.g.
        (1280, 800, 10), a resolution this camera was confirmed to support
        via its color stream profile list -- to request that profile
        instead and deliver frames at ITS native size with no further
        resize, applied to every camera set up in this call.

        Added for ArUco/AprilTag gripper-marker tracking
        (RealUMIEnvBase._estimate_gripper_percent_closed_from_markers):
        real-hardware testing found detection unreliable at the default
        640x480 delivery size, and the actual bottleneck was marker pixel
        footprint (measured ~4.3 px per tag module at 640x480, well under
        the ~8-10 px/module AprilTag/ArUco detectors want for robust
        decoding) rather than lighting/CLAHE tuning -- 1280x800 roughly
        doubles that to ~8.5 px/module. This is a resolution the *sensor*
        needs to actually support (see e.g.
        third_party/pyorbbecsdk/examples/color.py to list a camera's own
        supported color profiles); it is not a request that gets upscaled
        artificially.

        color_exposure/color_gain: None (default) leaves auto-exposure on,
        as before. Pass both (e.g. color_exposure=40, color_gain=150 --
        confirmed to still produce a usable, sharp image on this rig's
        Orbbec Gemini 305, vs. its auto-exposure default of exposure=100
        gain=16) to force a short, FIXED exposure time instead. Also added
        for gripper-marker detection: auto-exposure lengthens exposure time
        in dimmer rooms to keep brightness up, and a longer exposure means
        any hand/gripper motion during that window smears into visible
        motion blur -- which was measured to break marker decoding outright
        even in a since-brightness-corrected, reasonably-lit frame (a
        blurred marker's bit pattern is illegible regardless of overall
        brightness/contrast). A short, capped exposure trades this for more
        sensor noise (raising color_gain compensates the resulting
        darkness) -- noise degrades a detector's bit-thresholding far less
        than blur destroys the sharp edges it depends on. Applied to every
        camera set up in this call; use
        third_party/pyorbbecsdk's OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT /
        OB_PROP_COLOR_GAIN_INT ranges (device.get_int_property_range) to
        find this camera's own valid min/max if retuning.
        """
        if pointcloud_camera_ids is None:
            return

        self._femtobolt_color_resolution = color_resolution
        self._femtobolt_color_exposure = color_exposure
        self._femtobolt_color_gain = color_gain

        sys.path.append(
            os.path.join(os.path.dirname(__file__), "../../../third_party/pyorbbecsdk")
        )
        from pyorbbecsdk import Context

        ctx = Context()
        device_list = ctx.query_devices()
        curr_device_cnt = device_list.get_count()
        try:
            for (
                pointcloud_camera_name,
                pointcloud_camera_id,
            ) in pointcloud_camera_ids.items():
                # A string ID is matched against the device's serial number (stable
                # across reboots/replugs, e.g. for multi-camera rigs); an int ID is
                # matched by enumeration index (order is not guaranteed to be stable).
                if isinstance(pointcloud_camera_id, str):
                    try:
                        device = device_list.get_device_by_serial_number(
                            pointcloud_camera_id
                        )
                    except Exception as e:
                        # get_device_by_serial_number() can fail even though the
                        # camera is fully enumerated and openable -- observed when
                        # its USB string-descriptor read is momentarily flaky (a
                        # cable/connector issue), which get_device_by_index()
                        # doesn't seem to depend on. Fall back to matching the
                        # serial against the by-index listing before giving up.
                        device = None
                        for device_idx in range(curr_device_cnt):
                            if (
                                device_list.get_device_serial_number_by_index(
                                    device_idx
                                )
                                == pointcloud_camera_id
                            ):
                                device = device_list.get_device_by_index(device_idx)
                                break
                        if device is None:
                            raise RuntimeError(
                                f"[{self.__class__.__name__}] Specified camera (name: {pointcloud_camera_name}, serial: {pointcloud_camera_id}) not detected: {e}"
                            )
                else:
                    if pointcloud_camera_id > curr_device_cnt:
                        raise RuntimeError(
                            f"[{self.__class__.__name__}] Specified camera (name: {pointcloud_camera_name}, ID: {pointcloud_camera_id}) not detected. Max camera ID: {curr_device_cnt}"
                        )
                    device = device_list.get_device_by_index(pointcloud_camera_id)

                self.pointcloud_cameras[pointcloud_camera_name] = {
                    "queue": Queue(),
                    "device": device,
                    # Tracks consecutive frame timeouts, to trigger a pipeline restart
                    # after repeated misses (see get_pointcloud_camera_data).
                    "consecutive_failures": 0,
                    # Last successfully processed frame, reused as a frozen fallback
                    # image while a flaky camera is being reconnected, so a dropped
                    # camera doesn't crash or stall the teleop loop.
                    "last_result": None,
                    # Wall-clock time the most recent frame arrived. Used to tell
                    # "the loop is simply faster than the camera" (fine, reuse the
                    # last frame) from "the camera has actually gone quiet"
                    # (fault -> reconnect). See get_pointcloud_camera_data().
                    "last_frame_time": None,
                    # True while a background reconnect attempt is in flight, to avoid
                    # piling up multiple concurrent reconnect threads for one camera.
                    "reconnecting": False,
                }
                self._start_femtobolt_pipeline(pointcloud_camera_name)
        except Exception:
            # If one camera (of possibly several) fails partway through, any
            # pipelines already started above must be stopped before re-raising --
            # otherwise they're left open with nothing tracking them (this __init__
            # call never completes, so there's no env to close() later), and the
            # next run's uvc_open on those devices fails until they're USB-reset.
            for pointcloud_camera in self.pointcloud_cameras.values():
                pipeline = pointcloud_camera.get("pipeline")
                if pipeline is not None:
                    try:
                        pipeline.stop()
                    except Exception as stop_error:
                        print(
                            f"[{self.__class__.__name__}] Error stopping pointcloud "
                            f"camera pipeline during cleanup: {stop_error}"
                        )
            raise

    def _start_femtobolt_pipeline(self, pointcloud_camera_name):
        """(Re)start the pyorbbecsdk pipeline for one Orbbec camera. Used both for
        the initial connection and to recover a camera whose frame callback has
        stopped firing (flaky USB link).

        Both color and depth streams are enabled. Note: depth streaming roughly
        doubles the USB bandwidth/CPU load per camera, which was previously
        implicated in cameras dropping out on this rig when running 3 of them at
        once -- if that instability comes back, disabling the depth
        enable_stream() call below (and reverting get_pointcloud_camera_data()'s
        depth extraction to the zero placeholder) is the known-stable fallback."""
        from pyorbbecsdk import Config, OBFormat, OBSensorType, Pipeline

        pointcloud_camera = self.pointcloud_cameras[pointcloud_camera_name]
        device = pointcloud_camera["device"]

        color_exposure = getattr(self, "_femtobolt_color_exposure", None)
        color_gain = getattr(self, "_femtobolt_color_gain", None)
        if color_exposure is not None or color_gain is not None:
            from pyorbbecsdk import OBPropertyID

            # Manual exposure requires auto-exposure off first, or
            # set_int_property(..._EXPOSURE_INT) is silently overridden by
            # the auto-exposure loop on the next frame.
            device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, False)
            if color_exposure is not None:
                device.set_int_property(
                    OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, color_exposure
                )
            if color_gain is not None:
                device.set_int_property(OBPropertyID.OB_PROP_COLOR_GAIN_INT, color_gain)

        pipeline = Pipeline(device)
        config = Config()
        color_profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        if color_profile_list is None:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Camera '{pointcloud_camera_name}' has no "
                "color sensor."
            )
        color_resolution = getattr(self, "_femtobolt_color_resolution", None)
        if color_resolution is None:
            color_profile = color_profile_list.get_default_video_stream_profile()
            pointcloud_camera["color_target_size"] = (640, 480)
        else:
            width, height, fps = color_resolution
            color_profile = color_profile_list.get_video_stream_profile(
                width, height, OBFormat.RGB, fps
            )
            pointcloud_camera["color_target_size"] = (
                color_profile.get_width(),
                color_profile.get_height(),
            )
        config.enable_stream(color_profile)

        depth_profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        if depth_profile_list is None:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Camera '{pointcloud_camera_name}' has no "
                "depth sensor."
            )
        depth_profile = depth_profile_list.get_default_video_stream_profile()
        config.enable_stream(depth_profile)

        pipeline.start(
            config,
            lambda frame_set,
            pointcloud_camera_name=pointcloud_camera_name: self.femtobolt_callback(
                frame_set, pointcloud_camera_name
            ),
        )

        pointcloud_camera["pipeline"] = pipeline
        pointcloud_camera["consecutive_failures"] = 0
        # Restarting counts as "no frame received yet" for staleness purposes,
        # so the fresh pipeline gets a full RECONNECT_TIMEOUT_SEC to deliver
        # its first frame before being judged quiet again.
        pointcloud_camera["last_frame_time"] = None

        # Drain any stale frames left in the queue from before a restart.
        while not pointcloud_camera["queue"].empty():
            try:
                pointcloud_camera["queue"].get_nowait()
            except Empty:
                break

    def _reconnect_femtobolt_pipeline(self, pointcloud_camera_name):
        """Runs in a background daemon thread (see get_pointcloud_camera_data) so a
        camera whose USB link is bad enough that even a fresh connect attempt hangs
        in native code doesn't take the whole process down with it."""
        pointcloud_camera = self.pointcloud_cameras[pointcloud_camera_name]
        try:
            try:
                pointcloud_camera["pipeline"].stop()
            except Exception as e:
                print(
                    f"[{self.__class__.__name__}] Error stopping pipeline for "
                    f"'{pointcloud_camera_name}': {e}"
                )
            self._start_femtobolt_pipeline(pointcloud_camera_name)
            print(
                f"[{self.__class__.__name__}] Restarted pointcloud camera "
                f"'{pointcloud_camera_name}'."
            )
        except Exception as e:
            print(
                f"[{self.__class__.__name__}] Failed to restart pointcloud camera "
                f"'{pointcloud_camera_name}': {e}"
            )
        finally:
            pointcloud_camera["reconnecting"] = False

    def femtobolt_callback(self, frames, pointcloud_camera_name):
        if frames is None:
            return

        pointcloud_camera = self.pointcloud_cameras[pointcloud_camera_name]
        queue = pointcloud_camera["queue"]
        if queue.qsize() >= 5:
            queue.get()
        queue.put(frames)

    def setup_gelsight(self, gelsight_ids):
        if gelsight_ids is None:
            return

        for rgb_tactile_name, gelsight_id in gelsight_ids.items():
            for device_name in os.listdir("/sys/class/video4linux"):
                real_device_name = os.path.realpath(
                    "/sys/class/video4linux/" + device_name + "/name"
                )
                with (
                    open(real_device_name, "rt") as device_name_file
                ):  # "rt": read-text mode ("t" is default, so "r" alone is the same)
                    detected_gelsight_id = device_name_file.read().rstrip()
                if gelsight_id in detected_gelsight_id:
                    tactile_num = int(re.search("\d+$", device_name).group(0))
                    print(
                        f"[{self.__class__.__name__}] Found GelSight sensor. ID: {detected_gelsight_id}, device: {device_name}, num: {tactile_num}"
                    )

                    rgb_tactile = cv2.VideoCapture(tactile_num)
                    if rgb_tactile is None or not rgb_tactile.isOpened():
                        print(
                            f"[{self.__class__.__name__}] Unable to open video source of GelSight sensor."
                        )
                        continue

                    self.rgb_tactiles[rgb_tactile_name] = rgb_tactile
                    break

            if rgb_tactile_name not in self.rgb_tactiles:
                raise RuntimeError(
                    f"[{self.__class__.__name__}] Specified GelSight (name: {rgb_tactile_name}, ID: {gelsight_id}) not detected."
                )

    def setup_sanwa_keyboard(self, sanwa_keyboard_ids):
        if sanwa_keyboard_ids is None:
            return

        import hid

        for intensity_tactile_name, device_path in sanwa_keyboard_ids.items():
            if not os.path.exists(device_path):
                raise RuntimeError(
                    f"[{self.__class__.__name__}] Specified keyboard (path: {device_path}) not detected."
                )

            intensity_tactile_device = hid.Device(path=device_path.encode())
            if intensity_tactile_device is None:
                print(
                    f"[{self.__class__.__name__}] Unable to open keyboard (path: {device_path})."
                )
                continue
            intensity_tactile_buf = np.zeros(shape=(2, 3), dtype=np.uint8)
            intensity_tactile = {
                "device": intensity_tactile_device,
                "buf": intensity_tactile_buf,
            }

            self.intensity_tactiles[intensity_tactile_name] = intensity_tactile

    # Baud rate of the M5Stack scale's serial protocol (see
    # setup_m5stack_scale / _read_m5stack_scale_loop below).
    M5STACK_SCALE_BAUDRATE = 115200

    def setup_m5stack_scale(self, m5stack_ids):
        """Connect to one or more M5Stack + load-cell weight-scale rigs over
        USB serial (e.g. mounted on/near a UMI handheld gripper).

        m5stack_ids: dict mapping a scale name to a substring of the
        device's /dev/serial/by-id/* symlink (its USB serial number, stable
        across replugs/reboots -- unlike /dev/ttyACM<N>'s enumeration index,
        which can shift). Find it with `ls /dev/serial/by-id/`.

        Each matched device is read in a background daemon thread (see
        _read_m5stack_scale_loop) rather than inline in _get_obs(), so a
        per-step serial readline() cannot add latency to the ~50 Hz control
        loop -- the same reasoning as the femtobolt pointcloud camera's
        queue/thread setup above. get_m5stack_scale_data() below just returns
        the latest value the background thread has already parsed.
        """
        if m5stack_ids is None:
            return

        import serial

        by_id_dir = "/dev/serial/by-id"
        detected_device_names = (
            os.listdir(by_id_dir) if os.path.isdir(by_id_dir) else []
        )

        for scale_name, m5stack_id in m5stack_ids.items():
            device_path = None
            for device_name in detected_device_names:
                if m5stack_id in device_name:
                    device_path = os.path.join(by_id_dir, device_name)
                    break

            if device_path is None:
                raise RuntimeError(
                    f"[{self.__class__.__name__}] Specified M5Stack scale (name: "
                    f"{scale_name}, ID: {m5stack_id}) not detected in {by_id_dir}. "
                    f"Detected devices: {detected_device_names}"
                )

            connection = serial.Serial(
                device_path, self.M5STACK_SCALE_BAUDRATE, timeout=1.0
            )
            print(
                f"[{self.__class__.__name__}] Found M5Stack scale. name: "
                f"{scale_name}, device: {device_path}"
            )

            scale = {
                "connection": connection,
                "lock": threading.Lock(),
                "latest_weight": 0.0,
                "last_update_time": None,
                "stop_event": threading.Event(),
            }
            self.m5stack_scales[scale_name] = scale

            thread = threading.Thread(
                target=self._read_m5stack_scale_loop,
                args=(scale_name,),
                daemon=True,
            )
            scale["thread"] = thread
            thread.start()

    @staticmethod
    def _parse_m5stack_weight(line):
        """Parse one line of the M5Stack scale's serial protocol, e.g.:
            b'[RUN] raw=-460395 weight=6.11 HR1=61 HR5=0 offset=-431515 scale=-4725.901855 state=0\\n'
        Returns the weight in grams, or None if the line has no `weight=`
        field (e.g. a startup/log line, or a garbled read)."""
        try:
            text = line.decode("ascii", errors="ignore")
        except AttributeError:
            text = line

        for token in text.split():
            if token.startswith("weight="):
                try:
                    return float(token[len("weight=") :])
                except ValueError:
                    return None

        return None

    def _read_m5stack_scale_loop(self, scale_name):
        """Runs in a background daemon thread (see setup_m5stack_scale)."""
        scale = self.m5stack_scales[scale_name]
        connection = scale["connection"]
        stop_event = scale["stop_event"]

        while not stop_event.is_set():
            try:
                line = connection.readline()
            except Exception as e:
                print(
                    f"[{self.__class__.__name__}] Error reading M5Stack scale "
                    f"'{scale_name}': {e}"
                )
                stop_event.wait(0.1)
                continue

            if not line:
                # readline() timed out (see M5STACK_SCALE_BAUDRATE's Serial()
                # call above) with no newline received -- not itself a fault,
                # just loop back and check stop_event.
                continue

            weight = self._parse_m5stack_weight(line)
            if weight is None:
                continue

            with scale["lock"]:
                scale["latest_weight"] = weight
                scale["last_update_time"] = time.time()

    def get_m5stack_scale_data(self, scale_name):
        """Get the most recently received weight [g] from the named M5Stack
        scale. Returns 0.0 if no reading has arrived yet since connecting."""
        scale = self.m5stack_scales[scale_name]
        with scale["lock"]:
            return scale["latest_weight"]

    @property
    def m5stack_scale_names(self):
        """Get names of connected M5Stack scales."""
        return list(self.m5stack_scales.keys())

    def setup_insta360(self, camera_ids):
        """Connect to one or more Insta360 cameras via the insta360_bridge
        helper process (see envs/real/insta360_bridge/) -- a separate C++
        process that talks to the Insta360 CameraSDK and ORB-SLAM3, and
        streams decoded RGB frames and estimated 6-DoF poses over a local
        Unix domain socket (see common/utils/Insta360Protocol.py for the wire
        format). This method only consumes "frame" messages, for the
        rgb_cameras bucket below; Insta360InputDevice separately connects to
        the same socket to consume "pose" messages.

        camera_ids: dict mapping a camera name to the bridge's Unix domain
        socket path for that camera, e.g. {"hand": "/tmp/insta360_hand.sock"}.

        Like setup_femtobolt's pointcloud cameras, each camera is read in a
        background daemon thread rather than inline in _get_info(), so a
        stalled bridge/socket read cannot add latency to the ~50 Hz control
        loop -- get_rgb_camera_data() below just returns the latest frame the
        background thread has already received.
        """
        if camera_ids is None:
            return

        for camera_name, socket_path in camera_ids.items():
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.connect(socket_path)
            print(
                f"[{self.__class__.__name__}] Connected to Insta360 bridge. name: "
                f"{camera_name}, socket: {socket_path}"
            )

            camera = {
                "connection": connection,
                "lock": threading.Lock(),
                "latest_frame": None,
                "last_update_time": None,
                "stop_event": threading.Event(),
            }
            self.rgb_cameras[camera_name] = camera

            thread = threading.Thread(
                target=self._read_insta360_frame_loop,
                args=(camera_name,),
                daemon=True,
            )
            camera["thread"] = thread
            thread.start()

    def _read_insta360_frame_loop(self, camera_name):
        """Runs in a background daemon thread (see setup_insta360)."""
        camera = self.rgb_cameras[camera_name]
        connection = camera["connection"]
        stop_event = camera["stop_event"]

        while not stop_event.is_set():
            try:
                message = read_insta360_message(connection)
            except Exception as e:
                if not stop_event.is_set():
                    print(
                        f"[{self.__class__.__name__}] Error reading Insta360 "
                        f"bridge camera '{camera_name}': {e}"
                    )
                break

            if message["type"] != "frame":
                continue

            with camera["lock"]:
                camera["latest_frame"] = message["frame"]
                camera["last_update_time"] = time.time()

    def get_rgb_camera_data(self, camera_name, camera):
        """Get the most recently received frame from the named Insta360
        bridge camera, matching the {"rgb_images": ...} shape get_rgb_tactile_data
        returns. Returns a black placeholder frame if no frame has arrived
        yet since connecting."""
        with camera["lock"]:
            frame = camera["latest_frame"]
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return camera_name, {"rgb_images": frame}

    @property
    def rgb_camera_names(self):
        """Get names of connected Insta360 (or other bridge-fed RGB-only) cameras."""
        return list(self.rgb_cameras.keys())

    def get_latest_rgb_camera_frame(self, camera_name):
        """Get the most recently received frame for camera_name, checking
        every camera bucket that maintains a cached "latest frame" --
        rgb_cameras (see setup_insta360) and pointcloud_cameras (see
        setup_femtobolt/Orbbec Gemini; used as an interim marker-tracking
        source while insta360_bridge is not yet available, see
        envs/real/insta360_bridge/README.md). Returns None if camera_name
        isn't configured in either, or no frame has arrived yet.

        Unlike get_rgb_camera_data/get_pointcloud_camera_data (used by
        _get_info()'s per-step image bucket, which fall back to a
        black/last-known placeholder so the dataset never has a missing
        entry), this returns None in that case -- callers that need a REAL
        frame to do vision processing on (e.g. ArUco gripper-marker
        tracking) should treat None as "no estimate available this step",
        not process a placeholder.

        self.cameras (RealSense) is deliberately not checked here: unlike
        the other two buckets it has no cached "latest frame" of its own
        (each read is a live, uncached hardware call), so returning one
        here would mean a SECOND live grab beyond what _get_info() already
        does that step -- redundant, and not comparable in freshness to the
        other two buckets' semantics.
        """
        rgb_camera = self.rgb_cameras.get(camera_name)
        if rgb_camera is not None:
            with rgb_camera["lock"]:
                return rgb_camera["latest_frame"]

        pointcloud_camera = self.pointcloud_cameras.get(camera_name)
        if pointcloud_camera is not None:
            last_result = pointcloud_camera.get("last_result")
            if last_result is not None:
                return last_result.get("rgb_images")

        return None

    def get_input_device_kwargs(self, input_device_name):
        return {}

    def reset(self, *, seed=None, options=None):
        self.init_time = time.time()

        super().reset(seed=seed)

        self._reset_robot()

        # Drop the velocity clamp's anchor so the first command of the new
        # episode re-anchors on the arm's true position instead of a stale
        # command from the previous one (see overwrite_command_for_safety).
        # Done centrally so every Real*EnvBase gets it; subclasses that move
        # the arm OUTSIDE reset() (e.g. RealFairino5EnvBase.move_to_init_pose)
        # must clear it themselves as well.
        self._prev_arm_joint_pos_command = None
        # Likewise drop the step-period anchor, so the first step of the new
        # episode does not measure a "period" that spans the whole reset.
        self._last_step_time = None
        self._clamp_duration = None

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    # Bounds on the measured control period used as the velocity clamp's
    # duration below. The lower bound keeps a pair of unusually fast
    # back-to-back steps from collapsing the allowed motion to ~0.
    #
    # The upper bound is a SAFETY limit on how much motion a single command
    # may authorize, and must stay small. The clamp budgets this step's
    # allowed motion from the PREVIOUS step's measured period, so an
    # occasional slow tick (policy inference spike, camera/XML-RPC stall)
    # hands the next command a correspondingly large allowance -- which the
    # arm then executes as one lurch. Measured on hardware at 37 Hz with a
    # 0.5s bound and a 30 deg/s limit: a 0.31s inference tick authorized a
    # 9.4 deg single-step jump, and the log showed |sent - measured| peaking
    # at 9.77 deg amid otherwise ~0.1 deg steps -- felt like a runaway.
    # 0.1s bounds that worst case to 3 deg while still covering the normal
    # loop period (~0.03s) with room to spare.
    STEP_DURATION_MIN_SEC = 0.004
    STEP_DURATION_MAX_SEC = 0.1

    def step(self, action):
        # Measure the REAL elapsed control period rather than assuming the
        # nominal self.dt. The loop's actual rate is not a stable self.dt:
        # policy inference (tens to hundreds of ms, and only on the ticks
        # where it runs), camera reads, and synchronous XML-RPC state
        # readback all add jittery latency -- measured on this rig at
        # 26ms..472ms against a nominal dt of 20ms.
        #
        # This duration is what overwrite_command_for_safety multiplies by
        # the velocity limit to decide how far the command may move THIS
        # tick. Passing the fixed self.dt while the real period is 5-20x
        # longer starves every slow tick (it may only advance 20ms worth of
        # motion no matter how long it actually took), so the commanded
        # trajectory falls progressively behind and then lurches forward on
        # the next fast tick -- exactly the "joint command spikes" /
        # staircase jerk seen on hardware. RealFairino5EnvBase._set_action
        # already derives ServoJ's own cmdT from measured elapsed time for
        # the same reason; this keeps the safety clamp consistent with it.
        # NOTE this is deliberately NOT passed as _set_action's `duration`:
        # that argument doubles as the loop's pacing target (_set_action
        # sleeps out the remainder of `duration` when wait=True). Feeding the
        # measured period back in there is positive feedback -- each step
        # sleeps until it is at least as long as the previous one, so the
        # period ratchets upward until it pins at STEP_DURATION_MAX_SEC
        # (observed: the loop collapsing to 2 Hz with the arm sitting still
        # for 59% of ticks). The clamp reads it off self instead; `duration`
        # stays the nominal dt so pacing is unchanged.
        now = time.time()
        if self._last_step_time is None:
            self._clamp_duration = self.dt
        else:
            self._clamp_duration = float(
                np.clip(
                    now - self._last_step_time,
                    self.STEP_DURATION_MIN_SEC,
                    self.STEP_DURATION_MAX_SEC,
                )
            )
        self._last_step_time = now

        self._set_action(
            action,
            duration=self.dt,
            joint_vel_limit_scale=self.joint_vel_limit_scale,
            wait=True,
        )

        observation = self._get_obs()
        reward = 0.0
        terminated = False
        info = self._get_info()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def close(self):
        # Stop the Orbbec streaming pipelines explicitly. If the process exits
        # while they are still streaming, the cameras can be left in a state
        # where the next uvc_open fails (requiring a USB reset or replug).
        for pointcloud_camera_name, pointcloud_camera in self.pointcloud_cameras.items():
            pipeline = pointcloud_camera.get("pipeline")
            if pipeline is None:
                continue
            try:
                pipeline.stop()
            except Exception as e:
                print(
                    f"[{self.__class__.__name__}] Error stopping pointcloud camera "
                    f"'{pointcloud_camera_name}': {e}"
                )

        for rgb_tactile in self.rgb_tactiles.values():
            try:
                rgb_tactile.release()
            except Exception as e:
                print(f"[{self.__class__.__name__}] Error releasing GelSight: {e}")

        for scale_name, scale in self.m5stack_scales.items():
            scale["stop_event"].set()
            scale["thread"].join(timeout=1.0)
            try:
                scale["connection"].close()
            except Exception as e:
                print(
                    f"[{self.__class__.__name__}] Error closing M5Stack scale "
                    f"'{scale_name}': {e}"
                )

        for camera_name, camera in self.rgb_cameras.items():
            camera["stop_event"].set()
            try:
                # Unblock a thread parked in a blocking recv() on this socket
                # -- unlike the M5Stack scale's readline() (which has its own
                # 1.0s timeout), Insta360Protocol.read_message() has no
                # timeout of its own, so closing alone would not wake it.
                camera["connection"].shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            camera["thread"].join(timeout=1.0)
            try:
                camera["connection"].close()
            except Exception as e:
                print(
                    f"[{self.__class__.__name__}] Error closing Insta360 bridge "
                    f"camera '{camera_name}': {e}"
                )

        self.close_command_log()

    @abstractmethod
    def _reset_robot(self):
        pass

    @abstractmethod
    def _set_action(self):
        pass

    # How far the commanded joint position may run ahead of the measured one
    # before overwrite_command_for_safety warns. Generous enough not to fire
    # on normal servo lag, small enough to catch an arm that has actually
    # stopped following.
    joint_pos_tracking_error_warn_threshold = np.deg2rad(15.0)  # [rad]

    # Hard stop. Past this much command-vs-measured error the command stream
    # has demonstrably lost contact with the arm, and continuing to send
    # targets it cannot reach is how a bad command turns into a runaway.
    # Gate motion off and raise instead: no further ServoJ target is
    # transmitted, so the arm holds wherever it currently is.
    joint_pos_tracking_error_abort_threshold = np.deg2rad(30.0)  # [rad]

    def overwrite_command_for_safety(self, action, duration, joint_vel_limit_scale):
        arm_joint_idxes = np.concatenate(
            [
                body_config.arm_action_idxes
                for body_config in self.body_config_list
                if isinstance(body_config, ArmConfig)
            ]
        )
        # Clip to the physical joint range before anything else, so an IK
        # solution that overshoots a joint limit (e.g. while chasing a
        # teleop target near the edge of the workspace) can't reach the robot
        # and fault/E-stop it -- the velocity clamp below only limits how fast
        # a joint moves, not where it's allowed to end up.
        action[arm_joint_idxes] = np.clip(
            action[arm_joint_idxes],
            self.action_space.low[arm_joint_idxes],
            self.action_space.high[arm_joint_idxes],
        )
        arm_joint_pos_command = action[arm_joint_idxes]
        scaled_joint_vel_limit = (
            np.clip(joint_vel_limit_scale, 0.01, 10.0) * self.joint_vel_limit
        )

        if duration is None:
            duration_min, duration_max = 0.1, 10.0  # [s]
            duration = np.clip(
                np.max(
                    np.abs(arm_joint_pos_command - self.arm_joint_pos_actual)
                    / scaled_joint_vel_limit
                ),
                duration_min,
                duration_max,
            )
        else:
            # A large per-step joint delta (e.g. from an IK solution jump near a
            # wrist singularity) is clamped below by scaled_joint_vel_limit rather
            # than rejected outright, so a single noisy/singular command doesn't
            # crash the teleop session -- the velocity clamp is what actually
            # keeps the physical motion safe.
            #
            # The clamp is anchored on the PREVIOUS COMMAND, not on the
            # measured position. That distinction is the whole point: for a
            # stream of position commands to a servo, "velocity limit" means
            # how fast the COMMAND may move, and anchoring on the measurement
            # instead turns the robot's own tracking error into a throttle on
            # the command:
            #   sent = measured + clip(command - measured, +/-limit)
            # Once the arm falls even slightly behind, `command - measured`
            # exceeds the limit, so `sent` is pinned just ahead of where the
            # arm already is; the arm can never catch up, and the commanded
            # motion is silently attenuated instead of merely rate-limited.
            # Measured on the real FR5 replaying a UMI demo: the clamp fired
            # on 47% of steps and crushed commanded joint ranges from 105/94/90
            # deg (J4/J6/J5) down to 29/24/41 deg -- the arm tracked what it
            # was sent essentially perfectly, but what it was sent was wrong.
            # The visible symptom was a recorded roll rotation not happening
            # at all. MuJoCo never showed this because MujocoEnvBase.step()
            # passes the action straight to do_simulation() with no clamp.
            #
            # Anchoring on the previous command keeps the real guarantee
            # (successive commands never step more than the limit, so nothing
            # commands a jump) without coupling the command stream to tracking
            # error. Tracking error is still watched, but reported rather than
            # silently compensated -- see the warning below.
            if self._prev_arm_joint_pos_command is None:
                # First command after a reset: anchor on where the arm
                # actually is, so it cannot jump from an unknown state.
                self._prev_arm_joint_pos_command = self.arm_joint_pos_actual.copy()

            # Scale the allowed per-step motion by how much time REALLY
            # elapsed since the previous command, not by the nominal dt.
            # The loop's actual period is not a stable self.dt (policy
            # inference, camera reads and synchronous XML-RPC readback add
            # tens to hundreds of ms, and only on some ticks), so budgeting
            # every step as if it were dt starves the slow ticks: they may
            # advance only dt's worth of motion no matter how long they
            # actually took, the command falls behind, and the next fast tick
            # lurches to catch up -- the staircase/jerk seen on hardware.
            # `duration` itself is left alone because callers also use it as
            # the loop's pacing target (see step()).
            clamp_duration = getattr(self, "_clamp_duration", None)
            if clamp_duration is None:
                clamp_duration = duration
            max_joint_pos_delta = scaled_joint_vel_limit * clamp_duration
            arm_joint_pos_command_overwritten = (
                self._prev_arm_joint_pos_command
                + np.clip(
                    arm_joint_pos_command - self._prev_arm_joint_pos_command,
                    -1 * max_joint_pos_delta,
                    max_joint_pos_delta,
                )
            )
            self._prev_arm_joint_pos_command = arm_joint_pos_command_overwritten.copy()

            # An open-loop command stream can outrun the arm without the
            # clamp noticing, so surface it here instead. This is a real
            # fault condition (a stalled/blocked/faulted arm, or commands
            # simply too fast for it), and it used to be hidden by the
            # measured-anchored clamp quietly throttling the command.
            tracking_error = np.max(
                np.abs(
                    arm_joint_pos_command_overwritten - self.arm_joint_pos_actual
                )
            )
            if tracking_error > self.joint_pos_tracking_error_abort_threshold:
                # Stop transmitting before raising: _motion_enabled gates
                # ServoJ/gripper output (see RealFairino5EnvBase._set_action),
                # so the arm holds its last target instead of continuing to
                # chase a command stream it has already lost.
                self._motion_enabled = False
                raise RuntimeError(
                    f"[{self.__class__.__name__}] ABORT: commanded joint "
                    f"position is {np.rad2deg(tracking_error):.1f} deg ahead of "
                    f"the measured position (limit "
                    f"{np.rad2deg(self.joint_pos_tracking_error_abort_threshold):.0f} "
                    "deg). Motion has been disabled and the arm is holding "
                    "position. Reduce joint_vel_limit_scale before retrying."
                )
            if tracking_error > self.joint_pos_tracking_error_warn_threshold:
                print(
                    f"[{self.__class__.__name__}] WARNING: commanded joint "
                    f"position is {np.rad2deg(tracking_error):.1f} deg ahead of "
                    "the measured position -- the arm is not keeping up with "
                    "the command stream (slow it down, e.g. "
                    "ReplayUmiOnFairino5.py's --time_scale)."
                )

            action[arm_joint_idxes] = arm_joint_pos_command_overwritten

        if not np.all(np.isfinite(action)):
            raise RuntimeError(
                f"[{self.__class__.__name__}] Action contains NaN or Inf: {action}"
            )

        if duration is None or not np.isfinite(duration):
            raise RuntimeError(
                f"[{self.__class__.__name__}] Duration is NaN or Inf: {duration}"
            )

        return action, duration

    @abstractmethod
    def _get_obs(self):
        pass

    def _get_info(self):
        info = {}

        if (
            len(self.camera_names)
            + len(self.pointcloud_camera_names)
            + len(self.rgb_tactile_names)
            + len(self.intensity_tactile_names)
            + len(self.rgb_camera_names)
            == 0
        ):
            return info

        info["rgb_images"] = {}
        info["depth_images"] = {}

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}

            for camera_name, camera in self.cameras.items():
                futures[executor.submit(self.get_camera_data, camera_name, camera)] = (
                    camera_name
                )

            for (
                pointcloud_camera_name,
                pointcloud_camera,
            ) in self.pointcloud_cameras.items():
                futures[
                    executor.submit(
                        self.get_pointcloud_camera_data,
                        pointcloud_camera_name,
                        pointcloud_camera,
                    )
                ] = pointcloud_camera_name

            for rgb_tactile_name, rgb_tactile in self.rgb_tactiles.items():
                futures[
                    executor.submit(
                        self.get_rgb_tactile_data, rgb_tactile_name, rgb_tactile
                    )
                ] = rgb_tactile_name

            for (
                intensity_tactile_name,
                intensity_tactile,
            ) in self.intensity_tactiles.items():
                futures[
                    executor.submit(
                        self.get_intensity_tactile_data,
                        intensity_tactile_name,
                        intensity_tactile,
                    )
                ] = intensity_tactile_name

            for rgb_camera_name, rgb_camera in self.rgb_cameras.items():
                futures[
                    executor.submit(
                        self.get_rgb_camera_data, rgb_camera_name, rgb_camera
                    )
                ] = rgb_camera_name

            for future in concurrent.futures.as_completed(futures):
                name, result = future.result()
                for key, value in result.items():
                    if value is None:
                        continue
                    if key not in info:
                        info[key] = {}
                    info[key][name] = value

        return info

    def get_camera_data(self, camera_name, camera):
        rgb_image, depth_image = camera.read((640, 480))
        depth_image = (1e-3 * depth_image[:, :, 0]).astype(np.float32)  # [m]
        return camera_name, {"rgb_images": rgb_image, "depth_images": depth_image}

    # Number of consecutive frame timeouts (each RECONNECT_TIMEOUT_SEC long) before
    # a pointcloud camera's pipeline is automatically restarted.
    RECONNECT_FAILURE_THRESHOLD = 3
    RECONNECT_TIMEOUT_SEC = 2.0

    def get_pointcloud_camera_data(self, pointcloud_camera_name, pointcloud_camera):
        from pyorbbecsdk import OBFormat

        # queue.get() blocks forever if the camera's frame callback never fires (e.g.
        # a flaky USB link -- "Failed to query USB device interface name" at
        # enumeration time is a symptom of this). That hang would happen inside a
        # ThreadPoolExecutor worker in _get_info(), and the executor's context
        # manager waits for all workers on exit, so even Ctrl+C couldn't kill the
        # process. Time out instead of blocking forever: on repeated timeouts,
        # attempt to restart the camera's pipeline, and meanwhile fall back to the
        # last successfully received frame so a flaky camera doesn't crash or stall
        # data collection.
        #
        # Do NOT block waiting for a *new* frame when a previous one is
        # already in hand. The control loop calls _get_info() every step, so
        # blocking here pins the whole loop to the camera's frame rate: with
        # this rig's Orbbec running at 10 FPS, every env.step() waited ~100ms
        # for a frame, the loop ran at exactly 10 Hz (measured dt median
        # 0.100/0.101s across runs -- an externally clocked giveaway), and
        # each ServoJ command therefore had to cover ~10x more motion than at
        # the nominal 50 Hz. Large position steps at a low command rate are
        # exactly the visible "joint angle jumps"/staircase motion. The
        # policy only consumes an image every `skip` steps anyway, and a
        # frame that is a few ms stale is harmless, so reuse the most recent
        # frame and let the control loop run as fast as the robot I/O allows.
        frames = None
        try:
            # Drain to the NEWEST queued frame rather than taking the oldest:
            # femtobolt_callback keeps up to 5, and consuming them one per
            # step would feed the policy progressively staler images.
            while True:
                frames = pointcloud_camera["queue"].get_nowait()
        except Empty:
            pass

        now = time.time()
        if frames is not None:
            pointcloud_camera["last_frame_time"] = now
        elif pointcloud_camera["last_result"] is not None:
            # No new frame yet, but we have a recent one: this is the normal
            # case when the loop outruns the camera. Only treat it as a fault
            # once nothing has arrived for RECONNECT_TIMEOUT_SEC.
            last_frame_time = pointcloud_camera.get("last_frame_time")
            if (
                last_frame_time is not None
                and now - last_frame_time <= self.RECONNECT_TIMEOUT_SEC
            ):
                return pointcloud_camera_name, pointcloud_camera["last_result"]

        if frames is None:
            # Either no frame has ever arrived (startup -- wait for the first
            # one), or the camera has gone quiet for longer than the timeout.
            if pointcloud_camera["last_result"] is None:
                try:
                    frames = pointcloud_camera["queue"].get(
                        timeout=self.RECONNECT_TIMEOUT_SEC
                    )
                    pointcloud_camera["last_frame_time"] = time.time()
                except Empty:
                    frames = None

        if frames is None:
            pointcloud_camera["consecutive_failures"] += 1
            n = pointcloud_camera["consecutive_failures"]
            print(
                f"[{self.__class__.__name__}] No frame from pointcloud camera "
                f"'{pointcloud_camera_name}' for {self.RECONNECT_TIMEOUT_SEC}s "
                f"(consecutive misses: {n})."
            )
            if n >= self.RECONNECT_FAILURE_THRESHOLD and not pointcloud_camera.get(
                "reconnecting", False
            ):
                # A flaky USB link can make Pipeline(device)/pipeline.start() hang
                # in native code with no timeout of their own. Run the reconnect
                # attempt in its own daemon thread rather than inline here: this
                # method runs inside a ThreadPoolExecutor worker (see _get_info()),
                # and that executor's "with" block waits for all workers to finish
                # on exit -- so a stuck reconnect call would hang the whole process
                # unkillably, same as the original queue.get() with no timeout. A
                # daemon thread can be safely abandoned if it never returns.
                pointcloud_camera["reconnecting"] = True
                print(
                    f"[{self.__class__.__name__}] Restarting pointcloud camera "
                    f"'{pointcloud_camera_name}' after {n} consecutive misses "
                    "(in background)."
                )
                threading.Thread(
                    target=self._reconnect_femtobolt_pipeline,
                    args=(pointcloud_camera_name,),
                    daemon=True,
                ).start()
            if pointcloud_camera["last_result"] is not None:
                return pointcloud_camera_name, pointcloud_camera["last_result"]
            return pointcloud_camera_name, {}

        # A frame arrived, so the camera link is alive; reset the failure streak.
        pointcloud_camera["consecutive_failures"] = 0

        rgb_frame = frames.get_color_frame()
        if rgb_frame is None:
            # A frame set with no color sub-frame this tick isn't itself a dropped
            # connection; fall back to the last good frame instead of returning {},
            # which would make TeleopBase.draw_image() KeyError on this camera.
            if pointcloud_camera["last_result"] is not None:
                return pointcloud_camera_name, pointcloud_camera["last_result"]
            return pointcloud_camera_name, {}
        rgb_width = rgb_frame.get_width()
        rgb_height = rgb_frame.get_height()
        rgb_format = rgb_frame.get_format()
        rgb_data = np.asanyarray(rgb_frame.get_data())
        if rgb_format == OBFormat.RGB:
            rgb_image = np.resize(rgb_data, (rgb_height, rgb_width, 3))
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        elif rgb_format == OBFormat.BGR:
            rgb_image = np.resize(rgb_data, (rgb_height, rgb_width, 3))
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        elif rgb_format == OBFormat.YUYV:
            rgb_image = np.resize(rgb_data, (rgb_height, rgb_width, 2))
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_YUV2BGR_YUYV)
        elif rgb_format == OBFormat.MJPG:
            rgb_image = cv2.imdecode(rgb_data, cv2.IMREAD_COLOR)
        else:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Unsupported rgb format in pointcloud camera: {rgb_format}"
            )
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        color_target_size = pointcloud_camera.get("color_target_size", (640, 480))
        rgb_image = cv2.resize(rgb_image, color_target_size)

        # The depth stream is enabled in _start_femtobolt_pipeline(). Convert the
        # raw Y16 depth (device-native units, typically mm) to meters, matching
        # the convention used for RealSense depth elsewhere in this file (see
        # get_realsense_data()). Falls back to an all-zero placeholder if this
        # tick's frame set has no depth sub-frame, so a momentary miss doesn't
        # break TeleopBase.draw_image(), which expects a "depth_images" entry for
        # every camera not in rgb_tactile_names.
        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            depth_image = np.zeros(rgb_image.shape[:2], dtype=np.float32)
        else:
            depth_width = depth_frame.get_width()
            depth_height = depth_frame.get_height()
            depth_scale = depth_frame.get_depth_scale()
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
            depth_data = depth_data.reshape((depth_height, depth_width))
            depth_image = 1e-3 * depth_scale * depth_data.astype(np.float32)  # [m]
            depth_image = cv2.resize(
                depth_image, color_target_size, interpolation=cv2.INTER_NEAREST
            )

        result = {
            "rgb_images": rgb_image,
            "depth_images": depth_image,
        }
        pointcloud_camera["last_result"] = result
        return pointcloud_camera_name, result

    def get_rgb_tactile_data(self, rgb_tactile_name, rgb_tactile):
        ret, rgb_image = rgb_tactile.read()
        if not ret:
            raise RuntimeError(
                f"[{self.__class__.__name__}] Failed to read tactile image."
            )
        image_size = (640, 480)
        rgb_image = cv2.resize(rgb_image, image_size)
        return rgb_tactile_name, {"rgb_images": rgb_image}

    def get_intensity_tactile_data(self, intensity_tactile_name, intensity_tactile):
        intensity_tactile_device = intensity_tactile["device"]
        intensity_tactile_buf = intensity_tactile["buf"]
        key_binaries = intensity_tactile_device.read(9, timeout=3)

        if len(key_binaries) == 0:
            return intensity_tactile_name, {
                "intensity_tactile": intensity_tactile_buf.copy()
            }

        key_name_map = {
            0x69: "F17",
            0x6A: "F18",
            0x6B: "F19",
            0x6C: "F14",
            0x6D: "F15",
            0x6E: "F16",
        }
        key_idx_map = {
            "F14": 0,
            "F15": 1,
            "F16": 2,
            "F17": 3,
            "F18": 4,
            "F19": 5,
        }

        intensity_tactile_value = np.zeros(shape=(2, 3), dtype=np.uint8)
        for key_binary in key_binaries[3:]:
            if key_binary in key_name_map:
                key_name = key_name_map[key_binary]
                key_idx = key_idx_map[key_name]
                intensity_tactile_value[int(key_idx // 3)][int(key_idx % 3)] = 1

        intensity_tactile_buf[...] = intensity_tactile_value
        return intensity_tactile_name, {"intensity_tactile": intensity_tactile_value}

    def get_joint_pos_from_obs(self, obs):
        """Get joint position from observation."""
        return obs["joint_pos"]

    def get_joint_vel_from_obs(self, obs):
        """Get joint velocity from observation."""
        return obs["joint_vel"]

    def get_gripper_joint_pos_from_obs(self, obs):
        """Get gripper joint position from observation."""
        joint_pos = self.get_joint_pos_from_obs(obs)
        gripper_joint_pos = np.zeros(
            DataKey.get_dim(DataKey.COMMAND_GRIPPER_JOINT_POS, self)
        )

        for body_config in self.body_config_list:
            if not isinstance(body_config, ArmConfig):
                continue

            gripper_joint_pos[body_config.gripper_joint_idxes_in_gripper_joint_pos] = (
                joint_pos[body_config.gripper_joint_idxes]
            )

        return gripper_joint_pos

    def get_eef_wrench_from_obs(self, obs):
        """Get end-effector wrench (fx, fy, fz, nx, ny, nz) from observation."""
        return obs["wrench"]

    def get_weight_from_obs(self, obs):
        """Get measured weight [g] from an external scale (e.g. M5Stack) from observation."""
        return obs["weight"]

    def get_time(self):
        """Get real-world time. [s]"""
        return time.time() - self.init_time

    @property
    def camera_names(self):
        """Get camera names."""
        return list(self.cameras.keys())

    @property
    def pointcloud_camera_names(self):
        """Get pointcloud camera names."""
        return list(self.pointcloud_cameras.keys())

    @property
    def rgb_tactile_names(self):
        """Get names of tactile sensors with RGB output."""
        return list(self.rgb_tactiles.keys())

    @property
    def intensity_tactile_names(self):
        """Get names of tactile sensors with intensity output."""
        return list(self.intensity_tactiles.keys())

    def get_camera_fovy(self, camera_name):
        """Get vertical field-of-view of the camera."""
        return self.cameras[camera_name].depth_fovy

    def modify_world(self, world_idx=None, cumulative_idx=None):
        """Modify simulation world depending on world index."""
        raise NotImplementedError(
            f"[{self.__class__.__name__}] modify_world is not implemented."
        )

    def draw_box_marker(self, pos, mat, size, rgba):
        """Draw box marker."""
        # In a real-world environment, it is not possible to programmatically draw markers
        pass
