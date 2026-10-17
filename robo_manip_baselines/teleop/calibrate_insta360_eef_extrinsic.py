"""Calibrate Insta360InputDevice's insta360_to_eef_frame_rotation and
insta360_to_eef_translation (teleop/configs/Insta360UMI.yaml) -- currently a
PLACEHOLDER identity in that file (see its own comment). Unlike
calibrate_vive_rotation.py/calibrate_vive_world_frame.py (rotation-only, no
robot connection needed), this calibrates the FULL SE(3) camera-to-EEF
extrinsic, because the Insta360 camera's SLAM pose is its own 6-DoF
measurement (not a lighthouse-anchored tracker needing a separate world-frame
calibration), and because the resulting extrinsic also has to serve
misc/insta360_slam_pipeline/AddSlamPoseToRmb.py (transforming an offline SLAM
camera trajectory into the robot's EEF frame), where a translation-only or
rotation-only approximation would leave a real lever-arm error uncorrected.

Solves the classic AX=ZB "Robot-World/Hand-Eye" calibration problem (M.
Shah's Kronecker-product method, cv2.calibrateRobotWorldHandEye) from paired
samples of:
  - the robot's own FK EEF pose (base_T_gripper), read live via MotionManager
    the same way RolloutBase.run() does on real hardware
    (sync_arm_to_measured then arm_manager.current_se3)
  - the Insta360 camera's SLAM pose (world_T_cam) in ORB-SLAM3's own
    (arbitrary, drifting-reference-free once initialized) world frame, read
    live from insta360_bridge over the same socket Insta360InputDevice uses

No external calibration board/target is needed -- the camera's own SLAM
pose already IS the "world frame" measurement cv2.calibrateRobotWorldHandEye
expects (see its docstring: normally that comes from a fixed board observed
by the camera; here the camera observes its own SLAM map instead, which is
equally fixed/world-anchored once initialized).

SAFETY: this script runs the env in --observe_only mode (see
RealFairino5EnvBase's observe_only) -- it NEVER transmits a motion command,
so the arm must be moved between samples some other way (the robot's own
free-drive/teach mode, or a teach pendant) while this script stays purely
passive. UNVERIFIED ON REAL HARDWARE: whether --observe_only still returns
live, correct measured_joint_pos while the arm is being manually moved (it
should, if the robot connection's read path doesn't depend on this process
having sent any command) -- confirm this before trusting the collected
samples.

Usage:
    python ./teleop/calibrate_insta360_eef_extrinsic.py \\
        --robot_ip 192.168.58.2 --insta360_socket_path /tmp/insta360_hand.sock \\
        --num_samples 15

At each of --num_samples prompts: physically move the arm (free-drive/teach
pendant) AND the camera together into a new, sufficiently different
orientation (vary all 3 rotation axes across the session, not just
translation -- the Kronecker-product method needs rotational diversity to
be well-conditioned), make sure ORB-SLAM3's tracking_state is "OK" (not
"INIT"/"LOST"), then press Enter.
"""

import argparse
import socket
import threading
import time

import cv2
import gymnasium as gym
import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import MotionManager, read_insta360_message
from robo_manip_baselines.common.body.ArmManager import ArmManager


class Insta360PoseReader:
    """Background reader for the Insta360 bridge's "pose" messages -- same
    framing/filtering as Insta360InputDevice._read_bridge_loop, but
    standalone (no InputDeviceBase/arm_manager dependency) since this script
    only needs to sample poses, not drive teleop."""

    TRACKING_TIMEOUT = 0.5  # [s] -- same role as Insta360InputDevice's.

    def __init__(self, socket_path):
        self._connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._connection.connect(socket_path)
        self._lock = threading.Lock()
        self._latest = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        while not self._stop_event.is_set():
            try:
                message = read_insta360_message(self._connection)
            except Exception as e:
                if not self._stop_event.is_set():
                    print(f"[calibrate] Insta360 bridge connection error: {e}")
                break
            if message["type"] != "pose":
                continue
            with self._lock:
                self._latest = message

    def read_se3(self):
        """Returns the latest world_T_cam as a pin.SE3, or None if no
        recent OK-tracking pose is available."""
        with self._lock:
            message = self._latest
        if message is None:
            return None
        if (time.time() - message["t"]) > self.TRACKING_TIMEOUT:
            return None
        if message["tracking_state"] != "OK":
            return None
        pos = np.array(message["pos"], dtype=np.float64)
        rot = pin.Quaternion(*message["quat"]).toRotationMatrix()
        return pin.SE3(rot, pos)

    def close(self):
        self._stop_event.set()
        try:
            self._connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._thread.join(timeout=1.0)
        self._connection.close()


def capture_sample(env, motion_manager, arm_manager, pose_reader, label, sample_sec, sample_hz):
    """Averages many (FK, SLAM) pairs over sample_sec while the operator
    holds the rig still, instead of trusting one instantaneous reading --
    same rationale as calibrate_vive_rotation.py's capture_rotation(). FK
    and SLAM pose are each averaged independently via the same log/exp
    rotation-averaging scheme used elsewhere in this codebase (e.g.
    ArmManager.inverse_kinematics)."""
    input(f"[calibrate] {label} Press Enter when ready and holding both the "
          "arm and camera steady.")
    print(f"[calibrate]   Sampling for {sample_sec}s -- keep holding still...")

    fk_poses = []
    cam_poses = []
    obs, _ = env.reset()
    start = time.time()
    while time.time() - start < sample_sec:
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, _, _, _, _ = env.step(action)
        motion_manager.sync_arm_to_measured(obs)
        fk_poses.append(arm_manager.current_se3.copy())

        cam_se3 = pose_reader.read_se3()
        if cam_se3 is not None:
            cam_poses.append(cam_se3)

        time.sleep(1.0 / sample_hz)

    if len(cam_poses) == 0:
        raise RuntimeError(
            "[calibrate] No OK-tracking Insta360 pose arrived during this "
            "sample window -- check ORB-SLAM3 tracking_state and retry."
        )

    def average_se3(se3_list):
        r_mean = se3_list[0].rotation
        t_mean = np.mean([s.translation for s in se3_list], axis=0)
        for _ in range(10):
            rotvecs = [pin.log3(r_mean.T @ s.rotation) for s in se3_list]
            mean_rotvec = np.mean(rotvecs, axis=0)
            r_mean = r_mean @ pin.exp3(mean_rotvec)
            if np.linalg.norm(mean_rotvec) < 1e-8:
                break
        return pin.SE3(r_mean, t_mean)

    fk_se3 = average_se3(fk_poses)
    cam_se3 = average_se3(cam_poses)
    print(f"[calibrate]   -> captured ({len(fk_poses)} FK / {len(cam_poses)} "
          "camera samples averaged)\n")
    return fk_se3, cam_se3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot_ip", required=True)
    parser.add_argument("--insta360_socket_path", required=True)
    parser.add_argument(
        "--num_samples", type=int, default=15,
        help="At least 3 required by cv2.calibrateRobotWorldHandEye; many "
        "more (10-20), spread across varied rotations, recommended for a "
        "well-conditioned result.",
    )
    parser.add_argument("--sample_sec", type=float, default=1.0)
    parser.add_argument("--sample_hz", type=float, default=20.0)
    args = parser.parse_args()

    env = gym.make(
        "robo_manip_baselines/RealFairino5DemoEnv-v0",
        robot_ip=args.robot_ip,
        camera_ids=None,
        gelsight_ids=None,
        observe_only=True,
    ).unwrapped
    motion_manager = MotionManager(env)
    arm_manager = next(
        bm for bm in motion_manager.body_manager_list if isinstance(bm, ArmManager)
    )
    pose_reader = Insta360PoseReader(args.insta360_socket_path)

    fk_se3_list = []
    cam_se3_list = []
    try:
        for i in range(args.num_samples):
            fk_se3, cam_se3 = capture_sample(
                env, motion_manager, arm_manager, pose_reader,
                f"Sample {i + 1}/{args.num_samples}:",
                args.sample_sec, args.sample_hz,
            )
            fk_se3_list.append(fk_se3)
            cam_se3_list.append(cam_se3)
    except KeyboardInterrupt:
        print("\n[calibrate] Interrupted by Ctrl+C.")
        if len(fk_se3_list) < 3:
            print("[calibrate] Fewer than 3 samples captured -- nothing to solve.")
            return
        print(f"[calibrate] Proceeding with the {len(fk_se3_list)} samples captured so far.")
    finally:
        pose_reader.close()
        env.close()

    # cv2.calibrateRobotWorldHandEye wants base2gripper (inverse of our FK,
    # which is base_T_gripper) and world2cam (inverse of the SLAM pose,
    # which is world_T_cam) -- see its docstring's A/B/X/Z convention.
    r_base2gripper = [fk.rotation.T for fk in fk_se3_list]
    t_base2gripper = [-fk.rotation.T @ fk.translation for fk in fk_se3_list]
    r_world2cam = [cam.rotation.T for cam in cam_se3_list]
    t_world2cam = [-cam.rotation.T @ cam.translation for cam in cam_se3_list]

    _, _, r_gripper2cam, t_gripper2cam = cv2.calibrateRobotWorldHandEye(
        r_world2cam, t_world2cam, r_base2gripper, t_base2gripper,
    )

    # Insta360InputDevice's insta360_to_eef_frame_rotation/translation are
    # the CAMERA-to-EEF transform (see its docstring: "conjugates a
    # tracker-local rotation delta into the EEF's own local TCP/tool
    # frame", and translation similarly camera-frame -> EEF-frame) -- i.e.
    # the inverse of r_gripper2cam/t_gripper2cam above.
    r_cam2gripper = r_gripper2cam.T
    t_cam2gripper = -r_gripper2cam.T @ t_gripper2cam.reshape(3)

    # Residual check: how well does the fitted extrinsic actually relate
    # each sample's FK delta to its camera delta? (Reusing the Kabsch-style
    # fit-quality reporting pattern from calibrate_vive_rotation.py, but
    # over consecutive-sample relative motion rather than fixed test axes,
    # since there is no fixed desired direction here.)
    print("[calibrate] Per-consecutive-sample-pair rotation residual "
          "(should be small, a few deg at most):")
    for i in range(1, len(fk_se3_list)):
        fk_delta = fk_se3_list[i - 1].rotation.T @ fk_se3_list[i].rotation
        cam_delta = cam_se3_list[i - 1].rotation.T @ cam_se3_list[i].rotation
        predicted_fk_delta = r_cam2gripper @ cam_delta @ r_cam2gripper.T
        residual_deg = np.rad2deg(
            np.linalg.norm(pin.log3(fk_delta.T @ predicted_fk_delta))
        )
        flag = " <-- HIGH" if residual_deg > 5.0 else ""
        print(f"    sample {i - 1}->{i}: {residual_deg:.2f} deg{flag}")

    print(
        "\n[calibrate] Add this under device_params for Insta360InputDevice "
        "in teleop/configs/Insta360UMI.yaml (replacing the PLACEHOLDER "
        "identity values):\n"
    )
    print("insta360_to_eef_frame_rotation:")
    for row in r_cam2gripper:
        print(f"  - [{row[0]:.6f}, {row[1]:.6f}, {row[2]:.6f}]")
    print(f"insta360_to_eef_translation: [{t_cam2gripper[0]:.6f}, "
          f"{t_cam2gripper[1]:.6f}, {t_cam2gripper[2]:.6f}]")


if __name__ == "__main__":
    main()
