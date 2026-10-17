"""Runs offline SLAM on one teleop episode's Insta360 on-device recording
(saved by teleop/TeleopBase.py's --record_insta360_slam as
<episode>.rmb/insta360_raw.insv) and writes the resulting camera trajectory,
transformed into the robot's EEF frame, into that episode's main.rmb.hdf5
under DataKey.MEASURED_CAMERA_EEF_POSE -- an independent, SLAM-derived
alternative to the robot-FK-derived DataKey.MEASURED_EEF_POSE already there.

Pipeline (see envs/real/insta360_bridge/orb_slam3_hardware_fixes/README.md
and teleop/calibrate_insta360_eef_extrinsic.py for the pieces this chains
together):
  1. insv -> mp4 + IMU csv, via misc/ConvertInsvToMp4AndImu.py's converter
     class (telemetry_parser extracts the Insta360's embedded telemetry --
     same role GoPro's GPMF plays in UMI's own GoPro pipeline).
  2. mp4 + IMU csv -> camera trajectory, via ORB_SLAM3's ported
     mono_inertial_video executable (see
     envs/real/insta360_bridge/orb_slam3_hardware_fixes/mono_inertial_video.cc),
     run against Insta360X3_2880.yaml (calibrated for this native,
     un-cropped on-device recording resolution -- NOT Insta360_X4.yaml,
     which is calibrated for the live bridge's already-cropped/masked
     800x800 SLAM-input frames).
  3. Camera trajectory (SaveTrajectoryEuRoC format: "timestamp_ns tx ty tz
     qx qy qz qw" per line, timestamp relative to the recording's own
     start) -> EEF-frame trajectory, via the calibrated
     insta360_to_eef_frame_rotation/translation (teleop/configs/
     Insta360UMI.yaml, from calibrate_insta360_eef_extrinsic.py).
  4. Realigned onto the episode's own DataKey.TIME axis (wall-clock
     relative to env.unwrapped.init_time) using the
     insta360_slam_recording_offset_sec meta_data attribute
     TeleopBase.TeleopPhase.start()/save_data() recorded for this episode
     -- see those methods' comments for why this offset is needed (the
     trajectory's own timestamps are relative to a DIFFERENT clock origin:
     the on-device recording's start, not the episode's).

Usage:
    python ./misc/insta360_slam_pipeline/AddSlamPoseToRmb.py \\
        ./dataset/MyDemo_<date>/MyDemo_world0_000.rmb \\
        --vocab third_party/ORB_SLAM3/Vocabulary/ORBvoc.txt \\
        --settings envs/real/insta360_bridge/Insta360X3_2880.yaml \\
        --mono_inertial_video_bin third_party/ORB_SLAM3/Examples/Monocular-Inertial/mono_inertial_video \\
        --insta360_config teleop/configs/Insta360UMI.yaml
"""

import argparse
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from ConvertInsvToMp4AndImu import ConvertInsvToMp4AndImu  # noqa: E402

from robo_manip_baselines.common import DataKey, RmbData  # noqa: E402


def find_raw_recording(episode_dir):
    candidates = glob.glob(os.path.join(episode_dir, "insta360_raw.*"))
    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No insta360_raw.* file found in {episode_dir} -- this episode "
            "was not recorded with --record_insta360_slam, or the file was "
            "already consumed/moved."
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple insta360_raw.* files found in {episode_dir}: "
            f"{candidates} -- expected exactly one."
        )
    return candidates[0]


def convert_insv_to_mp4_and_imu(insv_path, work_dir):
    converter = ConvertInsvToMp4AndImu(
        dataset_dir=os.path.dirname(insv_path), output_dir=work_dir, overwrite=True
    )
    # Reuse its own per-file methods directly rather than .run() (which globs
    # *.insv in dataset_dir -- fine here since find_raw_recording() already
    # guaranteed exactly one, but calling the per-file methods directly
    # avoids relying on that glob picking the same file a second time).
    basename = os.path.splitext(os.path.basename(insv_path))[0]
    mp4_path = os.path.join(work_dir, f"{basename}.mp4")
    csv_path = os.path.join(work_dir, f"{basename}.csv")
    if not converter._convert_video(insv_path, mp4_path):
        raise RuntimeError(f"Failed to convert {insv_path} to mp4.")
    if not converter._convert_imu(insv_path, csv_path):
        raise RuntimeError(f"Failed to extract IMU telemetry from {insv_path}.")
    return mp4_path, csv_path


def run_mono_inertial_video(vocab, settings, mp4_path, imu_csv_path, mono_inertial_video_bin, work_dir):
    out_base = os.path.join(work_dir, "slam_out")
    subprocess.run(
        [mono_inertial_video_bin, vocab, settings, mp4_path, imu_csv_path, out_base],
        check=True,
    )
    trajectory_path = f"{out_base}_CameraTrajectory.txt"
    if not os.path.exists(trajectory_path):
        raise RuntimeError(
            f"{mono_inertial_video_bin} did not produce {trajectory_path} -- "
            "SLAM likely never initialized a map for this recording (check "
            "its stdout above for tracking/initialization failures)."
        )
    return trajectory_path


def load_camera_trajectory(trajectory_path):
    """Returns (timestamps_sec, positions (N,3), quats_xyzw (N,4)), relative
    to the recording's own start -- see this script's docstring for the
    SaveTrajectoryEuRoC line format."""
    rows = np.loadtxt(trajectory_path)
    if rows.ndim == 1:
        rows = rows[None, :]
    timestamps_sec = rows[:, 0] * 1e-9
    positions = rows[:, 1:4]
    quats_xyzw = rows[:, 4:8]
    return timestamps_sec, positions, quats_xyzw


def camera_trajectory_to_eef_trajectory(positions, quats_xyzw, insta360_to_eef_frame_rotation, insta360_to_eef_translation):
    """world_T_eef = world_T_cam @ cam_T_gripper, where cam_T_gripper is
    derived from the calibrated insta360_to_eef_frame_rotation/translation
    -- see teleop/calibrate_insta360_eef_extrinsic.py's docstring for the
    full derivation (this is the same relationship
    Insta360InputDevice.set_command_data's tcp_translation computation
    uses, generalized here to the full rotation too, not just translation
    deltas)."""
    cam_rotations = Rotation.from_quat(quats_xyzw)
    r = insta360_to_eef_frame_rotation
    t = insta360_to_eef_translation
    cam_t_gripper_rotation = r.T
    cam_t_gripper_translation = -r.T @ t

    eef_rotations = cam_rotations * Rotation.from_matrix(cam_t_gripper_rotation)
    eef_positions = (
        cam_rotations.apply(cam_t_gripper_translation) + positions
    )
    return eef_positions, eef_rotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dir", help="one episode's .rmb directory")
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--mono_inertial_video_bin", required=True)
    parser.add_argument("--insta360_config", required=True)
    args = parser.parse_args()

    with open(args.insta360_config, "r") as f:
        insta360_config = yaml.safe_load(f)
    insta360_to_eef_frame_rotation = np.array(
        insta360_config["insta360_to_eef_frame_rotation"], dtype=np.float64
    )
    insta360_to_eef_translation = np.array(
        insta360_config["insta360_to_eef_translation"], dtype=np.float64
    )

    insv_path = find_raw_recording(args.episode_dir)
    print(f"[AddSlamPoseToRmb] Raw recording: {insv_path}")

    with tempfile.TemporaryDirectory(prefix="insta360_slam_") as work_dir:
        mp4_path, imu_csv_path = convert_insv_to_mp4_and_imu(insv_path, work_dir)
        trajectory_path = run_mono_inertial_video(
            args.vocab, args.settings, mp4_path, imu_csv_path,
            args.mono_inertial_video_bin, work_dir,
        )
        traj_t_sec, traj_pos, traj_quat_xyzw = load_camera_trajectory(trajectory_path)

    print(f"[AddSlamPoseToRmb] SLAM trajectory: {len(traj_t_sec)} poses over "
          f"{traj_t_sec[-1] - traj_t_sec[0]:.1f}s (recording duration unknown "
          "without the source video's own length -- compare against the "
          "episode duration below to sanity-check coverage).")

    eef_pos, eef_rot = camera_trajectory_to_eef_trajectory(
        traj_pos, traj_quat_xyzw, insta360_to_eef_frame_rotation, insta360_to_eef_translation
    )

    with RmbData(args.episode_dir, mode="a") as rmb_data:
        if "insta360_slam_recording_offset_sec" not in rmb_data.attrs:
            raise RuntimeError(
                f"{args.episode_dir} has no 'insta360_slam_recording_offset_sec' "
                "attribute -- it was not recorded with --record_insta360_slam "
                "(or predates that feature)."
            )
        recording_offset_sec = rmb_data.attrs["insta360_slam_recording_offset_sec"]
        episode_time_sec = rmb_data[DataKey.TIME][:]

        # Realign the trajectory's own clock (relative to the recording's
        # start) onto the episode's clock (relative to env.unwrapped.init_time)
        # -- see this script's docstring and TeleopPhase.start()'s comment.
        traj_episode_time_sec = traj_t_sec + recording_offset_sec

        t_min, t_max = traj_episode_time_sec[0], traj_episode_time_sec[-1]
        coverage = np.clip(episode_time_sec, t_min, t_max)
        out_of_range_frac = np.mean(
            (episode_time_sec < t_min) | (episode_time_sec > t_max)
        )
        if out_of_range_frac > 0.0:
            print(
                f"[AddSlamPoseToRmb] WARNING: {out_of_range_frac * 100:.1f}% of "
                "this episode's DataKey.TIME steps fall outside the SLAM "
                "trajectory's covered time range -- those steps are clamped "
                "to the trajectory's first/last pose rather than extrapolated. "
                "This usually means tracking was lost near the start/end of "
                "the recording, or insta360_slam_recording_offset_sec is "
                "wrong."
            )

        pos_interp = np.stack(
            [np.interp(coverage, traj_episode_time_sec, eef_pos[:, i]) for i in range(3)],
            axis=1,
        )
        slerp = Slerp(traj_episode_time_sec, eef_rot)
        rot_interp = slerp(coverage)
        quat_wxyz_interp = rot_interp.as_quat()[:, [3, 0, 1, 2]]

        measured_camera_eef_pose = np.concatenate(
            [pos_interp, quat_wxyz_interp], axis=1
        ).astype(np.float32)

        if DataKey.MEASURED_CAMERA_EEF_POSE in rmb_data.h5file:
            del rmb_data.h5file[DataKey.MEASURED_CAMERA_EEF_POSE]
        rmb_data.h5file.create_dataset(
            DataKey.MEASURED_CAMERA_EEF_POSE, data=measured_camera_eef_pose
        )

    print(
        f"[AddSlamPoseToRmb] Wrote {DataKey.MEASURED_CAMERA_EEF_POSE} "
        f"({measured_camera_eef_pose.shape}) to {args.episode_dir}"
    )


if __name__ == "__main__":
    main()
