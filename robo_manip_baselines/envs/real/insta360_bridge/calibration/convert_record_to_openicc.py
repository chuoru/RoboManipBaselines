"""Convert an insta360_bridge --record output (<prefix>.avi,
<prefix>_frame_timestamps.csv, <prefix>.csv) into one clip folder of the
dataset layout OpenImuCameraCalibrator's (OpenICC) smartphone calibration
path expects (python/run_smartphone_calibration.py, see
third_party/OpenImuCameraCalibrator/docs/samsung_s20_calibration.md) --
i.e. the Pilotguru telemetry format OpenICC's
python/telemetry_converter.py::read_pilotguru_telemetry() parses. This is
the Insta360 analogue of that doc's Samsung S20 + Pilotguru-app workflow:
Insta360 has no GoPro-style GPMF telemetry muxed into its video, but
insta360_bridge's own --record mode already gives us the same raw
ingredients (separate video + IMU csv) Pilotguru would have produced.

Usage: python3 convert_record_to_openicc.py <record_prefix> <out_dir> \\
           --role {cam,imu_bias,cam_imu}
  <record_prefix>.avi                    -- cropped/masked SLAM-input frames
  <record_prefix>_frame_timestamps.csv   -- frame_index,timestamp_ms
  <record_prefix>.csv                    -- long-format IMU (see below)
Writes <out_dir>/<role>/{video.mp4,frames.json,accelerations.json,rotations.json}.

Run this 3 times (once per --record session, with the matching --role) to
assemble one OpenICCDataset/{cam,imu_bias,cam_imu}/ tree:
  cam:      slowly circle the calibration board for ~20-30s (intrinsics)
  imu_bias: camera stationary for ~10-20s (IMU bias)
  cam_imu:  fast motion exciting all 6 DOF, avoiding motion blur (extrinsics
            + time offset)
See envs/real/insta360_bridge/README.md's Calibration section.

Only "time_usec" is read from frames.json entries (used by OpenICC to derive
camera fps and the IMU-to-camera t0 offset -- confirmed against
read_pilotguru_telemetry()), and OpenICC's loader never reads
GPS/pressure/CAN fields Pilotguru's real app would also produce, so those
are omitted entirely here rather than faked.

video.mp4 is a container remux (ffmpeg -c copy) of the bridge's .avi, NOT a
re-encode -- OpenICC's own calibration (ChArUco/AprilGrid corner detection,
Ceres bundle adjustment) works directly on whatever codec the bridge wrote;
this just needs an .mp4 container for OpenICC's video reader.
"""

import argparse
import csv
import json
import os

import ffmpeg


def convert_frames(record_prefix, role_dir):
    frames = []
    with open(f"{record_prefix}_frame_timestamps.csv") as f:
        for row in csv.DictReader(f):
            frames.append({"time_usec": round(float(row["timestamp_ms"]) * 1e3)})
    with open(os.path.join(role_dir, "frames.json"), "w") as f:
        json.dump({"frames": frames}, f)
    print(f"[convert_record_to_openicc] frames.json: {len(frames)} entries")
    return len(frames)


def convert_imu(record_prefix, role_dir):
    # Matches convert_record_to_euroc.py's IMU-csv handling: long-format
    # "timestamp_ms,sensor_type,x,y,z", one gyro row then one accel row per
    # sample sharing the same timestamp. No unit conversion needed -- gyro
    # rad/s and accel m/s^2 are already what read_pilotguru_telemetry()
    # expects (it passes x/y/z straight through into IMU calibration).
    accelerations = []
    rotations = []
    with open(f"{record_prefix}.csv") as f:
        for row in csv.DictReader(f):
            entry = {
                "time_usec": round(float(row["timestamp_ms"]) * 1e3),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
            }
            if row["sensor_type"] == "accel":
                accelerations.append(entry)
            elif row["sensor_type"] == "gyro":
                rotations.append(entry)

    with open(os.path.join(role_dir, "accelerations.json"), "w") as f:
        json.dump({"accelerations": accelerations}, f)
    with open(os.path.join(role_dir, "rotations.json"), "w") as f:
        json.dump({"rotations": rotations}, f)
    print(
        f"[convert_record_to_openicc] accelerations.json: {len(accelerations)} "
        f"entries, rotations.json: {len(rotations)} entries"
    )


def convert_video(record_prefix, role_dir):
    video_path = os.path.join(role_dir, "video.mp4")
    ffmpeg.input(f"{record_prefix}.avi").output(video_path, c="copy").run(
        overwrite_output=True, quiet=True
    )
    print(f"[convert_record_to_openicc] video.mp4 written to {video_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record_prefix", help="insta360_bridge --record <prefix>")
    parser.add_argument(
        "out_dir", help="OpenICCDataset root (role subfolder created under it)"
    )
    parser.add_argument(
        "--role", required=True, choices=["cam", "imu_bias", "cam_imu"]
    )
    args = parser.parse_args()

    role_dir = os.path.join(args.out_dir, args.role)
    os.makedirs(role_dir, exist_ok=True)

    convert_video(args.record_prefix, role_dir)
    convert_frames(args.record_prefix, role_dir)
    convert_imu(args.record_prefix, role_dir)

    print(f"[convert_record_to_openicc] Done: {role_dir}")


if __name__ == "__main__":
    main()
