import argparse
import csv
import glob
import os

import ffmpeg
import telemetry_parser


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="directory containing *.insv files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="directory to write *.mp4/*.csv files (default: same as dataset_dir)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="regenerate outputs even if they already exist",
    )

    return parser.parse_args()


class ConvertInsvToMp4AndImu:
    IMU_CSV_HEADER = ["timestamp_ms", "sensor_type", "x", "y", "z"]

    def __init__(self, dataset_dir, output_dir, overwrite):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir if output_dir is not None else dataset_dir
        self.overwrite = overwrite

    def run(self):
        insv_path_list = sorted(glob.glob(os.path.join(self.dataset_dir, "*.insv")))
        print(
            f"[{self.__class__.__name__}] Found {len(insv_path_list)} .insv file(s) "
            f"under {self.dataset_dir}"
        )
        if not insv_path_list:
            return False

        os.makedirs(self.output_dir, exist_ok=True)

        video_ok_count = 0
        video_fail_count = 0
        imu_ok_count = 0
        imu_fail_count = 0

        for insv_path in insv_path_list:
            basename = os.path.splitext(os.path.basename(insv_path))[0]
            mp4_path = os.path.join(self.output_dir, f"{basename}.mp4")
            csv_path = os.path.join(self.output_dir, f"{basename}.csv")

            if self._convert_video(insv_path, mp4_path):
                video_ok_count += 1
            else:
                video_fail_count += 1

            if self._convert_imu(insv_path, csv_path):
                imu_ok_count += 1
            else:
                imu_fail_count += 1

        print(
            f"[{self.__class__.__name__}] Video: {video_ok_count} converted, "
            f"{video_fail_count} failed. IMU: {imu_ok_count} converted, "
            f"{imu_fail_count} failed."
        )

        return video_fail_count == 0 and imu_fail_count == 0

    def _convert_video(self, insv_path, mp4_path):
        if os.path.exists(mp4_path) and not self.overwrite:
            print(f"[{self.__class__.__name__}] Skip existing {mp4_path}")
            return True

        print(f"[{self.__class__.__name__}] Converting video {insv_path} -> {mp4_path}")
        try:
            ffmpeg.input(insv_path).output(mp4_path, c="copy").run(
                overwrite_output=True, quiet=True
            )
        except ffmpeg.Error as e:
            stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
            print(
                f"[{self.__class__.__name__}] Failed to convert video {insv_path}: "
                f"{stderr.strip().splitlines()[-1] if stderr.strip() else e}"
            )
            return False
        return True

    def _convert_imu(self, insv_path, csv_path):
        if os.path.exists(csv_path) and not self.overwrite:
            print(f"[{self.__class__.__name__}] Skip existing {csv_path}")
            return True

        print(f"[{self.__class__.__name__}] Extracting IMU {insv_path} -> {csv_path}")
        try:
            imu_samples = telemetry_parser.Parser(insv_path).normalized_imu()
        except Exception as e:
            print(f"[{self.__class__.__name__}] Failed to extract IMU from {insv_path}: {e}")
            return False

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.IMU_CSV_HEADER)
            for sample in imu_samples:
                timestamp_ms = sample["timestamp_ms"]
                if sample.get("gyro") is not None:
                    writer.writerow([timestamp_ms, "gyro", *sample["gyro"]])
                if sample.get("accl") is not None:
                    writer.writerow([timestamp_ms, "accel", *sample["accl"]])
        return True


if __name__ == "__main__":
    convert_insv_to_mp4_and_imu = ConvertInsvToMp4AndImu(**vars(parse_argument()))
    is_ok = convert_insv_to_mp4_and_imu.run()
    if not is_ok:
        exit(1)
