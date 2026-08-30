import argparse
import os

import ffmpeg

from robo_manip_baselines.common import DataKey, find_rmb_files


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="path to dataset directory (or a single *.rmb episode)",
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=None,
        help="camera names to check (default: infer from video files present)",
    )

    return parser.parse_args()


class CheckDepthVideos:
    def __init__(self, dataset_dir, camera_names):
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names

    def run(self):
        rmb_path_list = find_rmb_files(self.dataset_dir, dedupe=False)
        print(
            f"[{self.__class__.__name__}] Found {len(rmb_path_list)} episode(s) "
            f"under {self.dataset_dir}"
        )

        bad_entries = []
        ok_count = 0
        checked_count = 0

        for rmb_path in rmb_path_list:
            if not rmb_path.endswith(".rmb"):
                # Single-hdf5 episodes store video data inside the hdf5 file,
                # not as separate mp4s, so there is nothing to probe here.
                continue

            camera_names = self.camera_names
            if camera_names is None:
                camera_names = self._infer_camera_names(rmb_path)

            for camera_name in camera_names:
                for key in (
                    DataKey.get_rgb_image_key(camera_name),
                    DataKey.get_depth_image_key(camera_name),
                ):
                    video_path = os.path.join(rmb_path, f"{key}.rmb.mp4")
                    if not os.path.exists(video_path):
                        continue

                    checked_count += 1
                    error = self._probe(video_path)
                    if error is None:
                        ok_count += 1
                    else:
                        bad_entries.append((video_path, error))

        print(
            f"[{self.__class__.__name__}] Checked {checked_count} video file(s): "
            f"{ok_count} OK, {len(bad_entries)} bad"
        )
        if bad_entries:
            print(f"[{self.__class__.__name__}] Bad video files:")
            for video_path, error in bad_entries:
                print(f"  - {video_path}\n      {error}")

        return len(bad_entries) == 0

    @staticmethod
    def _infer_camera_names(rmb_path):
        camera_names = set()
        for filename in os.listdir(rmb_path):
            if not filename.endswith(".rmb.mp4"):
                continue
            key = filename[: -len(".rmb.mp4")]
            if DataKey.is_rgb_image_key(key):
                camera_names.add(key[: -len("_rgb_image")])
            elif DataKey.is_depth_image_key(key):
                camera_names.add(key[: -len("_depth_image")])
        return sorted(camera_names)

    @staticmethod
    def _probe(video_path):
        if os.path.getsize(video_path) == 0:
            return "file is zero bytes"
        try:
            ffmpeg.probe(video_path)
        except ffmpeg.Error as e:
            stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
            return f"ffprobe failed: {stderr.strip().splitlines()[-1] if stderr.strip() else e}"
        return None


if __name__ == "__main__":
    check_depth_videos = CheckDepthVideos(**vars(parse_argument()))
    is_ok = check_depth_videos.run()
    if not is_ok:
        exit(1)
