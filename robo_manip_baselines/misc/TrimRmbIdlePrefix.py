"""Write a copy of a RealUMIDemo dataset with each episode's idle prefix removed.

Episodes recorded before ViveInputDevice.is_ready() waited for teleop to be
ENABLED start with several seconds (30-46% of each RealUMIDemo_20261016
episode) in which the Vive anchor was still settling: the recorded
measured/command EEF pose stays exactly at the identity start pose while the
camera image already moves. A policy trained on that learns to stand still at
the start of the task. This keeps each episode from the last frame of that
frozen prefix onwards (so the first kept sample still starts at the identity
pose, and its action is the first real motion), leaving the source dataset
untouched.

Usage:
    python ./misc/TrimRmbIdlePrefix.py <src_dataset_dir> <dst_dataset_dir>
"""

import argparse
import glob
import os

import numpy as np

from robo_manip_baselines.common import DataKey, DataManager, RmbData


def find_start_idx(command_eef_pose):
    moved = np.abs(command_eef_pose - command_eef_pose[0]).max(axis=1) > 0
    if not moved.any():
        return None
    return max(int(np.argmax(moved)) - 1, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src_dir")
    parser.add_argument("dst_dir")
    args = parser.parse_args()

    if os.path.exists(args.dst_dir):
        raise FileExistsError(f"{args.dst_dir} already exists")
    data_manager = DataManager(None)

    for src in sorted(glob.glob(os.path.join(args.src_dir, "*.rmb"))):
        with RmbData(src) as rmb_data:
            start = find_start_idx(rmb_data[DataKey.COMMAND_EEF_POSE][:])
            if start is None:
                print(f"[TrimRmbIdlePrefix] skip (pose never moves): {src}")
                continue
            all_data_seq = {key: list(rmb_data[key][:][start:]) for key in rmb_data.keys()}
            # Keep TIME starting at 0, like a freshly recorded episode.
            time_seq = np.asarray(all_data_seq[DataKey.TIME])
            all_data_seq[DataKey.TIME] = list(time_seq - time_seq[0])
            meta_data = {key: rmb_data.attrs[key] for key in rmb_data.attrs.keys()}
            meta_data.pop("format", None)
            n_src = len(rmb_data[DataKey.TIME])

        dst = os.path.join(args.dst_dir, os.path.basename(src))
        os.makedirs(args.dst_dir, exist_ok=True)
        data_manager.save_data(dst, all_data_seq, meta_data, increment_episode_idx=False)
        print(f"[TrimRmbIdlePrefix] {os.path.basename(src)}: dropped {start}/{n_src} frames")


if __name__ == "__main__":
    main()
