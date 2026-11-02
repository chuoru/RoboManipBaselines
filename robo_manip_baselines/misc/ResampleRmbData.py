"""Resample RMB episodes onto a fixed time grid (the policy's control rate).

Teleop records at whatever rate the loop manages (camera-bound and jittery),
the segment runner records at its fixed control period. Training on a mix of
the two with one --skip would give the policy two different time scales for
"one step". Resampling everything to the rollout rate (default 5 Hz = the
segment runner's 0.2 s loop) and training with --skip 1 removes that.

Each output step takes the source step nearest in DataKey.TIME, for every key
(numbers and images alike) -- no interpolation, so binary signals (gripper,
intervention flags) stay binary and every image is a real camera frame.
The output's TIME is the uniform grid; the source index of each step is kept
as "resample_source_idx", and the attrs record resampled_hz.

Usage:
    python ./misc/ResampleRmbData.py \\
        --in_dir ./dataset/CupPickPlace_grasp_SM_iter000 \\
        --out_dir ./dataset/CupPickPlace_grasp_SM_iter000_5hz [--target_hz 5.0]
    python ./misc/ResampleRmbData.py --in_dir ./dataset/X --check   # report only
"""

import argparse
import os
import shutil

import cv2
import numpy as np

from robo_manip_baselines.common import DataKey, DataManager, RmbData, find_rmb_files

RESAMPLE_SOURCE_IDX_KEY = "resample_source_idx"


def get_resample_indices(time_seq, target_hz):
    """Indices into time_seq nearest to a uniform grid from time_seq[0] to
    time_seq[-1] at target_hz. time_seq must be non-decreasing."""
    time_seq = np.asarray(time_seq, dtype=np.float64)
    if len(time_seq) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    period = 1.0 / target_hz
    num_steps = int(np.floor((time_seq[-1] - time_seq[0]) / period + 1e-6)) + 1
    grid = time_seq[0] + period * np.arange(num_steps)
    right = np.clip(np.searchsorted(time_seq, grid), 0, len(time_seq) - 1)
    left = np.clip(right - 1, 0, len(time_seq) - 1)
    choose_left = np.abs(grid - time_seq[left]) <= np.abs(time_seq[right] - grid)
    indices = np.where(choose_left, left, right)
    return indices.astype(np.int64), grid


def describe_timing(time_seq):
    dt = np.diff(np.asarray(time_seq, dtype=np.float64))
    if len(dt) == 0:
        return "single step"
    return (
        f"{len(time_seq)} steps, {time_seq[-1] - time_seq[0]:.1f} s, "
        f"rate {1.0 / np.median(dt):.1f} Hz (dt median {np.median(dt) * 1000:.0f} ms, "
        f"max {dt.max() * 1000:.0f} ms)"
    )


def resample_episode(in_path, out_path, target_hz, image_size=None):
    with RmbData(in_path) as rmb_data:
        time_seq = rmb_data[DataKey.TIME][:]
        indices, grid = get_resample_indices(time_seq, target_hz)
        num_unique = len(np.unique(indices))
        all_data_seq = {}
        for key in rmb_data.keys():
            if key == DataKey.TIME:
                continue
            if DataKey.is_rgb_image_key(key):
                # Frame by frame: decoding the whole 1280x800 video at once
                # costs ~3 MB per frame of RAM.
                video = rmb_data[key]
                frames = {}
                seq = []
                for idx in indices:
                    idx = int(idx)
                    if idx not in frames:
                        frame = np.asarray(video[idx])
                        if image_size is not None:
                            frame = cv2.resize(
                                frame, tuple(image_size), interpolation=cv2.INTER_AREA
                            )
                        frames[idx] = frame
                    seq.append(frames[idx])
                all_data_seq[key] = seq
            elif DataKey.is_depth_image_key(key):
                depth = rmb_data[key][:]  # the depth reader decodes it all anyway
                seq = [depth[int(idx)] for idx in indices]
                if image_size is not None:
                    seq = [
                        cv2.resize(
                            frame, tuple(image_size), interpolation=cv2.INTER_NEAREST
                        )
                        for frame in seq
                    ]
                all_data_seq[key] = seq
            else:
                data = rmb_data[key][:]
                if len(data) != len(time_seq):
                    raise ValueError(
                        f"{in_path}: key '{key}' has {len(data)} steps, TIME has {len(time_seq)}"
                    )
                all_data_seq[key] = list(data[indices])
        all_data_seq[DataKey.TIME] = list(grid)
        all_data_seq[RESAMPLE_SOURCE_IDX_KEY] = list(indices)
        meta_data = {key: rmb_data.attrs[key] for key in rmb_data.attrs.keys()}
    meta_data.pop("format", None)
    meta_data["resampled_hz"] = float(target_hz)
    meta_data["resample_source"] = os.path.abspath(in_path)
    if image_size is not None:
        meta_data["resampled_image_size"] = list(image_size)

    data_manager = DataManager(None)
    data_manager.save_data(
        out_path, all_data_seq, meta_data, increment_episode_idx=False
    )
    return len(time_seq), len(indices), num_unique


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--in_dir", type=str, nargs="+", required=True)
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="output directory (with several --in_dir: one subdirectory per input)",
    )
    parser.add_argument("--target_hz", type=float, default=5.0)
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help="also shrink the images (default: keep the recorded size)",
    )
    parser.add_argument(
        "--check", action="store_true", help="only report the timing of each episode"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace existing output episodes"
    )
    args = parser.parse_args()

    if not args.check and args.out_dir is None:
        parser.error("--out_dir is required unless --check is given")

    for in_dir in args.in_dir:
        in_dir = os.path.normpath(in_dir)
        out_dir = args.out_dir
        if out_dir is not None and len(args.in_dir) > 1:
            out_dir = os.path.join(out_dir, os.path.basename(in_dir))
        for in_path in find_rmb_files(in_dir):
            with RmbData(in_path) as rmb_data:
                time_seq = rmb_data[DataKey.TIME][:]
                already = rmb_data.attrs.get("resampled_hz")
            print(f"{in_path}: {describe_timing(time_seq)}")
            if already is not None:
                print(f"  already resampled at {already} Hz")
            if args.check:
                continue
            rel_path = (
                os.path.relpath(in_path, in_dir)
                if os.path.isdir(in_dir)
                else os.path.basename(in_path)
            )
            out_path = os.path.join(out_dir, rel_path)
            if os.path.exists(out_path):
                if not args.overwrite:
                    print(f"  skip, exists: {out_path}")
                    continue
                if os.path.isdir(out_path):
                    shutil.rmtree(out_path)
                else:
                    os.remove(out_path)
            num_in, num_out, num_unique = resample_episode(
                in_path, out_path, args.target_hz, args.image_size
            )
            msg = f"  -> {out_path}: {num_in} -> {num_out} steps"
            if num_unique < num_out:
                # The source was slower than the target rate somewhere: some
                # frames are repeated.
                msg += f" ({num_out - num_unique} repeated: source slower than {args.target_hz} Hz)"
            print(msg)
