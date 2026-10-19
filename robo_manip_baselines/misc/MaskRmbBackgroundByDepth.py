"""Write a copy of an RMB dataset whose RGB images have the background removed
using the depth image (common/utils/DepthMaskUtils.py) -- the same function
RealEnvBase applies at rollout time when pointcloud_camera_background_max_depth
is set.

Masking is done at the recorded resolution. Optionally a second, downscaled
copy is written as well (e.g. 320x200 for Diffusion Policy), with RGB resized
(INTER_LINEAR, like the rollout-time cv2.resize) and depth resized (INTER_NEAREST); all other data is copied
unchanged. Non-RMB files in the source (e.g. manifest.csv) are copied too.

Usage:
    python ./misc/MaskRmbBackgroundByDepth.py <src_dir> <dst_dir> \\
        [--small_dst_dir <dst_dir_small> --small_size 320 200] [--max_depth 0.6]
"""

import argparse
import glob
import os
import shutil

import cv2
import numpy as np

from robo_manip_baselines.common import (
    DataKey,
    DataManager,
    RmbData,
    mask_background_by_depth,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src_dir")
    parser.add_argument("dst_dir")
    parser.add_argument("--camera_name", default="hand")
    parser.add_argument("--max_depth", type=float, default=0.6, help="[m]")
    parser.add_argument("--small_dst_dir", default=None)
    parser.add_argument("--small_size", type=int, nargs=2, default=[320, 200])
    args = parser.parse_args()

    for d in (args.dst_dir, args.small_dst_dir):
        if d is not None and os.path.exists(d):
            raise FileExistsError(f"{d} already exists")
    data_manager = DataManager(None)
    rgb_key = DataKey.get_rgb_image_key(args.camera_name)
    depth_key = DataKey.get_depth_image_key(args.camera_name)

    srcs = sorted(glob.glob(os.path.join(args.src_dir, "*.rmb")))
    for src in srcs:
        with RmbData(src) as rmb_data:
            seq = {key: rmb_data[key][:] for key in rmb_data.keys()}
            meta_data = {key: rmb_data.attrs[key] for key in rmb_data.attrs.keys()}
        meta_data.pop("format", None)
        meta_data[rgb_key + "_background_max_depth"] = args.max_depth

        seq[rgb_key] = np.stack(
            [mask_background_by_depth(rgb, depth, args.max_depth)
             for rgb, depth in zip(seq[rgb_key], seq[depth_key])]
        )
        name = os.path.basename(src)
        os.makedirs(args.dst_dir, exist_ok=True)
        data_manager.save_data(os.path.join(args.dst_dir, name),
                               {k: list(v) for k, v in seq.items()}, meta_data,
                               increment_episode_idx=False)

        if args.small_dst_dir is not None:
            w, h = args.small_size
            small = dict(seq)
            small[rgb_key] = np.stack(
                [cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR) for x in seq[rgb_key]])
            small[depth_key] = np.stack(
                [cv2.resize(x, (w, h), interpolation=cv2.INTER_NEAREST) for x in seq[depth_key]])
            os.makedirs(args.small_dst_dir, exist_ok=True)
            data_manager.save_data(os.path.join(args.small_dst_dir, name),
                                   {k: list(v) for k, v in small.items()}, meta_data,
                                   increment_episode_idx=False)
        print(f"[MaskRmbBackgroundByDepth] {name} ({len(seq[DataKey.TIME])} steps)", flush=True)

    for f in glob.glob(os.path.join(args.src_dir, "*")):
        if not f.endswith(".rmb") and os.path.isfile(f):
            for d in (args.dst_dir, args.small_dst_dir):
                if d is not None:
                    shutil.copy(f, d)
    print(f"[MaskRmbBackgroundByDepth] wrote {len(srcs)} episodes")


if __name__ == "__main__":
    main()
