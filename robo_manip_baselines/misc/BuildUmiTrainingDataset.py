"""Merge RealUMIDemo datasets into one training dataset, cleaning each episode.

Per episode:
  1. Idle prefix: drop the frozen-pose prefix recorded before the Vive anchor
     settled (episodes collected before ViveInputDevice.is_ready() required
     enabled teleop) -- same rule as misc/TrimRmbIdlePrefix.py. Episodes
     without one are left as is.
  2. Gripper signal: the marker-based gripper width occasionally misreads as
     "open" while the cup is still held (dim lighting, a finger tag occluded ->
     single-marker fallback), and very rarely flickers to "closed" while open.
     Closed blips shorter than --blip_s are removed, then open gaps of at most
     --gap_s between two closed segments are filled by linear interpolation
     between the surrounding closed values. All gripper-carrying keys are
     kept consistent (measured/command gripper, joint_pos[..., -1], *_rel).

The source datasets are not modified. A manifest.csv in the output records
where each episode came from and what was changed.

Usage:
    python ./misc/BuildUmiTrainingDataset.py <dst_dir> <src_dir> [<src_dir> ...]
"""

import argparse
import csv
import glob
import os

import numpy as np

from robo_manip_baselines.common import DataKey, DataManager, RmbData


def runs(mask):
    """[(start, end_exclusive, value)] runs of a boolean array."""
    out = []
    start = 0
    for i in range(1, len(mask) + 1):
        if i == len(mask) or mask[i] != mask[start]:
            out.append((start, i, bool(mask[start])))
            start = i
    return out


def clean_gripper(gripper, hz, threshold, blip_s, gap_s):
    gripper = gripper.copy()
    n_blips = n_gaps = 0
    closed = gripper > threshold
    for start, end, value in runs(closed):
        if value and (end - start) < blip_s * hz and start > 0 and end < len(gripper):
            gripper[start:end] = np.interp(
                np.arange(start, end), [start - 1, end], [gripper[start - 1], gripper[end]]
            )
            n_blips += 1
    closed = gripper > threshold
    for start, end, value in runs(closed):
        if (not value) and start > 0 and end < len(gripper) and (end - start) <= gap_s * hz:
            gripper[start:end] = np.interp(
                np.arange(start, end), [start - 1, end], [gripper[start - 1], gripper[end]]
            )
            n_gaps += 1
    return gripper, n_blips, n_gaps


def rel_of(seq):
    seq = np.asarray(seq)
    rel = np.zeros_like(seq)
    rel[1:] = seq[1:] - seq[:-1]
    return rel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dst_dir")
    parser.add_argument("src_dirs", nargs="+")
    parser.add_argument("--threshold", type=float, default=30.0, help="percent closed")
    parser.add_argument("--blip_s", type=float, default=0.3)
    parser.add_argument("--gap_s", type=float, default=1.5)
    args = parser.parse_args()

    if os.path.exists(args.dst_dir):
        raise FileExistsError(f"{args.dst_dir} already exists")
    os.makedirs(args.dst_dir)
    data_manager = DataManager(None)
    manifest = []
    out_idx = 0

    for src_dir in args.src_dirs:
        for src in sorted(glob.glob(os.path.join(src_dir, "*.rmb"))):
            with RmbData(src) as rmb_data:
                seq = {key: rmb_data[key][:] for key in rmb_data.keys()}
                meta_data = {key: rmb_data.attrs[key] for key in rmb_data.attrs.keys()}
            meta_data.pop("format", None)
            n_src = len(seq[DataKey.TIME])

            # 1. idle prefix
            cmd = seq[DataKey.COMMAND_EEF_POSE]
            frozen = np.abs(cmd - cmd[0]).max(axis=1) == 0
            start = 0
            if frozen[1:3].all():
                moved = ~frozen
                if not moved.any():
                    print(f"[BuildUmiTrainingDataset] skip (never moves): {src}")
                    continue
                start = max(int(np.argmax(moved)) - 1, 0)
            seq = {key: value[start:] for key, value in seq.items()}
            seq[DataKey.TIME] = seq[DataKey.TIME] - seq[DataKey.TIME][0]

            # 2. gripper
            t = seq[DataKey.TIME]
            hz = (len(t) - 1) / (t[-1] - t[0])
            measured = seq[DataKey.MEASURED_GRIPPER_JOINT_POS][:, 0]
            cleaned, n_blips, n_gaps = clean_gripper(
                measured, hz, args.threshold, args.blip_s, args.gap_s
            )
            if n_blips or n_gaps:
                # command(t) is the marker estimate taken right before step t,
                # i.e. the measured value one step later.
                command = np.r_[cleaned[1:], cleaned[-1]]
                seq[DataKey.MEASURED_GRIPPER_JOINT_POS] = cleaned[:, None].astype(np.float64)
                seq[DataKey.COMMAND_GRIPPER_JOINT_POS] = command[:, None].astype(np.float64)
                for joint_key, value in (
                    (DataKey.MEASURED_JOINT_POS, cleaned),
                    (DataKey.COMMAND_JOINT_POS, command),
                ):
                    joint_pos = seq[joint_key].copy()
                    joint_pos[:, -1] = value
                    seq[joint_key] = joint_pos
                for rel_key in (
                    DataKey.MEASURED_GRIPPER_JOINT_POS_REL,
                    DataKey.COMMAND_GRIPPER_JOINT_POS_REL,
                    DataKey.MEASURED_JOINT_POS_REL,
                    DataKey.COMMAND_JOINT_POS_REL,
                ):
                    if rel_key in seq:
                        seq[rel_key] = rel_of(seq[DataKey.get_abs_key(rel_key)])

            dst = os.path.join(args.dst_dir, f"RealUMIDemo_world0_{out_idx:03d}.rmb")
            data_manager.save_data(
                dst, {key: list(value) for key, value in seq.items()}, meta_data,
                increment_episode_idx=False,
            )
            manifest.append(dict(dst=os.path.basename(dst), src=src, src_steps=n_src,
                                 dropped_prefix=start, steps=len(t),
                                 gripper_blips_removed=n_blips, gripper_gaps_filled=n_gaps))
            print(f"[BuildUmiTrainingDataset] {os.path.basename(dst)} <- {os.path.relpath(src, os.path.dirname(src_dir))}: "
                  f"dropped {start}/{n_src}, gripper blips {n_blips}, gaps {n_gaps}")
            out_idx += 1

    with open(os.path.join(args.dst_dir, "manifest.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)
    print(f"[BuildUmiTrainingDataset] wrote {out_idx} episodes to {args.dst_dir}")


if __name__ == "__main__":
    main()
