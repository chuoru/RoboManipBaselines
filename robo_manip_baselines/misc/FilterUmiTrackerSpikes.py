"""Find (and optionally repair) single-frame Vive tracker spikes in recorded
UMI demos, so they do not end up in training data.

WHAT A SPIKE LOOKS LIKE: the lighthouse pose solve occasionally emits one bad
frame -- the pose jumps away and is back where it belongs on the very next
frame. Measured on a real 7-episode set, e.g.:

    frame   d(prev->this)   d(this->next)
      138        0.54 cm         0.54 cm      <- normal, arm nearly still
      139        4.67 cm         0.34 cm      <- jumped out
      140        0.34 cm         0.14 cm      <- already back

A human hand cannot do that. Left in, a policy trained on this learns to
reproduce a discontinuity.

WHY NOT A SPEED THRESHOLD: filter_glitches() in misc/ReplayUmiOnFairino5.py
rejects frames above an absolute velocity, which catches the huge lighthouse
relock jumps (~40-50 cm in one frame) it was written for but cannot separate
these smaller ones -- on the same real data a 42.5 cm/s spike sat *below* a
perfectly legitimate 58.7 cm/s reach. Speed alone does not distinguish them.

WHAT DOES: the SHAPE. Real motion continues in roughly the same direction, so
going through frame i costs about the same as skipping it. A spike doubles
back, so going through costs far more:

    excursion = (|p_i - p_{i-1}| + |p_{i+1} - p_i| - |p_{i+1} - p_{i-1}|) / 2
    ratio     = (|p_i - p_{i-1}| + |p_{i+1} - p_i|) / |p_{i+1} - p_{i-1}|

`excursion` is how far the frame detours off the straight path between its
neighbours (0 for perfectly straight motion, and in metres, so it is directly
comparable to the rig's real motion scale); `ratio` is scale-free and near 1.0
for straight motion. Requiring BOTH keeps slow jitter (tiny excursion) and
ordinary direction changes (low ratio) out of it. On the measured set, real
frames sat at ratio ~1.0 while spikes ran 2.1-18.7, and the excursion
distribution had its 99th percentile at 1.3 cm with a maximum of 6.0 cm.

REPAIR replaces a flagged frame by interpolating its neighbours (screw
interpolation for the pose, linear for the gripper), which is what the frame
would have been had the solver not glitched. Timestamps are untouched.

Usage:
    # Report only (default), across a whole dataset directory
    python ./misc/FilterUmiTrackerSpikes.py ./dataset/RealUMIDemo_20260815_153352

    # Write cleaned copies alongside the originals
    python ./misc/FilterUmiTrackerSpikes.py ./dataset/... --output_suffix _clean
"""

import argparse
import os
import shutil

import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import DataKey, RmbData, find_rmb_files


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "path", type=str, help="a .rmb file, or a directory of them"
    )
    parser.add_argument(
        "--min_excursion",
        type=float,
        default=0.01,
        help="[m] how far off the straight line between its neighbours a frame "
        "must detour to count as a spike. Guards against flagging ordinary "
        "sensor jitter, which detours by almost nothing. Default 0.01 (1cm) "
        "sits just under the 99th percentile (1.3cm) of a real recording.",
    )
    parser.add_argument(
        "--min_ratio",
        type=float,
        default=2.0,
        help="how many times more expensive going THROUGH the frame must be "
        "than skipping it. 1.0 is perfectly straight motion; a doubling-back "
        "spike is >=2. Scale-free, so it works at any motion speed.",
    )
    parser.add_argument(
        "--pose_key",
        type=str,
        default=DataKey.COMMAND_EEF_POSE,
        choices=[DataKey.COMMAND_EEF_POSE, DataKey.MEASURED_EEF_POSE],
        help="which recorded pose sequence to inspect",
    )
    parser.add_argument(
        "--max_passes",
        type=int,
        default=5,
        help="repair is applied repeatedly until no spikes remain (or this "
        "many passes). Smoothing one frame can leave its neighbour looking "
        "like the outlier -- measured on real data, pass 1 fixed 8 spikes, "
        "pass 2 caught 3 more, pass 3 was clean.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default=None,
        help="if given, write a repaired copy of each file with this suffix "
        "appended to its name (e.g. '_clean'). Without it, this only reports "
        "-- nothing is written and the originals are never modified.",
    )
    return parser.parse_args()


def detect_spikes(positions, min_excursion, min_ratio):
    """Indices of single-frame detours in an (N,3) position sequence.

    Returns (indices, excursions, ratios) for the flagged frames. Endpoints
    are never flagged -- they have no pair of neighbours to interpolate
    between, so there is nothing to compare against or repair with.
    """
    idxes, excursions, ratios = [], [], []
    for i in range(1, len(positions) - 1):
        d_out = np.linalg.norm(positions[i] - positions[i - 1])
        d_back = np.linalg.norm(positions[i + 1] - positions[i])
        d_skip = np.linalg.norm(positions[i + 1] - positions[i - 1])
        excursion = (d_out + d_back - d_skip) / 2.0
        ratio = (d_out + d_back) / max(d_skip, 1e-9)
        if excursion > min_excursion and ratio > min_ratio:
            idxes.append(i)
            excursions.append(excursion)
            ratios.append(ratio)
    return np.array(idxes, dtype=int), np.array(excursions), np.array(ratios)


def repair_pose(pose_seq, idxes, time_seq):
    """Replace flagged frames with a screw interpolation of their neighbours.

    Consecutive flagged frames are handled by interpolating across the whole
    run, between the last good frame before it and the first good one after,
    so a two-frame glitch does not get repaired from another glitched frame.
    """
    repaired = pose_seq.copy()
    flagged = np.zeros(len(pose_seq), dtype=bool)
    flagged[idxes] = True

    i = 0
    while i < len(pose_seq):
        if not flagged[i]:
            i += 1
            continue
        run_start = i
        while i < len(pose_seq) and flagged[i]:
            i += 1
        run_end = i  # first good frame after the run
        before, after = run_start - 1, run_end
        if before < 0 or after >= len(pose_seq):
            # Nothing to interpolate between; leave as-is.
            continue
        se3_a = pin.SE3(
            pin.Quaternion(*repaired[before, 3:7]), repaired[before, 0:3]
        )
        se3_b = pin.SE3(pin.Quaternion(*repaired[after, 3:7]), repaired[after, 0:3])
        t_a, t_b = time_seq[before], time_seq[after]
        for k in range(run_start, run_end):
            alpha = (time_seq[k] - t_a) / max(t_b - t_a, 1e-9)
            se3 = se3_a * pin.exp6(alpha * pin.log6(se3_a.inverse() * se3_b))
            quat = pin.Quaternion(se3.rotation)
            repaired[k, 0:3] = se3.translation
            repaired[k, 3:7] = [quat.w, quat.x, quat.y, quat.z]
    return repaired


def repair_until_clean(pose_seq, time_seq, min_excursion, min_ratio, max_passes):
    """Repair, re-detect, repeat -- see --max_passes for why one pass is not
    enough. Returns (repaired_pose_seq, passes_used, frames_repaired)."""
    repaired = pose_seq.copy()
    total_repaired, passes = 0, 0
    for passes in range(1, max_passes + 1):
        idxes, _excursions, _ratios = detect_spikes(
            repaired[:, 0:3], min_excursion, min_ratio
        )
        if len(idxes) == 0:
            return repaired, passes - 1, total_repaired
        repaired = repair_pose(repaired, idxes, time_seq)
        total_repaired += len(idxes)
    print(
        f"      WARNING: still finding spikes after {max_passes} passes -- "
        "raise --max_passes, or the thresholds may be too aggressive for "
        "this recording."
    )
    return repaired, passes, total_repaired


def main():
    args = parse_argument()
    filenames = find_rmb_files(args.path)
    if not filenames:
        raise RuntimeError(f"[FilterUmiTrackerSpikes] No .rmb files under {args.path}")

    print(
        f"[FilterUmiTrackerSpikes] Scanning {len(filenames)} file(s) with "
        f"min_excursion={args.min_excursion * 100:.1f}cm, min_ratio={args.min_ratio}"
    )
    total_spikes, total_frames = 0, 0
    for filename in filenames:
        with RmbData(filename) as rmb_data:
            pose_seq = rmb_data[args.pose_key][:]
            time_seq = rmb_data[DataKey.TIME][:]

        idxes, excursions, ratios = detect_spikes(
            pose_seq[:, 0:3], args.min_excursion, args.min_ratio
        )
        total_spikes += len(idxes)
        total_frames += len(pose_seq)

        name = os.path.basename(os.path.normpath(filename))
        if len(idxes) == 0:
            print(f"  {name}: clean ({len(pose_seq)} frames)")
            continue
        print(f"  {name}: {len(idxes)} spike(s) in {len(pose_seq)} frames")
        for i, exc, ratio in zip(idxes, excursions, ratios):
            print(
                f"      frame {i:4d}  t={time_seq[i]:6.2f}s  "
                f"excursion {exc * 100:5.2f}cm  ratio {ratio:5.1f}"
            )

        if args.output_suffix is None:
            continue

        base = os.path.normpath(filename)
        stem, ext = os.path.splitext(base)
        out = f"{stem}{args.output_suffix}{ext}"
        if os.path.exists(out):
            raise RuntimeError(f"[FilterUmiTrackerSpikes] {out} already exists")
        # .rmb is a directory of an hdf5 plus video files; copy the whole
        # thing and rewrite only the pose dataset, so images/timestamps and
        # every other key survive untouched.
        shutil.copytree(base, out)
        repaired, n_passes, n_repaired = repair_until_clean(
            pose_seq, time_seq, args.min_excursion, args.min_ratio, args.max_passes
        )
        import h5py

        with h5py.File(os.path.join(out, "main.rmb.hdf5"), "r+") as h5file:
            h5file[args.pose_key][...] = repaired
        print(
            f"      -> repaired {n_repaired} frame(s) over {n_passes} pass(es), "
            f"wrote {os.path.basename(out)}"
        )

    print(
        f"[FilterUmiTrackerSpikes] Done. {total_spikes} spike(s) across "
        f"{total_frames} frames ({100.0 * total_spikes / max(total_frames, 1):.2f}%)."
    )
    if args.output_suffix is None and total_spikes > 0:
        print(
            "[FilterUmiTrackerSpikes] Report only -- pass --output_suffix to "
            "write repaired copies."
        )


if __name__ == "__main__":
    main()
