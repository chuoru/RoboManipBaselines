"""Compare training iterations of the segment policies from their recorded
attempts (dataset/<Task>_<segment>_rollout_iterNNN, written by
bin/RolloutSegments.py --save_rollout --iteration NNN).

Data of iteration N was collected with the policy of iteration N-1, so each
row says how well THAT policy did. Per segment and iteration:
attempts, success rate, interventions per attempt, mean intervention time,
retry rate (attempts that were retries) and the failure reasons. Writes

    <out_dir>/iteration_summary.csv
    <out_dir>/iteration_summary.png              success rate / interventions per attempt
    <out_dir>/failures_<segment>_iterNNN.png     first camera frame of each failed attempt

The failure tiles show where the policy starts from when it fails -- look
for positions / orientations it has trouble with, and collect demos there.

M6 is met when, for comparable scenes, interventions per attempt go down or
the success rate goes up from one iteration to the next.

Usage:
    python ./misc/SummarizeIterations.py --task CupPickPlace [--dataset_root ./dataset]
"""

import argparse
import collections
import csv
import glob
import os
import re

import cv2
import matplotlib
import matplotlib.ticker

matplotlib.use("agg")
import matplotlib.pylab as plt  # noqa: E402
import numpy as np  # noqa: E402

from robo_manip_baselines.common import DataKey, RmbData, find_rmb_files  # noqa: E402


def _attr(attrs, key, default=None):
    value = attrs.get(key, default)
    if isinstance(value, bytes):
        value = value.decode()
    return value


def collect(dataset_root, task):
    """{(segment, iteration): [attrs dict + path]} of every recorded attempt."""
    attempts = collections.defaultdict(list)
    pattern = os.path.join(dataset_root, f"{task}_*_rollout_iter*")
    for directory in sorted(glob.glob(pattern)):
        match = re.match(
            rf"{re.escape(task)}_(.+)_rollout_iter(\d+)$", os.path.basename(directory)
        )
        if match is None or not os.path.isdir(directory):
            continue
        segment, iteration = match.group(1), int(match.group(2))
        for path in find_rmb_files(directory):
            with RmbData(path) as rmb_data:
                attrs = {
                    key: _attr(rmb_data.attrs, key) for key in rmb_data.attrs.keys()
                }
            attrs["_path"] = path
            attempts[(segment, iteration)].append(attrs)
    return attempts


def summarize(attempts):
    rows = []
    for (segment, iteration), items in sorted(attempts.items()):
        results = [item.get("segment_result", "") for item in items]
        decided = [r for r in results if r in ("success", "failure")]
        interventions = [int(item.get("num_interventions", 0) or 0) for item in items]
        intervention_times = [
            float(item.get("intervention_time", 0.0) or 0.0)
            for item, num in zip(items, interventions)
            if num > 0
        ]
        reasons = collections.Counter(
            item.get("fail_reason") or "unknown"
            for item in items
            if item.get("segment_result") == "failure"
        )
        checkpoints = sorted(
            {
                os.path.basename(os.path.dirname(item.get("checkpoint") or ""))
                for item in items
            }
        )
        rows.append(
            {
                "segment": segment,
                "data_iteration": iteration,
                "policy": ",".join(c for c in checkpoints if c) or "-",
                "attempts": len(items),
                "success_rate": (
                    float(np.mean([r == "success" for r in decided]))
                    if decided
                    else float("nan")
                ),
                "interventions_per_attempt": float(np.mean(interventions))
                if items
                else float("nan"),
                "mean_intervention_time": (
                    float(np.mean(intervention_times)) if intervention_times else 0.0
                ),
                "retry_rate": float(
                    np.mean([int(item.get("retry_idx", 0) or 0) > 0 for item in items])
                ),
                "fail_reasons": "; ".join(
                    f"{k}: {v}" for k, v in reasons.most_common()
                ),
            }
        )
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for segment in sorted({row["segment"] for row in rows}):
        seg_rows = sorted(
            (row for row in rows if row["segment"] == segment),
            key=lambda r: r["data_iteration"],
        )
        x = [row["data_iteration"] - 1 for row in seg_rows]  # policy iteration
        axes[0].plot(x, [row["success_rate"] for row in seg_rows], "o-", label=segment)
        axes[1].plot(
            x,
            [row["interventions_per_attempt"] for row in seg_rows],
            "o-",
            label=segment,
        )
    axes[0].set_title("success rate")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_title("interventions / attempt")
    for ax in axes:
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_xlabel("policy iteration")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def first_frame(path):
    with RmbData(path) as rmb_data:
        for key in rmb_data.keys():
            if DataKey.is_rgb_image_key(key):
                return np.asarray(rmb_data[key][0])
    return None


def tile_failures(items, path, max_images=24, tile_width=320):
    failed = [item for item in items if item.get("segment_result") == "failure"][
        :max_images
    ]
    tiles = []
    for item in failed:
        image = first_frame(item["_path"])
        if image is None:
            continue
        height = int(tile_width * image.shape[0] / image.shape[1])
        tile = cv2.cvtColor(cv2.resize(image, (tile_width, height)), cv2.COLOR_RGB2BGR)
        label = f"{item.get('fail_reason') or '?'} c{item.get('cycle_idx', '?')} r{item.get('retry_idx', '?')}"
        cv2.putText(
            tile, label, (6, 22), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1
        )
        tiles.append(tile)
    if not tiles:
        return False
    cols = min(6, len(tiles))
    blank = np.zeros_like(tiles[0])
    tiles += [blank] * (-len(tiles) % cols)
    grid = cv2.vconcat(
        [
            cv2.hconcat(tiles[row * cols : (row + 1) * cols])
            for row in range(len(tiles) // cols)
        ]
    )
    cv2.imwrite(path, grid)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, default="./dataset")
    parser.add_argument("--out_dir", type=str, default="./logs/iteration_summary")
    parser.add_argument(
        "--no_images", action="store_true", help="skip the failure tiles"
    )
    args = parser.parse_args()

    attempts = collect(args.dataset_root, args.task)
    if not attempts:
        raise SystemExit(
            f"No {args.task}_*_rollout_iter* attempts under {args.dataset_root}."
        )
    rows = summarize(attempts)
    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(rows, os.path.join(args.out_dir, "iteration_summary.csv"))
    plot(rows, os.path.join(args.out_dir, "iteration_summary.png"))

    print(
        "| segment | policy iter | attempts | success | interventions/attempt | intervention time [s] | retry rate | failures |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for row in rows:
        print(
            f"| {row['segment']} | {row['data_iteration'] - 1} ({row['policy']}) | {row['attempts']} | "
            f"{row['success_rate']:.0%} | {row['interventions_per_attempt']:.2f} | "
            f"{row['mean_intervention_time']:.1f} | {row['retry_rate']:.0%} | {row['fail_reasons'] or '-'} |"
        )
    if not args.no_images:
        for (segment, iteration), items in sorted(attempts.items()):
            out = os.path.join(
                args.out_dir, f"failures_{segment}_iter{iteration:03d}.png"
            )
            if tile_failures(items, out):
                print(f"Failure starts: {out}")
    print(f"Written to {args.out_dir}")
