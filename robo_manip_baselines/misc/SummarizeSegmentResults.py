"""Summarize segment-runner result YAMLs (logs/segment_results_*.yaml) for
the exhibit-mode endurance test (doc/segment_dagger_checklist.md, M7).

Prints, as Markdown ready for doc/evaluation_results.md:
  - per learned segment: attempts, success on the first try / after a retry,
    stops, failure reasons
  - stops for the staff and their reasons
  - cycle time (mean / median / max) and available memory over the run
  - the M7 verdict: at least --min_cycles cycles and at most --max_stops stops

Usage:
    python ./misc/SummarizeSegmentResults.py ./logs/segment_results_CupPickPlace_20261120_100000.yaml [more.yaml ...]
"""

import argparse
import collections

import numpy as np
import yaml


def summarize(result_paths):
    cycles = []
    segment_stats = collections.OrderedDict()
    for path in result_paths:
        with open(path, "r") as f:
            results = yaml.safe_load(f) or {}
        cycles += results.get("cycles", [])
        for name, stat in (results.get("segments") or {}).items():
            total = segment_stats.setdefault(
                name,
                {
                    "attempts": 0,
                    "success_first_try": 0,
                    "success_after_retry": 0,
                    "stopped": 0,
                    "fail_reasons": collections.Counter(),
                },
            )
            for key in (
                "attempts",
                "success_first_try",
                "success_after_retry",
                "stopped",
            ):
                total[key] += stat.get(key, 0)
            total["fail_reasons"].update(stat.get("fail_reasons") or {})
    return cycles, segment_stats


def format_report(cycles, segment_stats, min_cycles, max_stops):
    lines = []
    complete = [c for c in cycles if not c.get("incomplete")]
    stops = [c for c in cycles if c.get("stopped")]
    lines.append(f"- Cycles: {len(complete)} completed ({len(cycles)} recorded)")
    lines.append(
        f"- Stops for the staff: {len(stops)}"
        + (
            ""
            if not stops
            else " -- "
            + "; ".join(f"cycle {c['cycle']}: {c.get('stop_reason')}" for c in stops)
        )
    )
    durations = [
        c["duration"] for c in complete if not c.get("stopped") and "duration" in c
    ]
    if durations:
        lines.append(
            f"- Cycle time [s] (cycles without a stop): mean {np.mean(durations):.1f}, "
            f"median {np.median(durations):.1f}, max {np.max(durations):.1f}"
        )
    memory = [c["memory_available_gb"] for c in cycles if "memory_available_gb" in c]
    if memory:
        lines.append(
            f"- Available memory [GB]: first {memory[0]:.2f}, last {memory[-1]:.2f}, "
            f"min {np.min(memory):.2f}"
        )
    lines.append("")
    lines.append(
        "| Segment | Attempts | Success, 1st try | Success after retry | Stopped | Failure reasons |"
    )
    lines.append("|---|---|---|---|---|---|")
    for name, stat in segment_stats.items():
        episodes = (
            stat["success_first_try"] + stat["success_after_retry"] + stat["stopped"]
        )

        def pct(value):
            return (
                f"{value} ({100.0 * value / episodes:.0f}%)" if episodes else str(value)
            )

        reasons = ", ".join(
            f"{reason}: {num}" for reason, num in stat["fail_reasons"].most_common()
        )
        lines.append(
            f"| {name} | {stat['attempts']} | {pct(stat['success_first_try'])} | "
            f"{pct(stat['success_after_retry'])} | {pct(stat['stopped'])} | {reasons or '-'} |"
        )
    lines.append("")
    passed = len(complete) >= min_cycles and len(stops) <= max_stops
    lines.append(
        f"**M7 verdict: {'PASS' if passed else 'FAIL'}** "
        f"(needs >= {min_cycles} cycles and <= {max_stops} stops; "
        f"got {len(complete)} cycles, {len(stops)} stops)"
    )
    return "\n".join(lines), passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("result_files", nargs="+")
    parser.add_argument("--min_cycles", type=int, default=30)
    parser.add_argument("--max_stops", type=int, default=1)
    args = parser.parse_args()

    cycles, segment_stats = summarize(args.result_files)
    report, _ = format_report(cycles, segment_stats, args.min_cycles, args.max_stops)
    print(report)
