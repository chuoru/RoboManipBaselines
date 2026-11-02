"""Measure the true fully-open/fully-closed gripper-marker width (see
common/utils/ArucoGripperUtils.py, RealUMIEnvBase._estimate_gripper_percent_closed_from_markers)
by tracking the live RAW width_m's running min/max while you open/close the
UMI rig by hand -- raw meaning before the outlier-rejection/EMA-smoothing
that production teleop applies, so the numbers reported here aren't lagged
or clamped by those filters.

Use this to fill in gripper_marker_config's width_open_m/width_closed_m in
envs/configs/RealUMIDemo.yaml precisely, instead of guessing or
back-calculating from an indirect symptom (e.g. a percent-closed reading
that never reaches 100%, which just as easily means width_closed_m is set
too small as it means the rig wasn't closed all the way during calibration
-- this tool removes that ambiguity by showing the raw measured width
directly).

Usage:
    python ./misc/CalibrateGripperMarkerWidth.py \\
        --config ./envs/configs/RealUMIDemo.yaml
Open the rig fully, hold for a second, close it fully (both slowly AND at
least once quickly, to also sanity-check detection keeps up), hold, repeat a
few times covering the full range. Press 'q' or Esc to stop and print the
observed min/max width_m plus the exact YAML lines to paste in.

FR5 (tool_do gripper, tags on the fingers seen by the arm's hand camera):
    python ./misc/CalibrateGripperMarkerWidth.py --env RealFairino5DemoEnv-v0 \
        --config ./envs/configs/RealFairino5DemoEnv_SpaceMouse.yaml \
        --gripper_marker ./envs/configs/gripper_marker/CupPickPlace.yaml --save_to \
        ./envs/configs/gripper_marker/CupPickPlace.yaml
In the window: 'o' / 'c' open / close the gripper, then record ~2 s of
readings with '1' (fully open), '2' (closed on nothing) and '3' (closed on
the cup -- put the cup between the fingers first; repeat '3' at a few cup
positions/orientations). 'q' / Esc ends, prints the medians and the check
thresholds derived from them, and (with --save_to) writes them into the
gripper_marker file: width_open_m / width_closed_m and
checks.grasp.width_min_m / width_max_m, checks.release.width_open_min_m.
"""

import argparse
import re
import time

import cv2
import gymnasium as gym
import numpy as np
import yaml

import robo_manip_baselines.envs  # noqa: F401 (registers the gym envs)
from robo_manip_baselines.common import (
    detect_aruco_tags,
    get_gripper_width,
    parse_aruco_config,
    preprocess_low_light_image,
)


def set_yaml_values(path, values):
    """Replace the value of the first `key:` line for each key, keeping the
    file's comments and layout (the keys of a gripper_marker file are
    unique)."""
    with open(path, "r") as f:
        text = f.read()
    for key, value in values.items():
        text, num = re.subn(
            rf"^(\s*{re.escape(key)}:)[^\n#]*", rf"\g<1> {value}", text, count=1, flags=re.M
        )
        if num == 0:
            raise KeyError(f"{key} not found in {path}")
    with open(path, "w") as f:
        f.write(text)


def calibrate_fr5(args):
    """Interactive calibration on the FR5 (see the module docstring)."""
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if args.gripper_marker is not None:
        config["gripper_marker_config"] = args.gripper_marker
    if config.get("gripper_marker_config") is None:
        raise RuntimeError("Pass --gripper_marker (or set gripper_marker_config in --config).")
    env = gym.make(
        f"robo_manip_baselines/{args.env}",
        **{"camera_ids": None, "gelsight_ids": None, **config},
    )
    unwrapped = env.unwrapped
    tracker = unwrapped.gripper_marker_tracker
    env.reset()
    zero_action = np.zeros(unwrapped.action_space.shape)

    samples = {"open": [], "empty": [], "cup": []}
    recording = None  # (label, end time)
    print("o/c: open/close, 1: record open, 2: record closed-empty, 3: record closed-on-cup, q: finish")
    try:
        while True:
            # The arm is never enabled here (no move_to_init_pose), so this
            # only refreshes the observation and the camera frame.
            env.step(zero_action)
            frame = unwrapped.get_latest_rgb_camera_frame(tracker.camera_name)
            width_m = None if frame is None else tracker.detect_width(frame)
            if recording is not None:
                label, end_time = recording
                if width_m is not None:
                    samples[label].append(width_m)
                if time.time() > end_time:
                    print(f"  {label}: {len(samples[label])} samples so far")
                    recording = None
            if frame is not None:
                view = cv2.cvtColor(cv2.resize(frame, (960, 600)), cv2.COLOR_RGB2BGR)
                text = "no tags" if width_m is None else f"width {width_m * 1000:.1f} mm"
                if recording is not None:
                    text += f"  [recording {recording[0]}]"
                cv2.putText(view, text, (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow("CalibrateGripperMarkerWidth", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("o"):
                unwrapped.set_gripper_blocking(False)
            elif key == ord("c"):
                unwrapped.set_gripper_blocking(True)
            elif key in (ord("1"), ord("2"), ord("3")) and recording is None:
                label = {ord("1"): "open", ord("2"): "empty", ord("3"): "cup"}[key]
                recording = (label, time.time() + 2.0)
    finally:
        env.close()

    medians = {label: float(np.median(v)) for label, v in samples.items() if len(v) > 0}
    for label, values in samples.items():
        if values:
            print(
                f"{label:6s}: median {np.median(values) * 1000:.1f} mm, "
                f"min {np.min(values) * 1000:.1f}, max {np.max(values) * 1000:.1f} "
                f"({len(values)} samples)"
            )
    if len(medians) < 3:
        print(f"Need all three of open / empty / cup; got {sorted(medians)}. Nothing saved.")
        return
    if not medians["empty"] < medians["cup"] < medians["open"]:
        print("Expected empty < cup < open widths; check the tags / labels. Nothing saved.")
        return
    cup_min = float(np.min(samples["cup"]))
    cup_max = float(np.max(samples["cup"]))
    values = {
        "width_open_m": round(medians["open"], 4),
        "width_closed_m": round(medians["empty"], 4),
        # Halfway between the classes, from the closest cup reading.
        "width_min_m": round(0.5 * (medians["empty"] + cup_min), 4),
        "width_max_m": round(0.5 * (cup_max + medians["open"]), 4),
        "width_open_min_m": round(0.5 * (cup_max + medians["open"]), 4),
    }
    print("Thresholds:")
    for key, value in values.items():
        print(f"  {key}: {value}")
    if args.save_to is not None:
        set_yaml_values(args.save_to, values)
        print(f"Written to {args.save_to}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--env",
        type=str,
        default="RealUMIDemoEnv-v0",
        help="gym env id (without the robo_manip_baselines/ prefix); "
        "RealFairino5DemoEnv-v0 selects the interactive FR5 mode",
    )
    parser.add_argument(
        "--gripper_marker",
        type=str,
        default=None,
        help="FR5 mode: gripper_marker YAML with gripper_marker_config",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        default=None,
        help="FR5 mode: gripper_marker YAML to write the widths/thresholds into",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./envs/configs/RealUMIDemo.yaml",
        help="same config bin/Teleop.py RealUMIDemo uses -- must set "
        "gripper_marker_config and whichever camera_ids/pointcloud_camera_ids "
        "provides its camera_name",
    )
    args = parser.parse_args()

    if args.env.startswith("RealFairino5"):
        calibrate_fr5(args)
        raise SystemExit(0)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    gripper_marker_config = config.get("gripper_marker_config")
    if gripper_marker_config is None:
        raise RuntimeError(
            f"[{__file__}] {args.config} has no gripper_marker_config -- "
            "nothing to calibrate."
        )

    env = gym.make("robo_manip_baselines/RealUMIDemoEnv-v0", **config)
    unwrapped = env.unwrapped
    env.reset()

    aruco = parse_aruco_config(
        {
            "aruco_dict": gripper_marker_config["aruco_dict"],
            "marker_size_map": gripper_marker_config["marker_size_map"],
        }
    )
    camera_matrix = np.array(gripper_marker_config["camera_matrix"], dtype=np.float64)
    dist_coeffs = gripper_marker_config.get("dist_coeffs")
    dist_coeffs = None if dist_coeffs is None else np.array(dist_coeffs, dtype=np.float64)
    clahe_clip_limit = gripper_marker_config.get("clahe_clip_limit", 0.0)
    nominal_z = gripper_marker_config["nominal_z"]
    z_tolerance = gripper_marker_config.get("z_tolerance", 0.008)
    left_id = gripper_marker_config["left_marker_id"]
    right_id = gripper_marker_config["right_marker_id"]

    min_width_m = None
    max_width_m = None
    n_samples = 0
    n_misses = 0

    print(
        f"[{__name__}] Open/close the rig fully by hand (slowly AND quickly, "
        "several times). Press Ctrl+C to stop and print the calibration result."
    )
    try:
        while True:
            env.step(unwrapped.action_space.sample() * 0)
            frame = unwrapped.get_latest_rgb_camera_frame(
                gripper_marker_config["camera_name"]
            )
            if frame is None:
                continue

            detect_input = (
                preprocess_low_light_image(frame, clip_limit=clahe_clip_limit)
                if clahe_clip_limit > 0
                else frame
            )
            tag_dict = detect_aruco_tags(
                detect_input,
                aruco["aruco_dict"],
                aruco["marker_size_map"],
                camera_matrix,
                dist_coeffs=dist_coeffs,
                corner_refinement=aruco["corner_refinement"],
            )
            width_m = get_gripper_width(
                tag_dict,
                left_id=left_id,
                right_id=right_id,
                nominal_z=nominal_z,
                z_tolerance=z_tolerance,
            )
            if width_m is None:
                n_misses += 1
                continue

            n_samples += 1
            min_width_m = width_m if min_width_m is None else min(min_width_m, width_m)
            max_width_m = width_m if max_width_m is None else max(max_width_m, width_m)
            print(
                f"\r[{__name__}] width={width_m * 1000:6.1f} mm   "
                f"running min={min_width_m * 1000:6.1f} mm (-> width_closed_m)   "
                f"max={max_width_m * 1000:6.1f} mm (-> width_open_m)   "
                f"samples={n_samples} misses={n_misses}",
                end="",
                flush=True,
            )
    except KeyboardInterrupt:
        print()
    finally:
        env.close()

    if min_width_m is None:
        print(f"[{__name__}] No valid width samples were captured -- nothing to report.")
    else:
        print(
            f"\n[{__name__}] Result ({n_samples} samples, {n_misses} misses):\n"
            f"  width_open_m: {round(max_width_m, 3)}\n"
            f"  width_closed_m: {round(min_width_m, 3)}\n"
            f"Paste these into {args.config}'s gripper_marker_config if they "
            "differ meaningfully from the current values."
        )
