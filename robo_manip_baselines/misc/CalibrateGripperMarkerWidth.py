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
"""

import argparse

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="./envs/configs/RealUMIDemo.yaml",
        help="same config bin/Teleop.py RealUMIDemo uses -- must set "
        "gripper_marker_config and whichever camera_ids/pointcloud_camera_ids "
        "provides its camera_name",
    )
    args = parser.parse_args()

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
