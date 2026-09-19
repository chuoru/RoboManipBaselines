"""Live visual debugging tool for ArUco gripper-marker detection (see
common/utils/ArucoGripperUtils.py, RealUMIEnvBase._get_obs()): shows the
"hand" camera feed with detected marker outlines/IDs and the computed
gripper width overlaid in real time, in its own OpenCV window. Uses the
SAME gym env / config-loading / detection code path production teleop does
(RealUMIDemoEnv + envs/configs/RealUMIDemo.yaml's gripper_marker_config), so
what's shown here matches what teleop actually sees -- not a separate
reimplementation.

Meant for checking marker placement, lighting, and detection quality (does
CLAHE need to be stronger/weaker? is a marker too small/blurry/angled to
read reliably?) BEFORE or independent of running a full teleop session,
since bin/Teleop.py's own camera panel is a small multi-camera thumbnail
with no detection overlay.

Also prints/overlays brightness (mean gray level) and sharpness (Laplacian
variance) each frame -- concrete numbers for "is it too dark/blurry here",
rather than judging by eye alone -- and, for an Orbbec Gemini camera
(pointcloud_camera_ids), can force manual exposure/gain instead of relying
on auto-exposure + CLAHE alone (--exposure/--gain/--auto_exposure), since
auto-exposure not compensating enough in dim rooms -- not just insufficient
post-hoc contrast stretching -- was suspected after detection was reported
to fail completely in slightly darker spots.

Usage:
    python ./misc/ViewGripperMarkerDetection.py \\
        --config ./envs/configs/RealUMIDemo.yaml
    # Force manual exposure/gain instead of auto-exposure (Orbbec only):
    python ./misc/ViewGripperMarkerDetection.py \\
        --config ./envs/configs/RealUMIDemo.yaml --exposure 300 --gain 32
Press 'q' or Esc to quit, 's' to save the current raw/enhanced/overlay
frames to ./gripper_marker_snapshots/ for later analysis.
"""

import argparse
import os
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

SNAPSHOT_DIR = "./gripper_marker_snapshots"

# Panel is shown at this width regardless of the camera's native
# resolution, upscaled with INTER_NEAREST so individual marker
# pixels/modules stay crisp instead of blurring -- the whole point here is
# to visually judge detection quality close up.
DISPLAY_WIDTH = 960


def compute_image_quality(image):
    """(mean brightness [0-255], sharpness [Laplacian variance, higher =
    sharper/more in-focus, roughly >100 is usually fine for ArUco, low
    single digits means heavily blurred]) -- concrete numbers instead of
    judging brightness/focus by eye."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return brightness, sharpness


def draw_overlay(image, tag_dict, config, width_m, percent_closed, quality=None):
    display = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else cv2.cvtColor(
        image, cv2.COLOR_GRAY2BGR
    )

    for marker_id, tag in tag_dict.items():
        corners = tag["corners"].astype(int)
        is_gripper_marker = marker_id in (
            config["left_marker_id"],
            config["right_marker_id"],
        )
        color = (0, 255, 0) if is_gripper_marker else (0, 165, 255)
        cv2.polylines(display, [corners], isClosed=True, color=color, thickness=2)
        centroid = corners.mean(axis=0).astype(int)
        cv2.putText(
            display,
            f"id={marker_id} z={tag['tvec'][2]:.3f}m",
            tuple(centroid),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    left_id, right_id = config["left_marker_id"], config["right_marker_id"]
    if left_id in tag_dict and right_id in tag_dict:
        p1 = tuple(tag_dict[left_id]["corners"].mean(axis=0).astype(int))
        p2 = tuple(tag_dict[right_id]["corners"].mean(axis=0).astype(int))
        cv2.line(display, p1, p2, (255, 0, 0), 2)

    status_lines = [
        f"detected markers: {sorted(tag_dict.keys())}",
        (
            f"width: {width_m * 1000:.1f} mm  ->  {percent_closed:.1f}% closed"
            if width_m is not None
            else "width: N/A (need both/either gripper marker in view)"
        ),
    ]
    if quality is not None:
        brightness, sharpness = quality
        status_lines.append(
            f"brightness: {brightness:.1f}/255   sharpness: {sharpness:.0f} "
            "(low light or blur -> detection fails first)"
        )
    # Solid backing bar behind the text: the raw scene behind it is often
    # bright/white (as here, over a wooden table), which otherwise washes
    # out plain white text.
    cv2.rectangle(display, (0, 0), (display.shape[1], 24 + 28 * len(status_lines)), (0, 0, 0), -1)
    for i, line in enumerate(status_lines):
        cv2.putText(
            display,
            line,
            (10, 30 + 28 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    aspect = display.shape[0] / display.shape[1]
    return cv2.resize(
        display,
        (DISPLAY_WIDTH, int(DISPLAY_WIDTH * aspect)),
        interpolation=cv2.INTER_NEAREST,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="./envs/configs/RealUMIDemo.yaml",
        help="same config file bin/Teleop.py RealUMIDemo uses -- must set "
        "gripper_marker_config and whichever camera_ids/pointcloud_camera_ids "
        "provides gripper_marker_config's camera_name",
    )
    parser.add_argument(
        "--auto_exposure",
        type=int,
        choices=[0, 1],
        default=None,
        help="Orbbec only: force OB_PROP_COLOR_AUTO_EXPOSURE_BOOL on(1)/off(0). "
        "Leave unset to keep the camera's current setting.",
    )
    parser.add_argument(
        "--exposure",
        type=int,
        default=None,
        help="Orbbec only: manual OB_PROP_COLOR_EXPOSURE_INT. Implies "
        "--auto_exposure 0 unless that's explicitly set. Try raising this "
        "if brightness stays low even with --gain raised and CLAHE maxed "
        "out -- exposure recovers real signal, CLAHE can only stretch "
        "whatever contrast the sensor actually captured.",
    )
    parser.add_argument(
        "--gain",
        type=int,
        default=None,
        help="Orbbec only: manual OB_PROP_COLOR_GAIN_INT. Higher = brighter "
        "but noisier.",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    gripper_marker_config = config.get("gripper_marker_config")
    if gripper_marker_config is None:
        raise RuntimeError(
            f"[{__file__}] {args.config} has no gripper_marker_config -- "
            "nothing to visualize."
        )

    env = gym.make("robo_manip_baselines/RealUMIDemoEnv-v0", **config)
    unwrapped = env.unwrapped
    env.reset()

    camera_name = gripper_marker_config["camera_name"]
    pointcloud_camera = unwrapped.pointcloud_cameras.get(camera_name)
    if pointcloud_camera is not None and (
        args.exposure is not None or args.gain is not None or args.auto_exposure is not None
    ):
        from pyorbbecsdk import OBPropertyID

        device = pointcloud_camera["device"]
        auto_exposure = args.auto_exposure
        if auto_exposure is None and args.exposure is not None:
            auto_exposure = 0  # manual exposure requires auto-exposure off
        if auto_exposure is not None:
            device.set_bool_property(
                OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, bool(auto_exposure)
            )
        if args.exposure is not None:
            device.set_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, args.exposure)
        if args.gain is not None:
            device.set_int_property(OBPropertyID.OB_PROP_COLOR_GAIN_INT, args.gain)
        print(
            f"[{__name__}] Camera '{camera_name}' properties -- "
            f"auto_exposure={device.get_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL)} "
            f"exposure={device.get_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT)} "
            f"gain={device.get_int_property(OBPropertyID.OB_PROP_COLOR_GAIN_INT)}"
        )

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
    width_open_m = gripper_marker_config["width_open_m"]
    width_closed_m = gripper_marker_config["width_closed_m"]

    print(
        f"[{__name__}] Showing live detection. Press 'q' or Esc to quit, "
        "'s' to save a snapshot."
    )
    try:
        while True:
            env.step(unwrapped.action_space.sample() * 0)
            frame = unwrapped.get_latest_rgb_camera_frame(
                gripper_marker_config["camera_name"]
            )
            if frame is None:
                continue

            quality = compute_image_quality(frame)
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
                left_id=gripper_marker_config["left_marker_id"],
                right_id=gripper_marker_config["right_marker_id"],
                nominal_z=nominal_z,
                z_tolerance=z_tolerance,
            )
            percent_closed = (
                float(
                    np.clip(
                        100.0 * (width_open_m - width_m) / (width_open_m - width_closed_m),
                        0.0,
                        100.0,
                    )
                )
                if width_m is not None
                else None
            )

            display = draw_overlay(
                frame, tag_dict, gripper_marker_config, width_m, percent_closed, quality
            )
            cv2.imshow("Gripper marker detection", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                os.makedirs(SNAPSHOT_DIR, exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(
                    os.path.join(SNAPSHOT_DIR, f"{stamp}_raw.png"),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    os.path.join(SNAPSHOT_DIR, f"{stamp}_enhanced.png"), detect_input
                )
                cv2.imwrite(
                    os.path.join(SNAPSHOT_DIR, f"{stamp}_overlay.png"), display
                )
                print(
                    f"[{__name__}] Saved snapshot {stamp} (brightness="
                    f"{quality[0]:.1f}, sharpness={quality[1]:.0f}, "
                    f"detected={sorted(tag_dict.keys())}) to {SNAPSHOT_DIR}/"
                )
    finally:
        cv2.destroyAllWindows()
        env.close()
