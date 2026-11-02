"""Record named arm poses (joint angles) for the segment runner.

Move the FR5 to a pose -- with the SpaceMouse (--input_device spacemouse),
or by hand / teach pendant with no input device -- and press SUCCESS (pedal),
Enter or 's' to store the current joint angles under the next name. The
poses are written to a pose YAML (envs/configs/poses/<Task>.yaml) in degrees.

    python ./misc/RecordNamedPose.py \\
        --config ./envs/configs/RealFairino5DemoEnv_SpaceMouse.yaml \\
        --poses_file ./envs/configs/poses/CupPickPlace.yaml \\
        --names home carry \\
        --input_device spacemouse

Keys (OpenCV window): Enter/'s' = record, 'k' = skip this name, Esc = quit.
With --input_device, the arm first moves to the env's reset pose (that is
what enables streaming commands); without one the arm is never commanded.
"""

import argparse
import os

import cv2
import gymnasium as gym
import numpy as np
import yaml

import robo_manip_baselines.envs  # noqa: F401 (registers the gym envs)
from robo_manip_baselines.common import MotionManager
from robo_manip_baselines.common.base.SegmentPhases import hold_at_measured
from robo_manip_baselines.common.utils.GlobalKeyListener import GlobalKeyListener


def write_pose_file(path, poses_deg):
    """Update `path` with {name: [deg x 6]}, keeping other entries."""
    config = {}
    header = []
    if os.path.exists(path):
        with open(path, "r") as f:
            text = f.read()
        config = yaml.safe_load(text) or {}
        # Keep the explanatory comment block at the top of the file.
        for line in text.splitlines():
            if line.startswith("#"):
                header.append(line)
            else:
                break
    unit = config.get("unit", "deg")
    poses = config.get("poses", {})
    for name, value_deg in poses_deg.items():
        value = value_deg if unit == "deg" else np.deg2rad(value_deg)
        poses[name] = [round(float(v), 3 if unit == "deg" else 6) for v in value]
    config = {"unit": unit, "poses": poses}
    with open(path, "w") as f:
        if header:
            f.write("\n".join(header) + "\n")
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=None)


class _Op:
    """Minimal op for hold_at_measured()."""

    def __init__(self, env, motion_manager, obs):
        self.env = env
        self.motion_manager = motion_manager
        self.obs = obs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=str, required=True, help="FR5 env config")
    parser.add_argument("--poses_file", type=str, required=True)
    parser.add_argument("--names", type=str, nargs="+", required=True)
    parser.add_argument(
        "--input_device",
        type=str,
        default=None,
        choices=["spacemouse", "keyboard"],
        help="move the arm with this device (default: none, move it by hand)",
    )
    parser.add_argument(
        "--input_device_config",
        type=str,
        default="./teleop/configs/SpaceMouseRealFairino5.yaml",
    )
    parser.add_argument(
        "--pedal_config", type=str, default="./common/configs/pedal_default.yaml"
    )
    parser.add_argument("--env", type=str, default="RealFairino5DemoEnv-v0")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    env = gym.make(
        f"robo_manip_baselines/{args.env}",
        **{"camera_ids": None, "gelsight_ids": None, **config},
    )
    obs, info = env.reset()
    motion_manager = MotionManager(env)
    op = _Op(env, motion_manager, obs)

    device = None
    if args.input_device is not None:
        kwargs = {}
        if args.input_device == "spacemouse":
            with open(args.input_device_config, "r") as f:
                kwargs = yaml.safe_load(f) or {}
        env.unwrapped.move_to_init_pose()
        hold_at_measured(op)
        device = env.unwrapped.setup_input_device(
            args.input_device, motion_manager, kwargs
        )[0]
        device.connect()

    listener = GlobalKeyListener.from_config(
        args.pedal_config, enabled_actions={"success"}
    )
    if not listener.start():
        listener = None

    arm_joint_idxes = env.unwrapped.body_config_list[0].arm_joint_idxes
    recorded = {}
    name_idx = 0
    try:
        while name_idx < len(args.names):
            if device is not None:
                motion_manager.sync_arm_to_measured(op.obs)
                device.read()
                device.set_command_data()
            action = np.concatenate(
                [
                    motion_manager.get_command_data(key)
                    for key in env.unwrapped.command_keys_for_step
                ]
            )
            op.obs, _, _, _, info = env.step(action)
            joint_deg = np.rad2deg(
                env.unwrapped.get_joint_pos_from_obs(op.obs)[arm_joint_idxes]
            )

            panel = np.full((140, 900, 3), 255, dtype=np.uint8)
            cv2.putText(
                panel,
                f"next: {args.names[name_idx]}  (Enter/s: record, k: skip, Esc: quit)",
                (10, 40),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (0, 0, 0),
                1,
            )
            cv2.putText(
                panel,
                "joints [deg]: " + ", ".join(f"{v:.1f}" for v in joint_deg),
                (10, 90),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (0, 0, 0),
                1,
            )
            cv2.imshow("RecordNamedPose", panel)
            key = cv2.waitKey(1)
            pedal = listener.poll() if listener is not None else set()
            if key == 27:
                break
            if key in (13, 10, ord("s")) or "success" in pedal:
                name = args.names[name_idx]
                recorded[name] = joint_deg.copy()
                print(f"Recorded '{name}': {np.round(joint_deg, 2).tolist()} deg")
                write_pose_file(args.poses_file, {name: joint_deg})
                name_idx += 1
            elif key == ord("k"):
                print(f"Skipped '{args.names[name_idx]}'")
                name_idx += 1
    finally:
        if listener is not None:
            listener.stop()
        if device is not None:
            device.close()
        env.close()
    print(f"Poses written to {args.poses_file}: {sorted(recorded)}")
