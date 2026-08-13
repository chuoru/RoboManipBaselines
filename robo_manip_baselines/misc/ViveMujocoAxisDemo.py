"""Visualize a raw Vive tracker pose as an RGB coordinate-frame marker in a MuJoCo
scene, without commanding the robot arm.

Useful to check Vive position/orientation tracking accuracy (jitter, drift,
calibration correctness) before trusting it to drive the real IK-controlled arm
via ViveInputDevice -- see teleop/ViveInputDevice.py and
teleop/calibrate_vive_world_frame.py / teleop/calibrate_vive_rotation.py for the
vive_world_to_base_frame_rotation / vive_to_eef_frame_rotation calibration this
script reads from --input_device_config.

The displayed target frame is produced by calling ViveInputDevice.set_command_data()
directly (not a hand-copied re-implementation of its math -- an earlier version of
this script did that and silently went stale when set_command_data() was later
changed to incremental/TCP-local translation). set_command_data() does run inverse
kinematics internally (it updates arm_manager's own joint-angle state), but that
result is never fed into the MuJoCo simulation -- env.step() below always commands
the fixed home joint pose, so the rendered robot mesh stays visibly put regardless.
This isolates Vive's own tracking behavior (and the exact real transform code) from
any IK/arm-following error, without the risk of the display logic drifting out of
sync with teleop/ViveInputDevice.py again.

Usage:
    python ./misc/ViveMujocoAxisDemo.py MujocoFairino5Cable \\
        --input_device_config ./teleop/configs/Vive.yaml
"""

import argparse
import os
import threading
import time

import gymnasium as gym
import numpy as np
import yaml

import robo_manip_baselines.envs  # noqa: F401 -- registers gym env ids
from robo_manip_baselines.common import MotionManager
from robo_manip_baselines.teleop import ViveInputDevice

# Box-marker convention (see MujocoEnvBase.draw_box_marker / ArmManager.draw_markers):
# size is the half-extent along the box's own local X/Y/Z axes. We draw each axis
# bar as a box elongated along its own local X, then rotate that local X onto the
# target frame's X/Y/Z column to get the world-space bar direction.
AXIS_HALF_LEN = 0.08  # [m]
AXIS_HALF_THICKNESS = 0.004  # [m]

# ik_eef_joint_id's frame (what current_se3/home_se3 and the Vive-driven target are
# both anchored to) sits inside the gripper mount, occluded by the opaque robot
# mesh. Lift both displayed frames straight up by this much (world Z) so they clear
# the mesh and are actually visible, without changing any of the underlying pose
# math (this offset is display-only).
DISPLAY_Z_OFFSET = 0.12  # [m]

_IDENTITY = np.eye(3)
_ROT_X_TO_Y = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # Rz(+90deg)
_ROT_X_TO_Z = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])  # Ry(-90deg)

AXIS_SPEC = (
    ("x", _IDENTITY, (1.0, 0.0, 0.0, 1.0)),
    ("y", _ROT_X_TO_Y, (0.0, 1.0, 0.0, 1.0)),
    ("z", _ROT_X_TO_Z, (0.0, 0.0, 1.0, 1.0)),
)


def draw_axis_marker(env, se3, half_len=AXIS_HALF_LEN, rgba_scale=1.0):
    """Draw an RGB coordinate-frame gizmo (X=red, Y=green, Z=blue) at the given
    pinocchio SE3 (lifted by DISPLAY_Z_OFFSET so it clears the robot mesh -- see
    that constant's comment), as three elongated box markers extending outward
    from se3.translation along each column of se3.rotation."""
    origin = se3.translation + np.array([0.0, 0.0, DISPLAY_Z_OFFSET])
    size = (half_len, AXIS_HALF_THICKNESS, AXIS_HALF_THICKNESS)
    for axis_idx, (_axis_name, extra_rot, rgba) in enumerate(AXIS_SPEC):
        direction = se3.rotation[:, axis_idx]
        pos = origin + half_len * direction
        mat = se3.rotation @ extra_rot
        env.unwrapped.draw_box_marker(
            pos=pos,
            mat=mat,
            size=size,
            rgba=tuple(c * rgba_scale if c < 1.0 else c for c in rgba),
        )


def parse_argument():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "env", type=str, help="environment name, e.g. MujocoFairino5Cable"
    )
    parser.add_argument(
        "--input_device_config",
        type=str,
        required=True,
        help="Vive input device configuration file (device_params, pos_scale, "
        "vive_to_eef_frame_rotation, vive_world_to_base_frame_rotation)",
    )
    parser.add_argument("--world_idx", type=int, default=0)
    parser.add_argument(
        "--reference_axis_scale",
        type=float,
        default=1.4,
        help="size multiplier for the static home-pose reference frame, so it's "
        "visually distinguishable from the live Vive-driven frame",
    )
    return parser.parse_args()


def main():
    args = parse_argument()

    with open(args.input_device_config, "r") as f:
        device_kwargs = yaml.safe_load(f)

    device_params = device_kwargs["device_params"]
    pos_scale = device_kwargs.get("pos_scale", 1.0)
    vive_to_eef_frame_rotation = np.array(
        device_kwargs.get("vive_to_eef_frame_rotation", np.eye(3).tolist()),
        dtype=np.float64,
    )
    vive_world_to_base_frame_rotation = np.array(
        device_kwargs.get("vive_world_to_base_frame_rotation", np.eye(3).tolist()),
        dtype=np.float64,
    )

    env = gym.make(f"robo_manip_baselines/{args.env}Env-v0", render_mode="human")
    env.unwrapped.modify_world(world_idx=args.world_idx)
    env.reset()

    motion_manager = MotionManager(env)
    arm_manager = motion_manager.body_manager_list[0]
    # Static reference: the arm's home EEF pose. IK results are never fed into
    # env.step() in this demo (see the loop below), so the rendered arm/this
    # reference stay fixed for the whole run -- draw it once per frame as a
    # size-scaled frame to visually anchor scale/position against the live
    # Vive-driven frame.
    home_se3 = arm_manager.current_se3.copy()
    hold_action = np.concatenate(
        [
            arm_manager.body_config.init_arm_joint_pos,
            arm_manager.body_config.init_gripper_joint_pos,
        ]
    )

    vive = ViveInputDevice(
        arm_manager,
        device_params,
        pos_scale=pos_scale,
        vive_to_eef_frame_rotation=vive_to_eef_frame_rotation,
        vive_world_to_base_frame_rotation=vive_world_to_base_frame_rotation,
    )
    vive.connect()

    print(
        "[ViveMujocoAxisDemo] Waiting for the Vive tracker to be visible and to "
        "settle (hold it still for ~0.5s)...\n"
        "  - larger frame = arm's home EEF pose (static reference)\n"
        "  - smaller frame = ViveInputDevice's real target_se3 -- IK runs for real, "
        "but the result is never sent to the simulated arm\n"
        "Press Ctrl+C in this terminal to quit."
    )

    try:
        while True:
            vive.read()

            draw_axis_marker(
                env, home_se3, half_len=AXIS_HALF_LEN * args.reference_axis_scale
            )

            if vive.enabled_teleop and vive.state is not None:
                # Call the real ViveInputDevice code path (not a re-implementation
                # of its math) so this display can never drift out of sync with
                # teleop/ViveInputDevice.py again. This does run IK and updates
                # arm_manager's internal joint/target state, but that state is
                # never sent to env.step() below, so the rendered arm doesn't move.
                vive.set_command_data()
                draw_axis_marker(env, vive.arm_manager.target_se3)

            env.step(hold_action)
            time.sleep(env.unwrapped.dt)
    except KeyboardInterrupt:
        print("\n[ViveMujocoAxisDemo] Stopped.")
    finally:
        try:
            env.close()
        except Exception:
            pass
        # ViveInputDevice.close() -> pysurvive.simple_close() has been observed to
        # hang indefinitely on some rigs, even ignoring Ctrl+C (native blocking
        # call -- see the identical warning in
        # teleop/calibrate_vive_world_frame.py). Fire it in a daemon thread and
        # force-exit right after instead of waiting on it, so this script always
        # actually terminates on Ctrl+C.
        threading.Thread(target=vive.close, daemon=True).start()
        os._exit(0)


if __name__ == "__main__":
    main()
