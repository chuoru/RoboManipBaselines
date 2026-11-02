"""Check the SpaceMouse without the robot: open it with a teleop config and
print, live, the command it would send -- linear velocity [m/s] and angular
velocity [deg/s] in the frame the config commands in (robot base frame, or
the tool frame with control_frame: tool), the gripper command and the
precision mode.

The values go through exactly the processing teleop uses (deadzone,
smoothing, axis_sign / axis_enable, velocity scaling), so this is where to
check that the cap at rest reads 0 (no drift) and that pushing the cap
moves the target the expected way before the arm is involved. Fix the
directions with axis_sign in the teleop config.

Usage:
    python ./misc/CheckSpacemouse.py [--input_device_config ./teleop/configs/SpaceMouseRealFairino5.yaml]
Ctrl+C to quit.
"""

import argparse
import sys
import time

import numpy as np
import pinocchio as pin
import yaml

from robo_manip_baselines.teleop import SpacemouseInputDevice


class _DummyArmManager:
    """Stands in for ArmManager: holds the target pose and gripper command
    the input device writes to, starting from identity / open."""

    def __init__(self):
        self.target_se3 = pin.SE3.Identity()
        self.gripper_joint_pos = np.zeros(1)

    def set_command_eef_pose(self, target_se3):
        self.target_se3 = target_se3

    def get_command_gripper_joint_pos(self):
        return self.gripper_joint_pos

    def set_command_gripper_joint_pos(self, gripper_joint_pos):
        self.gripper_joint_pos = gripper_joint_pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input_device_config",
        type=str,
        default="./teleop/configs/SpaceMouseRealFairino5.yaml",
    )
    parser.add_argument("--rate", type=float, default=10.0, help="print rate [Hz]")
    args = parser.parse_args()

    with open(args.input_device_config, "r") as f:
        config = yaml.safe_load(f) or {}
    arm_manager = _DummyArmManager()
    device = SpacemouseInputDevice(arm_manager, **config)
    device.connect()
    print(f"Opened the SpaceMouse with {args.input_device_config}. Ctrl+C to quit.")
    # The dummy target starts at identity, so base and tool frame coincide
    # and the printed vectors are in the frame the config commands in.
    if device.control_frame == "tool":
        frame = "tool frame (x: image right, y: image down, z: towards the scene)"
    else:
        frame = "robot base frame"
    print(
        f"command in the {frame}:\n"
        "lin [m/s] x y z | ang [deg/s] x y z | gripper | precision | raw buttons"
    )

    period = 1.0 / args.rate
    prev_time = None
    try:
        while True:
            prev_se3 = arm_manager.target_se3.copy()
            device.read()
            device.set_command_data()
            now = time.time()
            dt = None if prev_time is None else now - prev_time
            prev_time = now
            if dt:
                lin_vel = (
                    arm_manager.target_se3.translation - prev_se3.translation
                ) / dt
                ang_vel = np.rad2deg(
                    pin.log3(arm_manager.target_se3.rotation @ prev_se3.rotation.T) / dt
                )
                sys.stdout.write(
                    "\rlin {:+.3f} {:+.3f} {:+.3f} | ang {:+6.1f} {:+6.1f} {:+6.1f} | "
                    "gripper {:5.1f} | precision {:3} | buttons {}   ".format(
                        *lin_vel,
                        *ang_vel,
                        float(np.mean(arm_manager.gripper_joint_pos)),
                        "ON" if device.precision_mode else "off",
                        list(device.state.buttons),
                    )
                )
                sys.stdout.flush()
            time.sleep(period)
    except KeyboardInterrupt:
        print()
