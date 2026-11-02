import sys
import types
import unittest
from unittest import mock

import numpy as np
import pinocchio as pin

from robo_manip_baselines.teleop.SpacemouseInputDevice import SpacemouseInputDevice


class FakeArmManager:
    def __init__(self):
        self.target_se3 = pin.SE3.Identity()
        self.gripper = np.zeros(1)

    def set_command_eef_pose(self, se3):
        self.target_se3 = se3

    def get_command_gripper_joint_pos(self):
        return self.gripper

    def set_command_gripper_joint_pos(self, gripper):
        self.gripper = np.clip(gripper, 0.0, 100.0)


def make_state(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0, buttons=(0, 0)):
    return types.SimpleNamespace(
        x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, buttons=list(buttons)
    )


def make_device(**kwargs):
    arm = FakeArmManager()
    device = SpacemouseInputDevice(arm, **kwargs)
    device.connected = True
    return device, arm


class TestSpacemouseInputDevice(unittest.TestCase):
    def test_default_behaviour_is_unchanged(self):
        device, arm = make_device()
        device.state = make_state(x=0.5, y=-1.0, z=0.2, buttons=(1, 0))
        device.set_command_data()
        np.testing.assert_allclose(
            arm.target_se3.translation, 1e-2 * np.array([1.0, 0.5, 0.2])
        )
        # incremental gripper: button 0 closes by gripper_scale per tick
        np.testing.assert_allclose(arm.gripper, [5.0])
        device.set_command_data()
        np.testing.assert_allclose(arm.gripper, [10.0])
        device.state = make_state(buttons=(0, 1))
        device.set_command_data()
        np.testing.assert_allclose(arm.gripper, [5.0])

    def test_deadzone_and_axis_mask(self):
        device, arm = make_device(
            deadzone=0.1, axis_enable=[1, 0, 1, 1, 1, 1], axis_sign=[-1, 1, 1, 1, 1, 1]
        )
        device.state = make_state(x=0.05, y=0.9, z=0.5)
        device.set_command_data()
        # x below deadzone, y disabled, z passes
        np.testing.assert_allclose(arm.target_se3.translation, [0.0, 0.0, 5e-3])
        device.state = make_state(x=0.5)
        device.set_command_data()
        np.testing.assert_allclose(arm.target_se3.translation, [0.0, -5e-3, 5e-3])

    def test_smoothing_and_reset_filter(self):
        device, arm = make_device(smoothing_alpha=0.5)
        device.state = make_state(z=1.0)
        device.set_command_data()  # first sample passes through
        self.assertAlmostEqual(arm.target_se3.translation[2], 1e-2)
        device.state = make_state(z=0.0)
        device.set_command_data()  # EMA: 0.5
        self.assertAlmostEqual(arm.target_se3.translation[2], 1.5e-2)
        self.assertAlmostEqual(device.input_norm(), 0.5)
        device.reset_filter()
        self.assertEqual(device.input_norm(), 0.0)
        device.set_command_data()
        self.assertAlmostEqual(arm.target_se3.translation[2], 1.5e-2)

    def test_toggle_gripper_on_rising_edge(self):
        device, arm = make_device(gripper_mode="toggle", gripper_button=0)
        # Held while connecting: must not toggle on the first read.
        device.state = make_state(buttons=(1, 0))
        device.set_command_data()
        np.testing.assert_allclose(arm.gripper, [0.0])
        device.state = make_state(buttons=(0, 0))
        device.set_command_data()
        device.state = make_state(buttons=(1, 0))
        device.set_command_data()
        np.testing.assert_allclose(arm.gripper, [100.0])
        device.set_command_data()  # still held: no second toggle
        np.testing.assert_allclose(arm.gripper, [100.0])
        device.state = make_state(buttons=(0, 0))
        device.set_command_data()
        device.state = make_state(buttons=(1, 0))
        device.set_command_data()
        np.testing.assert_allclose(arm.gripper, [0.0])

    def test_precision_mode(self):
        device, arm = make_device(precision_button=1, precision_scale=0.25)
        device.state = make_state(buttons=(0, 0))
        device.set_command_data()
        device.state = make_state(z=1.0, buttons=(0, 1))
        device.set_command_data()
        self.assertTrue(device.precision_mode)
        self.assertAlmostEqual(arm.target_se3.translation[2], 0.25e-2)

    def test_velocity_mode_uses_elapsed_time(self):
        device, arm = make_device(max_lin_vel=0.1, max_dt=0.25)
        device.state = make_state(z=1.0)
        with mock.patch("time.time", side_effect=[100.0, 100.2, 101.0]):
            device.set_command_data()  # first call: no elapsed time yet
            self.assertAlmostEqual(arm.target_se3.translation[2], 0.0)
            device.set_command_data()  # 0.2 s at 0.1 m/s
            self.assertAlmostEqual(arm.target_se3.translation[2], 0.02)
            device.set_command_data()  # 0.8 s gap is capped at max_dt
            self.assertAlmostEqual(arm.target_se3.translation[2], 0.045)

    def test_tool_frame_rotation(self):
        device, arm = make_device(rotation_frame="tool", rpy_scale=0.1)
        arm.target_se3 = pin.SE3(pin.rpy.rpyToMatrix(0.0, 0.0, np.pi / 2), np.zeros(3))
        device.state = make_state(roll=-1.0)  # +0.1 rad about the tool x axis
        device.set_command_data()
        expected = pin.rpy.rpyToMatrix(0.0, 0.0, np.pi / 2) @ pin.rpy.rpyToMatrix(
            0.1, 0.0, 0.0
        )
        np.testing.assert_allclose(arm.target_se3.rotation, expected, atol=1e-12)

    def test_tool_control_frame_translation_follows_the_view(self):
        device, arm = make_device(control_frame="tool", pos_scale=0.1)
        # fingers pointing horizontally along base +x, tool y (image down)
        # pointing down: tool x = base -y, tool y = base -z, tool z = base +x
        rotation = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        arm.target_se3 = pin.SE3(rotation, np.array([0.3, 0.0, 0.2]))
        # cap forward = TCP forward (tool +z), right = image right (tool +x),
        # up = image up (tool -y)
        for state, base_dir in (
            (make_state(y=1.0), [1.0, 0.0, 0.0]),
            (make_state(x=1.0), [0.0, -1.0, 0.0]),
            (make_state(z=1.0), [0.0, 0.0, 1.0]),
        ):
            start = arm.target_se3.translation.copy()
            device.state = state
            device.set_command_data()
            np.testing.assert_allclose(
                arm.target_se3.translation - start,
                0.1 * np.array(base_dir),
                atol=1e-12,
            )
            np.testing.assert_allclose(arm.target_se3.rotation, rotation, atol=1e-12)

    def test_tool_control_frame_rotation_pivots_about_the_tcp(self):
        rotation = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        # same signs as the world mapping, whose twist turns about base +z:
        # twist = yaw about the TCP's up axis (tool -y = base +z here), tilt
        # left/right = roll about the fingers (tool +z), tilt forward/back =
        # pitch about the TCP's right axis (tool +x)
        for state, tool_axis in (
            (make_state(yaw=0.5), [0.0, 1.0, 0.0]),
            (make_state(roll=1.0), [0.0, 0.0, 1.0]),
            (make_state(pitch=-1.0), [1.0, 0.0, 0.0]),
        ):
            device, arm = make_device(control_frame="tool", rpy_scale=0.1)
            arm.target_se3 = pin.SE3(rotation, np.array([0.3, 0.1, 0.2]))
            device.state = state
            device.set_command_data()
            # the controlled point (TCP) stays put
            np.testing.assert_allclose(arm.target_se3.translation, [0.3, 0.1, 0.2])
            expected = rotation @ pin.exp3(0.1 * np.array(tool_axis))
            np.testing.assert_allclose(arm.target_se3.rotation, expected, atol=1e-12)
        # the twist turns the same way as in the world mapping when the TCP's
        # up axis is the base +z
        world_device, world_arm = make_device(rpy_scale=0.1)
        world_arm.target_se3 = pin.SE3(rotation, np.zeros(3))
        world_device.state = make_state(yaw=0.5)
        world_device.set_command_data()
        np.testing.assert_allclose(
            world_arm.target_se3.rotation,
            rotation @ pin.exp3(np.array([0.0, 0.1, 0.0])),
            atol=1e-12,
        )

    def test_axis_scale(self):
        device, arm = make_device(rpy_scale=0.1, axis_scale=[1, 1, 1, 1, 1, 0.5])
        device.state = make_state(yaw=1.0)
        device.set_command_data()
        np.testing.assert_allclose(
            pin.log3(arm.target_se3.rotation), [0.0, 0.0, -0.1], atol=1e-12
        )

    def test_tool_view_yaw(self):
        device, arm = make_device(
            control_frame="tool", pos_scale=0.1, tool_view_yaw_deg=90.0
        )
        device.state = make_state(x=1.0)  # image right = tool +y after 90 deg
        device.set_command_data()
        np.testing.assert_allclose(
            arm.target_se3.translation, [0.0, 0.1, 0.0], atol=1e-12
        )

    def test_connect_without_device_params_and_failure(self):
        fake = types.ModuleType("pyspacemouse")
        fake.open = mock.Mock(return_value=object())
        with mock.patch.dict(sys.modules, {"pyspacemouse": fake}):
            device = SpacemouseInputDevice(FakeArmManager())
            device.connect()
            fake.open.assert_called_once_with()
            self.assertTrue(device.connected)

            fake.open = mock.Mock(return_value=None)
            device = SpacemouseInputDevice(
                FakeArmManager(), device_params={"device": "SpaceMouse Compact"}
            )
            with self.assertRaisesRegex(RuntimeError, "hidapi"):
                device.connect()
            fake.open.assert_called_once_with(device="SpaceMouse Compact")
            self.assertFalse(device.connected)


if __name__ == "__main__":
    unittest.main()
