"""make_primitive_phase: the {teleop: ...} step and the load-time check of
named poses."""

import types
import unittest

import numpy as np

from robo_manip_baselines.common.base.SegmentPhases import (
    FreeTeleopPhase,
    JointMovePhase,
    make_primitive_phase,
)

POSES = {"home": "init", "carry": np.zeros(6), "fixture": None}


class TeleopOp:
    SUPPORTS_TELEOP_STEPS = True


class FakeDevice:
    def __init__(self):
        self.calls = []

    def connect(self):
        self.calls.append("connect")

    def reset_filter(self):
        self.calls.append("reset_filter")

    def read(self):
        self.calls.append("read")

    def set_command_data(self):
        self.calls.append("set_command_data")


class TestSegmentPhases(unittest.TestCase):
    def test_joint_pos_is_checked_when_loaded(self):
        op = types.SimpleNamespace()
        self.assertIsInstance(
            make_primitive_phase(op, {"joint_pos": "carry"}, POSES), JointMovePhase
        )
        self.assertIsInstance(
            make_primitive_phase(op, {"joint_pos": "home"}, POSES), JointMovePhase
        )
        with self.assertRaises(ValueError):
            make_primitive_phase(op, {"joint_pos": "fixture"}, POSES)
        with self.assertRaises(KeyError):
            make_primitive_phase(op, {"joint_pos": "nowhere"}, POSES)

    def test_teleop_step_only_in_demo_collection(self):
        with self.assertRaises(ValueError):
            make_primitive_phase(types.SimpleNamespace(), {"teleop": "x"}, POSES)
        op = TeleopOp()
        phase = make_primitive_phase(op, {"teleop": "grasp the cup"}, POSES)
        self.assertIsInstance(phase, FreeTeleopPhase)
        self.assertEqual(phase.display_name, "grasp the cup")

    def test_teleop_step_drives_the_devices_until_n(self):
        op = TeleopOp()
        device = FakeDevice()
        op.input_device_list = [device]
        op.key = -1
        # PhaseBase.start() reads the env time
        op.env = types.SimpleNamespace(
            unwrapped=types.SimpleNamespace(get_time=lambda: 0.0)
        )
        phase = make_primitive_phase(op, {"teleop": "move the cup"}, POSES)
        phase.start()
        self.assertEqual(device.calls, ["connect", "reset_filter"])
        phase.pre_update()
        self.assertEqual(device.calls[2:], ["read", "set_command_data"])
        self.assertFalse(phase.check_transition())
        op.key = ord("n")
        self.assertTrue(phase.check_transition())


if __name__ == "__main__":
    unittest.main()
