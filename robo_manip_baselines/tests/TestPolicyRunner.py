"""PolicyRunner: tick/buffer logic (stub policy), and -- when
RMB_TEST_DP_CHECKPOINT points at a trained image diffusion policy and the
Fairino SDK is importable -- that a runner produces exactly the commands the
existing RolloutDiffusionPolicy produces for the same observations."""

import os
import sys
import types
import unittest
from unittest import mock

import numpy as np

from robo_manip_baselines.common.base.PolicyRunnerBase import PolicyRunnerBase


class FakeRollout:
    """Stands in for a Rollout* class: records what the runner calls."""

    calls = []

    def setup_model_meta_info(self):
        self.model_meta_info = {"data": {"skip": 3}}
        self.calls.append("setup_model_meta_info")

    def setup_policy(self):
        self.calls.append("setup_policy")

    def setup_variables(self):
        self.calls.append("setup_variables")

    def reset_variables(self):
        self.buffer = []
        self.calls.append("reset_variables")

    def setup_eef_pose_retargeter(self):
        self.calls.append("setup_eef_pose_retargeter")

    def infer_policy(self):
        self.buffer.append(self.obs)
        self.calls.append(("infer", self.rollout_time_idx, self.obs))

    def set_command_data(self):
        self.calls.append(("command", self.rollout_time_idx))


class FakeRunner(PolicyRunnerBase):
    @classmethod
    def get_rollout_class(cls):
        return FakeRollout


def make_op(obs=0):
    import datetime

    return types.SimpleNamespace(
        env=None,
        motion_manager=None,
        demo_name="Task",
        datetime_now=datetime.datetime.now(),
        obs=obs,
        info={},
    )


class TestPolicyRunnerStub(unittest.TestCase):
    def setUp(self):
        FakeRollout.calls = []

    def test_load_and_tick_cadence(self):
        op = make_op()
        runner = FakeRunner("grasp", policy_skip=2).load("ckpt", op)
        self.assertEqual(
            FakeRollout.calls,
            [
                "setup_model_meta_info",
                "setup_policy",
                "setup_variables",
                "reset_variables",
            ],
        )
        self.assertEqual(runner.shell.args.skip, 2)
        self.assertEqual(runner.shell.args.checkpoint, "ckpt")
        self.assertEqual(runner.shell.demo_name, "Task_grasp")

        FakeRollout.calls = []
        for t in range(4):
            op.obs = t
            runner.infer_and_command(op)
        self.assertEqual(
            FakeRollout.calls,
            [
                ("infer", 0, 0),
                ("command", 0),
                ("command", 1),
                ("infer", 2, 2),
                ("command", 2),
                ("command", 3),
            ],
        )

    def test_default_skip_is_one_policy_step_per_tick(self):
        runner = FakeRunner("grasp").load("ckpt", make_op())
        self.assertEqual(runner.shell.args.skip, 1)

    def test_reset_buffers_restarts_inference(self):
        op = make_op()
        runner = FakeRunner("grasp").load("ckpt", op)
        runner.infer_and_command(op)
        runner.infer_and_command(op)
        FakeRollout.calls = []
        op.obs = 7
        runner.reset_buffers(op)
        self.assertEqual(runner.shell.buffer, [])
        self.assertEqual(runner.shell.rollout_time_idx, 0)
        self.assertIn("setup_eef_pose_retargeter", FakeRollout.calls)
        runner.infer_and_command(op)
        self.assertEqual(FakeRollout.calls[-2], ("infer", 0, 7))

    def test_hold_current_command(self):
        op = make_op()
        runner = FakeRunner("grasp").load("ckpt", op)
        runner.infer_and_command(op)
        with mock.patch(
            "robo_manip_baselines.common.base.PolicyRunnerBase.hold_at_measured"
        ) as hold:
            runner.hold_current_command(op)
            hold.assert_called_once_with(op, refresh_obs=False)
        self.assertEqual(runner.shell.buffer, [])


def _fairino_available():
    try:
        import fairino  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(
    os.environ.get("RMB_TEST_DP_CHECKPOINT") and _fairino_available(),
    "set RMB_TEST_DP_CHECKPOINT to an image diffusion policy checkpoint (and make "
    "the Fairino SDK importable) to run",
)
class TestDiffusionPolicyRunnerMatchesRollout(unittest.TestCase):
    def test_same_commands_as_rollout_diffusion_policy(self):
        import datetime

        import torch
        import yaml

        from robo_manip_baselines.common import DataKey, MotionManager
        from robo_manip_baselines.common.base.PolicyRunnerBase import make_policy_runner
        from robo_manip_baselines.envs.operation.OperationRealFairino5Demo import (
            OperationRealFairino5Demo,
        )
        from robo_manip_baselines.policy.diffusion_policy import RolloutDiffusionPolicy

        checkpoint = os.environ["RMB_TEST_DP_CHECKPOINT"]
        config_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "envs",
            "configs",
            "RealFairino5DemoEnv_SpaceMouse_DryRun.yaml",
        )
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        config["command_log_path"] = None

        class Rollout(OperationRealFairino5Demo, RolloutDiffusionPolicy):
            policy_name = "DiffusionPolicy"

        argv = ["Rollout.py", "--checkpoint", checkpoint, "--no_plot", "--no_render"]
        with mock.patch.object(sys, "argv", argv):
            rollout = Rollout(**config)
        rollout.obs, rollout.info = rollout.env.reset(seed=0)
        rollout.reset_variables()
        rollout.setup_eef_pose_retargeter()
        rollout.rollout_time_idx = 0

        op = types.SimpleNamespace(
            env=rollout.env,
            motion_manager=MotionManager(rollout.env),
            demo_name="Test",
            datetime_now=datetime.datetime.now(),
            obs=rollout.obs,
            info=rollout.info,
        )
        runner = make_policy_runner("grasp", checkpoint, op)
        runner.reset_buffers(op)

        camera_name = rollout.camera_names[0]
        rng = np.random.default_rng(0)
        base_obs = rollout.obs
        for tick in range(12):
            obs = {key: value.copy() for key, value in base_obs.items()}
            obs["joint_pos"][-1] = 100.0 * (tick > 6)
            obs["joint_pos"][:6] += 0.01 * tick
            info = {
                "rgb_images": {
                    camera_name: rng.integers(0, 255, (200, 320, 3), dtype=np.uint8)
                }
            }

            rollout.obs, rollout.info = obs, info
            rollout.motion_manager.sync_arm_to_measured(obs)
            torch.manual_seed(tick)
            if rollout.rollout_time_idx % rollout.args.skip == 0:
                with torch.inference_mode():
                    rollout.infer_policy()
            rollout.set_command_data()
            rollout.rollout_time_idx += 1

            op.obs, op.info = obs, info
            op.motion_manager.sync_arm_to_measured(obs)
            torch.manual_seed(tick)
            runner.infer_and_command(op)

            for key in (DataKey.COMMAND_EEF_POSE, DataKey.COMMAND_GRIPPER_JOINT_POS):
                np.testing.assert_allclose(
                    op.motion_manager.get_command_data(key),
                    rollout.motion_manager.get_command_data(key),
                    atol=1e-5,
                    err_msg=f"tick {tick}, {key}",
                )
        runner.close()
        rollout.env.close()


if __name__ == "__main__":
    unittest.main()
