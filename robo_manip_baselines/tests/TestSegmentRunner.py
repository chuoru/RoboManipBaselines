"""Segment runner control flow on the FR5 env in dry run (no hardware; the
Fairino SDK import is stubbed if it is not installed): end conditions, the
grasp/lift/release checks, retries, stopping for the operator, labels of the
recorded attempts, exhibit-mode pedal filtering."""

import os
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import yaml

if "fairino" not in sys.modules:
    try:
        import fairino  # noqa: F401
    except ImportError:
        fake_fairino = types.ModuleType("fairino")
        fake_fairino.Robot = types.SimpleNamespace(RPC=None)
        sys.modules["fairino"] = fake_fairino

import cv2  # noqa: E402

from robo_manip_baselines.common import DataKey, RmbData, find_rmb_files  # noqa: E402
from robo_manip_baselines.common.base.SegmentRunnerBase import (  # noqa: E402
    SegmentRunnerBase,
    WaitForOperatorPhase,
)
from robo_manip_baselines.envs.operation.OperationRealFairino5Demo import (  # noqa: E402
    OperationRealFairino5Demo,
)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "envs", "configs")

# Finger-tag widths [m] the fake tracker reports per check outcome.
WIDTHS = {"grasped": 0.07, "empty": 0.02, "released": 0.12, "hidden": None}


class FakeOperator:
    """Closes / opens the gripper (toggle) after a few ticks of moving down."""

    def __init__(self, arm_manager, toggle_tick=4):
        self.arm_manager = arm_manager
        self.toggle_tick = toggle_tick
        self.t = 0
        self.connected = True

    def connect(self):
        pass

    def close(self):
        pass

    def reset_filter(self):
        self.t = 0

    def input_norm(self):
        return 1.0

    def read(self):
        pass

    def set_command_data(self):
        self.t += 1
        se3 = self.arm_manager.target_se3.copy()
        if self.t < self.toggle_tick:
            se3.translation[2] -= 0.003
        self.arm_manager.set_command_eef_pose(se3)
        if self.t == self.toggle_tick:
            gripper = self.arm_manager.get_command_gripper_joint_pos().copy()
            gripper[:] = 0.0 if gripper[0] >= 50 else 100.0
            self.arm_manager.set_command_gripper_joint_pos(gripper)


class FakePolicyRunner:
    """Policy stand-in: does not move; optionally flips the gripper command
    after `toggle_after` inference ticks."""

    def __init__(self, op, toggle_after=None):
        self.op = op
        self.toggle_after = toggle_after
        self.calls = []
        self.ticks = 0
        self.checkpoint = "fake.ckpt"

    def reset_buffers(self, op):
        self.calls.append("reset_buffers")
        self.ticks = 0

    def reset_action_buffers(self, op):
        self.calls.append("reset_action_buffers")

    def observe(self, op):
        self.calls.append("observe")

    def infer_and_command(self, op):
        self.calls.append("infer")
        self.ticks += 1
        if self.toggle_after is not None and self.ticks == self.toggle_after:
            arm_manager = op.motion_manager.body_manager_list[0]
            gripper = arm_manager.get_command_gripper_joint_pos().copy()
            gripper[:] = 0.0 if gripper[0] >= 50 else 100.0
            arm_manager.set_command_gripper_joint_pos(gripper)

    def hold_current_command(self, op):
        self.calls.append("hold")

    def close(self):
        pass


class FakeTracker:
    """Reports scripted widths: each check consumes the next outcome."""

    async_detection = True
    camera_name = "hand"

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.checks = []

    def get_raw_widths_since(self, since_time):
        outcome = self.outcomes.pop(0) if self.outcomes else "grasped"
        self.checks.append(outcome)
        return [WIDTHS[outcome]] * 5

    def get_reading(self):
        return {
            "width_m": 0.07,
            "percent_closed": 50.0,
            "detected": True,
            "raw_width_m": 0.07,
            "time": 0.0,
        }

    def stop(self):
        pass


class TestSegmentRunner(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.cwd = os.getcwd()
        os.chdir(self.tmp_dir)  # logs/ goes here
        marker_path = os.path.join(self.tmp_dir, "marker.yaml")
        with open(os.path.join(CONFIG_DIR, "gripper_marker", "CupPickPlace.yaml")) as f:
            marker = yaml.safe_load(f)
        marker["checks"].update(
            settle_sec=0.0,
            window_sec=0.0,
            grasp={"width_min_m": 0.04, "width_max_m": 0.10},
            release={"width_open_min_m": 0.10},
        )
        with open(marker_path, "w") as f:
            yaml.safe_dump(marker, f)
        with open(os.path.join(CONFIG_DIR, "segments", "CupPickPlace.yaml")) as f:
            task = yaml.safe_load(f)
        task["poses"] = os.path.abspath(
            os.path.join(CONFIG_DIR, "poses", "DryRunExample.yaml")
        )
        task["gripper_marker"] = marker_path
        for spec in task["segments"]:
            if spec["type"] == "learned":
                spec["timeout_s"] = 5.0
        self.task = task
        with open(
            os.path.join(CONFIG_DIR, "RealFairino5DemoEnv_SpaceMouse_DryRun.yaml")
        ) as f:
            self.env_config = yaml.safe_load(f)
        self.env_config["command_log_path"] = None
        self.dataset_dir = os.path.join(self.tmp_dir, "dataset")

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(self.tmp_dir)

    def run_runner(
        self,
        outcomes,
        extra_args=(),
        keys=None,
        max_ticks=3000,
        tweak_task=None,
        operator_kwargs=None,
        on_tick=None,
        policies=None,
    ):
        """policies: None = every learned segment driven by the (fake)
        operator; else {segment: FakePolicyRunner kwargs} with fake policies
        and --enable_intervention."""
        task = yaml.safe_load(yaml.safe_dump(self.task))
        if tweak_task is not None:
            tweak_task(task)
        task_path = os.path.join(self.tmp_dir, "task.yaml")
        with open(task_path, "w") as f:
            yaml.safe_dump(task, f)
        tracker = FakeTracker(outcomes)
        self.fake_policies = {}

        def make_fake_policy(segment_name, checkpoint, op, **_):
            runner = FakePolicyRunner(op, **policies.get(segment_name, {}))
            self.fake_policies[segment_name] = runner
            return runner

        class Runner(OperationRealFairino5Demo, SegmentRunnerBase):
            def setup_teleop_device(self):
                self.teleop_device = FakeOperator(
                    self.motion_manager.body_manager_list[0], **(operator_kwargs or {})
                )

            def setup_checks(self):
                real_tracker = self.env.unwrapped.gripper_marker_tracker
                if real_tracker is not None:
                    real_tracker.stop()
                self.env.unwrapped.gripper_marker_tracker = tracker
                super().setup_checks()

        key_queue = list(keys or [])
        state = {"ticks": 0, "runner": None}

        def wait_key(*_):
            state["ticks"] += 1
            runner = state["runner"]
            if on_tick is not None:
                on_tick(runner)
            if state["ticks"] > max_ticks:
                return 27
            if key_queue and key_queue[0][0](runner):
                return ord(key_queue.pop(0)[1])
            return -1

        if policies is None:
            mode_args = ["--replace_learned_with_teleop"]
        else:
            # Any existing file passes the checkpoint existence check.
            mode_args = [
                "--enable_intervention",
                "--checkpoint",
                f"grasp={task_path}",
                f"place={task_path}",
            ]
        argv = [
            "RolloutSegments.py",
            "--task",
            task_path,
            *mode_args,
            "--no_pedal",
            "--control_period",
            "0.0",
            "--save_rollout",
            "--dataset_dir",
            self.dataset_dir,
            *extra_args,
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(
            cv2, "imshow"
        ), mock.patch.object(cv2, "waitKey", side_effect=wait_key), mock.patch(
            "builtins.print"
        ), mock.patch(
            "robo_manip_baselines.common.base.SegmentRunnerBase.make_policy_runner",
            side_effect=make_fake_policy,
        ):
            runner = Runner(**self.env_config)
            state["runner"] = runner
            runner.run()
        return runner, tracker

    @staticmethod
    def in_run(name, min_ticks=0):
        """Key condition: segment `name` has been running for min_ticks."""

        def condition(runner):
            phase = runner.phase
            return (
                getattr(phase, "label", None) == name
                and getattr(phase, "state", None) == "run"
                and len(runner.data_manager.all_data_seq.get(DataKey.TIME, []))
                >= min_ticks
            )

        return condition

    @staticmethod
    def intervening(name, flag=True):
        def condition(runner):
            phase = runner.phase
            return getattr(phase, "label", None) == name and (
                getattr(phase, "is_intervening", False) == flag
            )

        return condition

    @staticmethod
    def start_key():
        return (lambda runner: runner.phase.label == "start", "n")

    def saved_episodes(self):
        episodes = {}
        for path in find_rmb_files(self.dataset_dir):
            with RmbData(path) as rmb_data:
                episodes[os.path.basename(path)] = dict(rmb_data.attrs)
                episodes[os.path.basename(path)]["_len"] = len(
                    rmb_data[DataKey.TIME][:]
                )
                episodes[os.path.basename(path)]["_keys"] = list(rmb_data.keys())
        return episodes

    def test_full_cycle_with_checks(self):
        runner, tracker = self.run_runner(
            ["grasped", "grasped", "released"],
            ["--num_cycles", "1"],
            keys=[self.start_key()],
        )
        self.assertEqual(tracker.checks, ["grasped", "grasped", "released"])
        self.assertEqual(runner.stats["cycles"], 1)
        self.assertEqual(runner.stats["cycles_without_stop"], 1)
        self.assertEqual(runner.segment_stats["grasp"]["success_first_try"], 1)
        self.assertEqual(runner.segment_stats["place"]["success_first_try"], 1)
        episodes = self.saved_episodes()
        self.assertEqual(len(episodes), 2)
        for name, attrs in episodes.items():
            self.assertTrue(name.endswith("_success.rmb"), name)
            self.assertEqual(attrs["segment_result"], "success")
            self.assertGreater(attrs["_len"], 0)
            self.assertIn(DataKey.MEASURED_GRIPPER_MARKER_WIDTH, attrs["_keys"])
        with open(runner.args.result_filename) as f:
            results = yaml.safe_load(f)
        self.assertEqual(len(results["cycles"]), 1)
        self.assertFalse(results["cycles"][0].get("incomplete", False))
        self.assertEqual(results["cycles"][0]["segments"]["grasp"]["result"], "success")

    def test_empty_grasp_is_retried(self):
        runner, tracker = self.run_runner(
            ["empty", "grasped", "grasped", "released"],
            ["--num_cycles", "1"],
            keys=[self.start_key()],
        )
        self.assertEqual(tracker.checks, ["empty", "grasped", "grasped", "released"])
        stat = runner.segment_stats["grasp"]
        self.assertEqual(stat["attempts"], 2)
        self.assertEqual(stat["success_after_retry"], 1)
        self.assertEqual(stat["fail_reasons"], {"empty": 1})
        names = sorted(self.saved_episodes())
        self.assertTrue(
            any(n.endswith("_r0_failure.rmb") and "_grasp_" in n for n in names), names
        )
        self.assertTrue(
            any(n.endswith("_r1_success.rmb") and "_grasp_" in n for n in names), names
        )

    def test_drop_while_lifting_relabels_the_grasp(self):
        runner, tracker = self.run_runner(
            ["grasped", "empty", "grasped", "grasped", "released"],
            ["--num_cycles", "1"],
            keys=[self.start_key()],
        )
        stat = runner.segment_stats["grasp"]
        self.assertEqual(stat["fail_reasons"], {"dropped": 1})
        failed = [
            a for n, a in self.saved_episodes().items() if n.endswith("_failure.rmb")
        ]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0]["fail_reason"], "dropped")
        self.assertEqual(failed[0]["check_result"], "empty")

    def test_hidden_tags_deferred_to_lift_check(self):
        runner, tracker = self.run_runner(
            ["hidden", "grasped", "released"],
            ["--num_cycles", "1"],
            keys=[self.start_key()],
        )
        self.assertEqual(runner.segment_stats["grasp"]["success_first_try"], 1)
        runner, tracker = self.run_runner(
            ["hidden", "hidden", "grasped", "grasped", "released"],
            ["--num_cycles", "1"],
            keys=[self.start_key()],
        )
        self.assertEqual(
            runner.segment_stats["grasp"]["fail_reasons"], {"undetected": 1}
        )

    def test_stop_after_max_retries_and_resume(self):
        def resume_when_stopped(runner):
            return isinstance(runner.phase, WaitForOperatorPhase)

        runner, tracker = self.run_runner(
            ["empty", "empty", "empty", "grasped", "grasped", "released"],
            ["--num_cycles", "2"],
            keys=[self.start_key(), (resume_when_stopped, "n")],
        )
        self.assertEqual(runner.stats["cycles"], 2)
        self.assertEqual(runner.stats["stops"], 1)
        self.assertEqual(runner.stats["cycles_without_stop"], 1)
        self.assertEqual(runner.segment_stats["grasp"]["stopped"], 1)
        self.assertTrue(runner.cycle_results[0]["stopped"])
        self.assertIn("failed 3 times", runner.cycle_results[0]["stop_reason"])
        self.assertEqual(len(runner.cycle_results), 2)

    def test_timeout(self):
        # The operator never closes the gripper; time under human control
        # does not count, so put the segment under "policy" time by
        # pretending nobody is in control.
        def tweak(task):
            for spec in task["segments"]:
                if spec["name"] == "grasp":
                    spec["timeout_s"] = 0.0
                    spec["max_retries"] = 0

        def no_human(runner):
            phase = getattr(runner, "phase", None)
            if phase is not None and getattr(phase, "label", None) == "grasp":
                phase.human_control = False
                phase.policy_runner = types.SimpleNamespace(
                    reset_buffers=lambda op: None,
                    infer_and_command=lambda op: None,
                    hold_current_command=lambda op: None,
                    checkpoint="none",
                )

        runner, _ = self.run_runner(
            [],
            ["--num_cycles", "1"],
            keys=[self.start_key()],
            tweak_task=tweak,
            max_ticks=200,
            on_tick=no_human,
        )
        self.assertEqual(runner.segment_stats["grasp"]["fail_reasons"], {"timeout": 1})
        self.assertIsInstance(runner.phase, WaitForOperatorPhase)

    def test_checks_require_calibration(self):
        def uncalibrated(task):
            task["gripper_marker"] = os.path.join(
                CONFIG_DIR, "gripper_marker", "CupPickPlace.yaml"
            )

        with self.assertRaisesRegex(RuntimeError, "not calibrated"):
            self.run_runner([], tweak_task=uncalibrated)

    def test_exhibit_mode_only_accepts_success(self):
        runner, _ = self.run_runner(
            ["grasped", "grasped", "released"],
            ["--num_cycles", "1", "--skip_checks"],
            keys=[self.start_key()],
        )
        runner.args.mode = "exhibit"
        runner.key_listener = types.SimpleNamespace(
            poll=lambda: {"fail", "intervene", "success"}
        )
        runner.key = ord("f")
        runner._last_action_time = {}
        self.assertEqual(runner.poll_actions(), {"success"})
        runner.args.mode = "train"
        runner._last_action_time = {}
        self.assertEqual(runner.poll_actions(), {"fail", "intervene", "success"})

    def grasp_episode(self, suffix="_success.rmb"):
        paths = [
            p
            for p in find_rmb_files(self.dataset_dir)
            if "_grasp_" in os.path.basename(p) and p.endswith(suffix)
        ]
        self.assertEqual(len(paths), 1, paths)
        with RmbData(paths[0]) as rmb_data:
            self.last_command_eef_pose = rmb_data[DataKey.COMMAND_EEF_POSE][:]
            return (
                dict(rmb_data.attrs),
                rmb_data[DataKey.INTERVENTION][:].tolist(),
                rmb_data[DataKey.INTERVENTION_INPUT_ACTIVE][:].tolist(),
            )

    def assert_no_command_jump(self, max_step_m=0.005):
        # Taking over / handing back must not move the commanded target by
        # more than one SpaceMouse step (the fake operator moves 3 mm/tick,
        # the fake policy not at all).
        steps = np.linalg.norm(
            np.diff(self.last_command_eef_pose[:, :3], axis=0), axis=1
        )
        self.assertLess(steps.max(), max_step_m)

    def operator_ticks(self, name, min_ticks):
        # Presses of the same action closer than ACTION_DEBOUNCE_SEC (0.3 s)
        # count once; a dry-run tick takes >= 20 ms, so keep min_ticks >= 20.
        def condition(runner):
            return (
                self.intervening(name)(runner) and runner.teleop_device.t >= min_ticks
            )

        return condition

    def test_intervention_ends_the_segment(self):
        # Method A: take over mid-grasp, close the gripper, hand back -> done.
        runner, tracker = self.run_runner(
            ["grasped", "grasped", "released"],
            ["--num_cycles", "1"],
            keys=[
                self.start_key(),
                (self.in_run("grasp", 3), "i"),
                (self.operator_ticks("grasp", 25), "i"),
            ],
            policies={"grasp": {}, "place": {"toggle_after": 3}},
        )
        self.assertEqual(runner.stats["cycles_without_stop"], 1)
        calls = self.fake_policies["grasp"].calls
        self.assertIn("observe", calls)
        self.assertNotIn("reset_action_buffers", calls)
        # No inference while the human was in control.
        first_observe = calls.index("observe")
        last_observe = len(calls) - 1 - calls[::-1].index("observe")
        self.assertNotIn("infer", calls[first_observe:last_observe])
        attrs, intervention, input_active = self.grasp_episode()
        self.assertEqual(attrs["num_interventions"], 1)
        self.assertEqual(attrs["intervention_methods"], "A")
        self.assertGreater(attrs["intervention_time"], 0.0)
        self.assertEqual(intervention[:3], [0, 0, 0])
        self.assertEqual(intervention[-1], 1)
        self.assertEqual(input_active, intervention)  # FakeOperator always moving
        self.assert_no_command_jump()
        self.assertEqual(runner.segment_stats["grasp"]["interventions"], 1)

    def test_intervention_hands_back_to_the_policy(self):
        runner, tracker = self.run_runner(
            ["grasped", "grasped", "released"],
            ["--num_cycles", "1"],
            keys=[
                self.start_key(),
                (self.in_run("grasp", 3), "i"),
                (self.operator_ticks("grasp", 25), "i"),
            ],
            operator_kwargs={"toggle_tick": 1000},
            policies={"grasp": {"toggle_after": 8}, "place": {"toggle_after": 3}},
        )
        calls = self.fake_policies["grasp"].calls
        self.assertIn("reset_action_buffers", calls)
        self.assertIn("infer", calls[calls.index("reset_action_buffers") :])
        attrs, intervention, _ = self.grasp_episode()
        self.assertEqual(attrs["num_interventions"], 1)
        self.assertEqual(intervention[-1], 0)  # the policy finished the segment
        self.assertIn(1, intervention)
        self.assert_no_command_jump()

    def test_method_b_after_max_retries(self):
        def tweak(task):
            for spec in task["segments"]:
                if spec["name"] == "grasp":
                    spec["timeout_s"] = 0.05
                    spec["max_retries"] = 0

        runner, tracker = self.run_runner(
            ["grasped", "grasped", "released"],
            ["--num_cycles", "1"],
            keys=[
                self.start_key(),
                (lambda runner: isinstance(runner.phase, WaitForOperatorPhase), "i"),
                (self.operator_ticks("grasp", 25), "i"),
            ],
            tweak_task=tweak,
            policies={"grasp": {}, "place": {"toggle_after": 3}},
        )
        self.assertEqual(runner.stats["stops"], 1)
        self.assertEqual(runner.stats["cycles"], 1)
        self.assertTrue(runner.cycle_results[0]["stopped"])
        stat = runner.segment_stats["grasp"]
        self.assertEqual(stat["attempts"], 2)
        self.assertEqual(stat["fail_reasons"], {"timeout": 1})
        attrs, intervention, _ = self.grasp_episode()
        self.assertEqual(attrs["intervention_methods"], "B")
        self.assertTrue(all(flag == 1 for flag in intervention))
        self.assertEqual(runner.segment_stats["place"]["success_first_try"], 1)

    def test_intervention_not_allowed_in_exhibit_mode(self):
        with self.assertRaises(SystemExit), mock.patch("sys.stderr"):
            self.run_runner([], ["--mode", "exhibit"], policies={})


if __name__ == "__main__":
    unittest.main()
