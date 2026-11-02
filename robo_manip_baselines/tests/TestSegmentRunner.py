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
    ):
        task = yaml.safe_load(yaml.safe_dump(self.task))
        if tweak_task is not None:
            tweak_task(task)
        task_path = os.path.join(self.tmp_dir, "task.yaml")
        with open(task_path, "w") as f:
            yaml.safe_dump(task, f)
        tracker = FakeTracker(outcomes)

        class Runner(OperationRealFairino5Demo, SegmentRunnerBase):
            def setup_teleop_device(self):
                self.teleop_device = FakeOperator(
                    self.motion_manager.body_manager_list[0], **(operator_kwargs or {})
                )

            def setup_checks(self):
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

        argv = [
            "RolloutSegments.py",
            "--task",
            task_path,
            "--replace_learned_with_teleop",
            "--no_pedal",
            "--control_period",
            "0.0",
            "--save_rollout",
            "--dataset_dir",
            self.dataset_dir,
            *extra_args,
        ]
        with (
            mock.patch.object(sys, "argv", argv),
            mock.patch.object(cv2, "imshow"),
            mock.patch.object(cv2, "waitKey", side_effect=wait_key),
            mock.patch("builtins.print"),
        ):
            runner = Runner(**self.env_config)
            state["runner"] = runner
            runner.run()
        return runner, tracker

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


if __name__ == "__main__":
    unittest.main()
