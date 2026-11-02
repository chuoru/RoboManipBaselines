"""Run a task as a cycle of scripted and learned segments.

A task YAML (envs/configs/segments/<Task>.yaml) lists the segments of one
cycle, e.g. for the exhibition task:

    go_home (scripted) -> grasp (learned) -> lift (scripted) -> lift_check
    -> go_carry (scripted) -> place (learned) -> retreat (scripted) -> back_home

and the runner repeats that cycle. Only the parts of the task that vary
(where the cup is, where the coaster is) are learned; everything else is a
fixed move. Each learned segment has a fixed start pose (the scripted segment
before it always brings the arm there), a fixed end condition (the gripper
closed / opened) and a fixed observation, so its policy can be trained and
swapped on its own.

Switching rules (design D5):
  learned -> scripted: on the end condition, drop the rest of the action
    chunk and hold the measured pose; the next scripted segment starts with
    a relative retreat (where the grasp happened differs every time).
  scripted -> learned: wait for the arm to stand still, re-read the
    observation, re-anchor the IK on it, clear every policy buffer, then
    start inferring.

Failures (design D6): a grasp/release check of the finger AprilTags
(common/utils/GripperCheckUtils.py), a timeout, or -- in train mode only --
the FAIL pedal run the learned segment's on_fail steps (back to its start
pose) and retry it; past max_retries the arm stops there and the screen asks
for the staff. Exhibit mode reacts to nothing but SUCCESS (resume after a
stop) and Esc, so a visitor on a pedal cannot change what the robot does.

Usage: bin/RolloutSegments.py.
"""

import argparse
import collections
import datetime
import logging
import logging.handlers
import os
import queue
import sys
import threading
import time

import cv2
import numpy as np
import psutil
import yaml

from ..data.DataKey import DataKey
from ..data.OperationDataMixin import OperationDataMixin
from ..manager.DataManager import DataManager
from ..manager.MotionManager import MotionManager
from ..utils.DisplayTextUtils import draw_text
from ..utils.GlobalKeyListener import GlobalKeyListener
from ..utils.GripperCheckUtils import evaluate_grasp, evaluate_release
from ..utils.MathUtils import set_random_seed
from .PhaseBase import PhaseBase
from .PolicyRunnerBase import make_policy_runner
from .SegmentPhases import (
    PACKAGE_DIR,
    get_gripper_close_threshold,
    get_measured_gripper,
    hold_at_measured,
    load_named_poses,
    make_primitive_phase,
    resolve_path,
)

# Failure reasons written to the results and the episode attrs.
FAIL_REASON_TEXT = {
    "empty": "empty grasp",
    "dropped": "dropped while lifting",
    "not_closed": "gripper did not close",
    "not_released": "gripper did not open",
    "undetected": "gripper tags not visible",
    "timeout": "timeout",
    "operator": "marked as failed by the operator",
    "camera": "camera stopped",
}


class SegmentPhaseBase(PhaseBase):
    """A phase of the segment runner. label is shown on screen."""

    def __init__(self, op, label):
        super().__init__(op)
        self.label = label

    @property
    def display_name(self):
        return self.label

    def handle_actions(self, actions):
        """Logical operator actions of this tick (fail/intervene/success)."""
        pass


class StartPhase(SegmentPhaseBase):
    """Wait for the operator before the arm moves at all."""

    def start(self):
        super().start()
        self.go = False
        self.op.log("Press SUCCESS (pedal) or 'n' (OpenCV window) to start. Esc quits.")

    def handle_actions(self, actions):
        if "success" in actions:
            self.go = True

    def check_transition(self):
        return self.go


class MoveToInitSegmentPhase(SegmentPhaseBase):
    """Move to the env's reset pose and enable streaming commands (the
    segment runner's counterpart of OperationRealFairino5Demo's
    MoveToInitPhase)."""

    def start(self):
        super().start()
        env = self.op.env.unwrapped
        if hasattr(env, "move_to_init_pose"):
            env.move_to_init_pose()
        gripper_dim = DataKey.get_dim(DataKey.COMMAND_GRIPPER_JOINT_POS, self.op.env)
        # move_to_init_pose() opened the gripper; the command must agree.
        self.op.motion_manager.set_command_data(
            DataKey.COMMAND_GRIPPER_JOINT_POS, np.zeros(gripper_dim)
        )
        hold_at_measured(self.op)

    def check_transition(self):
        return True


class WaitForOperatorPhase(SegmentPhaseBase):
    """Stopped at a safe pose; the operator decides how to continue."""

    def __init__(self, op, label, reason, learned_phase=None):
        super().__init__(op, label)
        self.reason = reason
        self.learned_phase = learned_phase
        self.choice = None

    @property
    def display_name(self):
        return "STOPPED"

    def start(self):
        super().start()
        self.choice = None
        hold_at_measured(self.op, refresh_obs=False)
        self.op.log(
            f"STOPPED: {self.reason}. Fix the scene, then press SUCCESS to "
            "restart the cycle."
        )

    def handle_actions(self, actions):
        if "success" in actions:
            self.choice = "resume"

    def check_transition(self):
        return self.choice is not None


class GripperCheck:
    """One grasp/release check: wait settle_sec after it starts, then take
    the median of the raw finger-tag widths over window_sec."""

    def __init__(self, op, kind, undetected="fail", empty_reason="empty"):
        if kind not in ("grasp_check", "released_check"):
            raise ValueError(f"Invalid check: {kind} (grasp_check / released_check)")
        if undetected not in ("pass", "fail", "defer"):
            raise ValueError(
                f"Invalid undetected policy: {undetected} (pass / fail / defer)"
            )
        self.op = op
        self.kind = kind
        self.undetected = undetected
        self.empty_reason = empty_reason

    def start(self):
        self.start_time = time.time()
        self.outcome = None

    def update(self):
        """None while still measuring, then (passed, fail_reason, info)."""
        config = self.op.check_config
        elapsed = time.time() - self.start_time
        if elapsed < config["settle_sec"] + config["window_sec"]:
            return None
        widths = self.op.env.unwrapped.gripper_marker_tracker.get_raw_widths_since(
            self.start_time + config["settle_sec"]
        )
        if self.kind == "grasp_check":
            result, width, num = evaluate_grasp(
                widths,
                config["grasp"]["width_min_m"],
                config["grasp"]["width_max_m"],
                config["min_detections"],
            )
            passed = result == "grasped"
            fail_reason = self.empty_reason if result == "empty" else result
        else:
            result, width, num = evaluate_release(
                widths, config["release"]["width_open_min_m"], config["min_detections"]
            )
            passed = result == "released"
            fail_reason = result
        if result == "undetected":
            # "defer": treat as passed for now and let a later check (e.g.
            # after lifting, with the cup clear of the table) decide.
            passed = self.undetected in ("pass", "defer")
        info = {
            "check": self.kind,
            "check_result": result,
            "check_width_m": float("nan") if width is None else width,
            "check_detections": num,
            "check_frames": len(widths),
        }
        width_str = "-" if width is None else f"{width * 1000:.1f} mm"
        self.op.log(
            f"{self.kind}: {result} (median width {width_str}, {num}/{len(widths)} frames)"
        )
        return passed, None if passed else fail_reason, info


class CheckSegmentPhase(SegmentPhaseBase):
    """A check between scripted segments (e.g. after lifting: still holding
    the cup?). A failure counts against the learned segment it follows."""

    def __init__(self, op, spec):
        super().__init__(op, spec["name"])
        self.spec = spec
        self.checker = GripperCheck(
            op,
            spec.get("check", "grasp_check"),
            undetected=spec.get("undetected", "fail"),
            empty_reason=spec.get("fail_reason", "dropped"),
        )
        self.owner = None  # set by the runner: the preceding learned phase

    def start(self):
        super().start()
        hold_at_measured(self.op, refresh_obs=False)
        self.result = None
        self.fail_reason = None
        self.check_info = {}
        if not self.op.checks_enabled:
            self.result = "success"
            return
        self.checker.start()

    def post_update(self):
        if self.result is not None:
            return
        outcome = self.checker.update()
        if outcome is not None:
            passed, self.fail_reason, self.check_info = outcome
            self.result = "success" if passed else "failure"

    def check_transition(self):
        return self.result is not None


class LearnedSegmentPhase(SegmentPhaseBase):
    """A learned segment: settle -> run the policy (or the operator, with
    --replace_learned_with_teleop) until the end condition -> check -> result."""

    # Settling before the first inference: the arm must be still (all joint
    # speeds below this [rad/s]) for SETTLE_STILL_TICKS consecutive ticks,
    # at least SETTLE_MIN_TICKS ticks, or SETTLE_TIMEOUT_SEC passes.
    SETTLE_JOINT_VEL = 0.02
    SETTLE_STILL_TICKS = 2
    SETTLE_MIN_TICKS = 2
    SETTLE_TIMEOUT_SEC = 3.0

    def __init__(self, op, spec, policy_runner=None):
        super().__init__(op, spec["name"])
        self.spec = spec
        self.policy_runner = policy_runner
        self.done_condition = spec.get("done", "gripper_closed_for_ticks")
        self.done_ticks = int(spec.get("done_ticks", 5))
        self.timeout_s = spec.get("timeout_s")
        self.max_retries = int(spec.get("max_retries", 2))
        self.on_fail_specs = spec.get("on_fail", [])
        self.teleop_only = policy_runner is None
        self.checker = None
        if spec.get("check") is not None:
            self.checker = GripperCheck(
                op,
                spec["check"],
                undetected=spec.get("check_undetected", "defer"),
                empty_reason="empty",
            )

    @property
    def display_name(self):
        return f"{self.label} [{self.state}]"

    @property
    def is_recording_state(self):
        return self.state == "run"

    def start(self):
        super().start()
        self.state = "settle"
        self.result = None
        self.fail_reason = None
        self.check_info = {}
        self.settle_ticks = 0
        self.still_ticks = 0
        self.done_count = 0
        self.run_time = 0.0
        self.last_tick_time = None
        self.human_control = self.teleop_only
        hold_at_measured(self.op, refresh_obs=False)

    # ---- per tick -------------------------------------------------------

    def pre_update(self):
        if self.state != "run":
            return
        if self.human_control:
            self.op.apply_teleop_command()
            if self.policy_runner is not None:
                self.policy_runner.observe(self.op)
        else:
            self.policy_runner.infer_and_command(self.op)

    def post_update(self):
        if self.state == "settle":
            self._update_settle()
        elif self.state == "run":
            now = time.time()
            if self.last_tick_time is not None and not self.human_control:
                # Only time under policy control counts toward timeout_s.
                self.run_time += now - self.last_tick_time
            self.last_tick_time = now
            if self._is_done():
                self.op.log(
                    f"[{self.label}] End condition '{self.done_condition}' met."
                )
                self.end_run()
            elif self.timeout_s is not None and self.run_time > self.timeout_s:
                self.end_run("timeout")
        elif self.state == "check":
            outcome = self.checker.update()
            if outcome is not None:
                passed, fail_reason, self.check_info = outcome
                self.finish("success" if passed else "failure", fail_reason)

    def _update_settle(self):
        self.settle_ticks += 1
        joint_vel = self.op.env.unwrapped.get_joint_vel_from_obs(self.op.obs)
        arm_joint_idxes = self.op.env.unwrapped.body_config_list[0].arm_joint_idxes
        if np.max(np.abs(joint_vel[arm_joint_idxes])) < self.SETTLE_JOINT_VEL:
            self.still_ticks += 1
        else:
            self.still_ticks = 0
        settled = (
            self.settle_ticks >= self.SETTLE_MIN_TICKS
            and self.still_ticks >= self.SETTLE_STILL_TICKS
        )
        if settled or self.get_elapsed_duration() > self.SETTLE_TIMEOUT_SEC:
            self._start_run()

    def _start_run(self):
        # Scripted -> learned (D5): re-anchor on the measured state and clear
        # every policy buffer before the first inference.
        hold_at_measured(self.op)
        if self.policy_runner is not None:
            self.policy_runner.reset_buffers(self.op)
        if self.human_control:
            self.op.reset_teleop_device()
        self.state = "run"
        self.last_tick_time = None
        self.op.start_segment_recording(self)
        who = "operator" if self.human_control else "policy"
        self.op.log(f"[{self.label}] Start ({who} in control).")

    def _is_done(self):
        gripper = get_measured_gripper(self.op)
        threshold = get_gripper_close_threshold(self.op)
        if self.done_condition == "gripper_closed_for_ticks":
            condition = gripper >= threshold
        elif self.done_condition == "gripper_opened_for_ticks":
            condition = gripper < threshold
        elif self.done_condition == "success_pedal":
            return False
        else:
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid done condition: {self.done_condition}"
            )
        self.done_count = self.done_count + 1 if condition else 0
        return self.done_count >= self.done_ticks

    def handle_actions(self, actions):
        if self.state != "run":
            return
        if "fail" in actions and self.op.is_action_enabled("fail"):
            self.end_run("operator")
        elif "success" in actions and self.op.args.mode != "exhibit":
            self.op.log(f"[{self.label}] Marked as success by the operator.")
            self.end_run(check=False)

    def stop_control(self):
        # Learned -> scripted (D5): drop the rest of the chunk, hold here.
        if self.policy_runner is not None:
            self.policy_runner.hold_current_command(self.op)
        else:
            hold_at_measured(self.op, refresh_obs=False)

    def end_run(self, fail_reason=None, check=True):
        """Stop the run; on success go through the check (if any)."""
        self.stop_control()
        if fail_reason is not None:
            self.finish("failure", fail_reason)
        elif check and self.checker is not None and self.op.checks_enabled:
            self.state = "check"
            self.checker.start()
        else:
            self.finish("success")

    def finish(self, result, fail_reason=None):
        self.state = "done"
        self.result = result
        self.fail_reason = fail_reason
        if result == "failure":
            self.op.log(
                f"[{self.label}] FAILED: {FAIL_REASON_TEXT.get(fail_reason, fail_reason)}."
            )

    def abort(self, fail_reason):
        """Stop at once, from outside (e.g. the camera stopped)."""
        if self.state in ("settle", "run", "check"):
            self.stop_control()
            self.finish("failure", fail_reason)

    def check_transition(self):
        return self.result is not None


class SegmentRunnerBase(OperationDataMixin):
    MotionManagerClass = MotionManager
    DataManagerClass = DataManager

    def __init__(self):
        self.setup_args()
        set_random_seed(self.args.seed)
        self.datetime_now = datetime.datetime.now()
        self.setup_logger()
        self.load_task()

        self.setup_env(render_mode=None)
        self.demo_name = self.task_name

        self.motion_manager = self.MotionManagerClass(self.env)
        self.data_manager = self.DataManagerClass(self.env, demo_name=self.demo_name)
        self.data_manager.setup_camera_info()
        self.obs, self.info = self.env.reset(seed=self.args.seed)
        self.reward = 0.0

        self.setup_checks()
        self.setup_key_listener()
        self.setup_teleop_device()
        self.build_phases()
        self.setup_recording()

    # ---- setup -----------------------------------------------------------

    def setup_args(self, parser=None, argv=None):
        if parser is None:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
        parser.add_argument(
            "--task",
            type=str,
            required=True,
            help="segment task YAML (e.g. ./envs/configs/segments/CupPickPlace.yaml)",
        )
        parser.add_argument(
            "--mode",
            type=str,
            default="train",
            choices=["train", "exhibit"],
            help="train: pedals may mark failures and intervene; exhibit: "
            "fully automatic, only SUCCESS (restart after a stop) and Esc work",
        )
        parser.add_argument(
            "--checkpoint",
            type=str,
            nargs="*",
            default=[],
            metavar="SEGMENT=PATH",
            help="override the checkpoint of a learned segment, e.g. grasp=./checkpoint/...",
        )
        parser.add_argument(
            "--poses",
            type=str,
            default=None,
            help="named pose YAML to use instead of the task YAML's poses "
            "(e.g. ./envs/configs/poses/DryRunExample.yaml for a dry run)",
        )
        parser.add_argument(
            "--replace_learned_with_teleop",
            action="store_true",
            help="drive every learned segment with the SpaceMouse instead of a "
            "policy (end condition or SUCCESS ends it) -- to try the whole "
            "cycle before any policy exists",
        )
        parser.add_argument(
            "--skip_checks",
            action="store_true",
            help="run without the finger-tag grasp/release checks (only "
            "timeouts and the FAIL pedal detect failures). Not allowed in "
            "exhibit mode.",
        )
        parser.add_argument(
            "--input_device",
            type=str,
            default="spacemouse",
            choices=["spacemouse", "keyboard"],
            help="device for teleop segments / interventions (keyboard: for "
            "dry runs without a SpaceMouse)",
        )
        parser.add_argument(
            "--input_device_config",
            type=str,
            default="./teleop/configs/SpaceMouseRealFairino5.yaml",
            help="config of the SpaceMouse for teleop segments / interventions "
            "(ignored for --input_device keyboard)",
        )
        parser.add_argument(
            "--pedal_config",
            type=str,
            default="./common/configs/pedal_default.yaml",
            help="pedal key bindings (see common/utils/GlobalKeyListener.py)",
        )
        parser.add_argument(
            "--no_pedal", action="store_true", help="do not listen for pedal keys"
        )
        parser.add_argument(
            "--control_period",
            type=float,
            default=None,
            help="loop period [s] = one policy step (default: the task YAML's "
            "control_period, else 0.2)",
        )
        parser.add_argument(
            "--policy_skip",
            type=int,
            default=None,
            help="infer every this many loops (default 1: one policy step per loop)",
        )
        parser.add_argument(
            "--num_cycles",
            type=int,
            default=None,
            help="stop after this many cycles (default: run until Esc / Ctrl+C)",
        )
        parser.add_argument(
            "--save_rollout",
            action=argparse.BooleanOptionalAction,
            default=None,
            help="record every learned-segment attempt as an RMB episode "
            "(default: on in exhibit mode, off in train mode)",
        )
        parser.add_argument(
            "--iteration",
            type=int,
            default=1,
            help="training iteration the recorded data is for: attempts go to "
            "dataset/<Task>_<segment>_rollout_iterNNN",
        )
        parser.add_argument(
            "--dataset_dir",
            type=str,
            default=None,
            help="where the rollout data directories are created (default: the "
            "repository's dataset/)",
        )
        parser.add_argument(
            "--result_filename",
            type=str,
            default=None,
            help="cycle results YAML (default: ./logs/segment_results_<Task>_<time>.yaml)",
        )
        parser.add_argument(
            "--camera_timeout",
            type=float,
            default=3.0,
            help="stop for the operator if the camera delivers no frame for this long [s]",
        )
        parser.add_argument(
            "--window_width",
            type=int,
            default=960,
            help="width [px] of the status window",
        )
        parser.add_argument("--seed", type=int, default=-1, help="random seed")

        self.set_additional_args(parser)

        if argv is None:
            argv = sys.argv
        self.args = parser.parse_args(argv[1:])
        if self.args.seed < 0:
            self.args.seed = int(time.time()) % (2**32)
        if self.args.save_rollout is None:
            self.args.save_rollout = self.args.mode == "exhibit"
        if self.args.mode == "exhibit" and self.args.skip_checks:
            parser.error("--skip_checks is not allowed with --mode exhibit")
        if self.args.mode == "exhibit" and self.args.replace_learned_with_teleop:
            parser.error(
                "--replace_learned_with_teleop is not allowed with --mode exhibit"
            )

    def set_additional_args(self, parser):
        pass

    def setup_env(self, render_mode=None):
        raise NotImplementedError(
            f"[{self.__class__.__name__}] This method should be defined in the Operation class and inherited from it."
        )

    def setup_logger(self):
        os.makedirs("logs", exist_ok=True)
        self.logger = logging.getLogger(f"SegmentRunner_{id(self)}")
        self.logger.setLevel(logging.INFO)
        # Rotated so a day-long exhibition cannot fill the disk.
        handler = logging.handlers.RotatingFileHandler(
            os.path.join(
                "logs", f"segment_runner_{self.datetime_now:%Y%m%d_%H%M%S}.log"
            ),
            maxBytes=5 * 1024**2,
            backupCount=5,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        self.logger.addHandler(handler)

    def log(self, msg):
        print(f"[{self.__class__.__name__}] {msg}", flush=True)
        self.logger.info(msg)

    def load_task(self):
        task_path = resolve_path(self.args.task)
        with open(task_path, "r") as f:
            self.task_config = yaml.safe_load(f)
        task_dir = os.path.dirname(os.path.abspath(task_path))
        self.task_name = self.task_config.get(
            "task_name", os.path.splitext(os.path.basename(task_path))[0]
        )
        if self.args.poses is not None:
            poses_path = resolve_path(self.args.poses)
        else:
            poses_path = resolve_path(self.task_config["poses"], task_dir)
        self.named_poses = load_named_poses(poses_path)
        self.segment_specs = [dict(spec) for spec in self.task_config["segments"]]
        if self.args.control_period is None:
            self.args.control_period = float(
                self.task_config.get("control_period", 0.2)
            )

        # Finger-tag config: detection settings for the env, thresholds for
        # the checks. The env is built after this, from the Operation's
        # gripper_marker_config -- fill that in unless the env config set one.
        self.gripper_marker_path = None
        if self.task_config.get("gripper_marker") is not None:
            self.gripper_marker_path = resolve_path(
                self.task_config["gripper_marker"], task_dir
            )
            if getattr(self, "gripper_marker_config", None) is None and hasattr(
                self, "gripper_marker_config"
            ):
                self.gripper_marker_config = self.gripper_marker_path

        checkpoint_overrides = {}
        for item in self.args.checkpoint:
            if "=" not in item:
                raise ValueError(f"--checkpoint expects SEGMENT=PATH: {item}")
            name, path = item.split("=", 1)
            checkpoint_overrides[name] = path
        for spec in self.segment_specs:
            if spec["name"] in checkpoint_overrides:
                spec["checkpoint"] = checkpoint_overrides.pop(spec["name"])
            if spec.get("checkpoint") is not None:
                spec["checkpoint"] = resolve_path(spec["checkpoint"], task_dir)
        if checkpoint_overrides:
            raise ValueError(
                f"--checkpoint names unknown segments: {sorted(checkpoint_overrides)}"
            )
        self.validate_segments()
        self.log(
            f"Task '{self.task_name}' ({task_path}): "
            + " -> ".join(spec["name"] for spec in self.segment_specs)
        )

    def validate_segments(self):
        names = [spec.get("name") for spec in self.segment_specs]
        if None in names or len(set(names)) != len(names):
            raise ValueError(f"Every segment needs a unique name: {names}")
        seen_learned = False
        for idx, spec in enumerate(self.segment_specs):
            if spec.get("type") not in ("scripted_move", "gripper", "learned", "check"):
                raise ValueError(
                    f"Segment '{spec['name']}' has an invalid type: {spec}"
                )
            if spec["type"] == "check" and not seen_learned:
                raise ValueError(
                    f"Check segment '{spec['name']}' must come after a learned segment "
                    "(its failure retries that segment)."
                )
            if spec["type"] != "learned":
                continue
            seen_learned = True
            next_spec = self.segment_specs[(idx + 1) % len(self.segment_specs)]
            if not (
                next_spec.get("type") == "scripted_move" and "rel_xyz" in next_spec
            ):
                # Where a learned segment ends differs every time; going
                # straight to a fixed pose from there can sweep through the
                # object. D5: retreat relatively first.
                self.log(
                    f"WARNING: learned segment '{spec['name']}' is not followed by a "
                    "relative retreat (rel_xyz)."
                )

    def uses_checks(self):
        return any(
            spec["type"] == "check" or spec.get("check") is not None
            for spec in self.segment_specs
        )

    def setup_checks(self):
        """Enable the finger-tag checks, or explain why they cannot run."""
        self.checks_enabled = False
        self.check_config = None
        if not self.uses_checks():
            return
        if self.args.skip_checks:
            self.log("WARNING: grasp/release checks disabled (--skip_checks).")
            return
        problems = []
        if not getattr(self.env.unwrapped, "has_gripper_marker_tracking", False):
            problems.append(
                "the env has no finger-tag tracking (set gripper_marker in the task "
                "YAML or gripper_marker_config in the env config)"
            )
        if self.gripper_marker_path is None:
            problems.append(
                "the task YAML has no gripper_marker file (check thresholds)"
            )
        else:
            with open(self.gripper_marker_path, "r") as f:
                self.check_config = (yaml.safe_load(f) or {}).get("checks")
            needed = [
                ("grasp", "width_min_m"),
                ("grasp", "width_max_m"),
                ("release", "width_open_min_m"),
            ]
            if self.check_config is None:
                problems.append(f"{self.gripper_marker_path} has no 'checks' section")
            else:
                missing = [
                    f"{group}.{key}"
                    for group, key in needed
                    if (self.check_config.get(group) or {}).get(key) is None
                ]
                if missing:
                    problems.append(
                        f"thresholds not calibrated in {self.gripper_marker_path}: "
                        f"{', '.join(missing)} (misc/CalibrateGripperMarkerWidth.py)"
                    )
        if problems:
            raise RuntimeError(
                f"[{self.__class__.__name__}] The task uses grasp/release checks but "
                + "; ".join(problems)
                + ". Pass --skip_checks to run without them (train mode only)."
            )
        self.check_config.setdefault("settle_sec", 0.6)
        self.check_config.setdefault("window_sec", 1.0)
        self.check_config.setdefault("min_detections", 3)
        self.checks_enabled = True

    def setup_key_listener(self):
        self.key_listener = None
        if not self.args.no_pedal:
            self.key_listener = GlobalKeyListener.from_config(
                resolve_path(self.args.pedal_config),
                enabled_actions=self.get_enabled_actions(),
            )
            if not self.key_listener.start():
                self.key_listener = None
        self.key = -1
        self.actions = set()
        self._last_action_time = {}

    def get_enabled_actions(self):
        if self.args.mode == "exhibit":
            # A visitor stepping on a pedal must not change what the robot
            # does: only "resume after a stop" stays live.
            return {"success"}
        return {"fail", "intervene", "success"}

    def is_action_enabled(self, name):
        return name in self.get_enabled_actions()

    # OpenCV window keys -> logical actions (the pedal file adds its own).
    CV_KEY_ACTIONS = {"n": "success", "s": "success", "f": "fail", "i": "intervene"}
    # Same action from the window key and the pedal within this window [s]
    # counts once (a pedal that types 'n' also reaches a focused window).
    ACTION_DEBOUNCE_SEC = 0.3

    def poll_actions(self):
        actions = set()
        if self.key_listener is not None:
            actions |= self.key_listener.poll()
        if self.key != -1:
            action = self.CV_KEY_ACTIONS.get(chr(self.key & 0xFF).lower())
            if action is not None:
                actions.add(action)
        actions &= self.get_enabled_actions()
        now = time.time()
        debounced = set()
        for action in actions:
            last = self._last_action_time.get(action)
            if last is None or now - last >= self.ACTION_DEBOUNCE_SEC:
                debounced.add(action)
                self._last_action_time[action] = now
        return debounced

    def needs_teleop_device(self):
        return self.args.replace_learned_with_teleop

    def setup_teleop_device(self):
        self.teleop_device = None
        if not self.needs_teleop_device():
            return
        kwargs = {}
        if self.args.input_device == "spacemouse":
            with open(resolve_path(self.args.input_device_config), "r") as f:
                kwargs = yaml.safe_load(f) or {}
        self.teleop_device = self.env.unwrapped.setup_input_device(
            self.args.input_device, self.motion_manager, kwargs
        )[0]
        self.teleop_device.connect()
        self.log(f"Input device '{self.args.input_device}' connected.")

    def reset_teleop_device(self):
        if self.teleop_device is not None and hasattr(
            self.teleop_device, "reset_filter"
        ):
            self.teleop_device.reset_filter()

    def apply_teleop_command(self):
        self.teleop_device.read()
        self.teleop_device.set_command_data()

    def build_phases(self):
        self.policy_runners = {}
        self.segment_phases = []
        last_learned = None
        self.dependent_checks = collections.defaultdict(list)
        for spec in self.segment_specs:
            if spec["type"] == "learned":
                phase = self.make_learned_phase(spec)
                last_learned = phase
            elif spec["type"] == "check":
                phase = CheckSegmentPhase(self, spec)
                phase.owner = last_learned
                self.dependent_checks[last_learned.label].append(phase.label)
            else:
                phase = make_primitive_phase(
                    self, spec, self.named_poses, label=spec["name"]
                )
            self.segment_phases.append(phase)
        self.on_fail_phases = {
            spec["name"]: [
                make_primitive_phase(self, step, self.named_poses)
                for step in spec.get("on_fail", [])
            ]
            for spec in self.segment_specs
            if spec["type"] == "learned"
        }
        resume_specs = self.task_config.get("on_operator_resume", [{"gripper": "open"}])
        self.resume_phases = [
            make_primitive_phase(self, step, self.named_poses) for step in resume_specs
        ]

    def make_learned_phase(self, spec):
        runner = None
        if not self.args.replace_learned_with_teleop:
            checkpoint = spec.get("checkpoint")
            if checkpoint is None or not os.path.exists(checkpoint):
                raise FileNotFoundError(
                    f"[{self.__class__.__name__}] Learned segment '{spec['name']}' "
                    f"has no checkpoint ({checkpoint}). Train it first, pass "
                    "--checkpoint SEGMENT=PATH, or use --replace_learned_with_teleop."
                )
            self.log(f"Load policy for '{spec['name']}': {checkpoint}")
            runner = make_policy_runner(
                spec["name"],
                checkpoint,
                self,
                policy_args=spec.get("policy_args"),
                policy_skip=self.args.policy_skip,
            )
            self.policy_runners[spec["name"]] = runner
        return LearnedSegmentPhase(self, spec, runner)

    # ---- sequencing --------------------------------------------------------

    def reset_sequence(self):
        self.cycle_idx = 0
        self.segment_idx = 0
        self.pending_phases = [
            StartPhase(self, "start"),
            MoveToInitSegmentPhase(self, "init"),
        ]
        self.retry_count = {}
        self.cycle_result = None
        self.held_episodes = {}
        self.stats = {
            "cycles": 0,
            "cycles_without_stop": 0,
            "consecutive_successes": 0,
            "stops": 0,
        }
        self.segment_stats = {}
        self.start_cycle()
        self.phase = None
        self.next_phase()

    def start_cycle(self):
        self.retry_count = {}
        self.cycle_result = {
            "cycle": self.cycle_idx,
            "start_time": datetime.datetime.now().isoformat(timespec="seconds"),
            "segments": {},
            "stopped": False,
            "stop_reason": None,
        }
        self.cycle_start_time = time.time()

    def next_phase(self):
        if self.pending_phases:
            self.phase = self.pending_phases.pop(0)
        else:
            self.phase = self.segment_phases[self.segment_idx]
        self.phase.start()

    def is_current_segment(self, phase):
        return self.segment_idx < len(self.segment_phases) and (
            phase is self.segment_phases[self.segment_idx]
        )

    def on_phase_done(self, phase):
        if isinstance(phase, WaitForOperatorPhase):
            self.on_operator_choice(phase)
        elif self.is_current_segment(phase):
            if isinstance(phase, LearnedSegmentPhase):
                self.on_learned_done(phase)
            elif isinstance(phase, CheckSegmentPhase):
                self.on_check_done(phase)
            else:
                self.advance_segment()
        if self.quit_flag:
            return
        self.next_phase()

    def segment_record(self, name):
        return self.cycle_result["segments"].setdefault(
            name, {"result": None, "attempts": 0, "fail_reasons": []}
        )

    def segment_stat(self, name):
        return self.segment_stats.setdefault(
            name,
            {
                "attempts": 0,
                "success_first_try": 0,
                "success_after_retry": 0,
                "stopped": 0,
                "fail_reasons": {},
            },
        )

    def count_failure(self, name, reason):
        record = self.segment_record(name)
        record["result"] = "failure"
        record["fail_reasons"].append(reason)
        reasons = self.segment_stat(name)["fail_reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1

    def count_success(self, name):
        record = self.segment_record(name)
        record["result"] = "success"
        stat = self.segment_stat(name)
        if self.retry_count.get(name, 0) == 0:
            stat["success_first_try"] += 1
        else:
            stat["success_after_retry"] += 1

    def on_learned_done(self, phase):
        name = phase.label
        self.segment_record(name)["attempts"] += 1
        self.segment_stat(name)["attempts"] += 1
        episode = self.take_segment_recording(phase)
        if phase.result == "success":
            if self.dependent_checks.get(name):
                # Not final yet: a later check (e.g. after lifting) may still
                # turn this grasp into a failure. Label the episode then.
                self.held_episodes[name] = (episode, list(self.dependent_checks[name]))
            else:
                self.commit_episode(episode, "success")
                self.count_success(name)
            self.advance_segment()
        else:
            self.commit_episode(episode, "failure", phase.fail_reason)
            self.count_failure(name, phase.fail_reason)
            self.fail_segment(phase, phase.fail_reason)

    def on_check_done(self, phase):
        owner = phase.owner.label
        held = self.held_episodes.get(owner)
        if phase.result == "success":
            if held is not None:
                episode, remaining = held
                remaining = [name for name in remaining if name != phase.label]
                if remaining:
                    self.held_episodes[owner] = (episode, remaining)
                else:
                    del self.held_episodes[owner]
                    self.commit_episode(episode, "success")
                    self.count_success(owner)
            self.advance_segment()
        else:
            if held is not None:
                del self.held_episodes[owner]
                held[0]["meta_data"].update(phase.check_info)
                self.commit_episode(held[0], "failure", phase.fail_reason)
            self.count_failure(owner, phase.fail_reason)
            self.fail_segment(phase.owner, phase.fail_reason)

    def fail_segment(self, learned_phase, reason):
        """Run the learned segment's on_fail steps, then retry it -- or, past
        max_retries, stop at its start pose and wait for the operator."""
        name = learned_phase.label
        self.retry_count[name] = self.retry_count.get(name, 0) + 1
        self.segment_idx = self.segment_phases.index(learned_phase)
        self.pending_phases = list(self.on_fail_phases[name])
        if self.retry_count[name] > learned_phase.max_retries:
            self.segment_stat(name)["stopped"] += 1
            self.pending_phases.append(
                WaitForOperatorPhase(
                    self,
                    "wait",
                    f"'{name}' failed {self.retry_count[name]} times "
                    f"(last: {FAIL_REASON_TEXT.get(reason, reason)})",
                    learned_phase,
                )
            )
        else:
            self.log(
                f"Retry '{name}' ({self.retry_count[name]}/{learned_phase.max_retries})."
            )

    def stop_for_operator(self, reason, fail_reason):
        """Stop right now (not at a start pose) -- e.g. the camera died."""
        if isinstance(self.phase, WaitForOperatorPhase):
            return
        if isinstance(self.phase, LearnedSegmentPhase) and self.phase.result is None:
            self.phase.abort(fail_reason)
            episode = self.take_segment_recording(self.phase)
            self.commit_episode(episode, "failure", fail_reason)
            self.count_failure(self.phase.label, fail_reason)
        else:
            hold_at_measured(self, refresh_obs=False)
        self.pending_phases = []
        self.phase = WaitForOperatorPhase(self, "wait", reason)
        self.phase.start()

    def on_operator_choice(self, phase):
        self.stats["stops"] += 1
        self.cycle_result["stopped"] = True
        self.cycle_result["stop_reason"] = phase.reason
        self.finish_cycle()
        # Restart the cycle from the top; resume_phases (by default: open the
        # gripper, which the attendant is holding the cup from) first.
        self.segment_idx = 0
        self.pending_phases = list(self.resume_phases)

    def advance_segment(self):
        self.segment_idx += 1
        if self.segment_idx < len(self.segment_phases):
            return
        self.segment_idx = 0
        self.stats["consecutive_successes"] += 1
        self.stats["cycles_without_stop"] += 1
        self.finish_cycle()

    def finish_cycle(self):
        # Anything still waiting for a check that never ran is final now.
        for episode, _ in self.held_episodes.values():
            self.commit_episode(episode, "success")
        for name in list(self.held_episodes):
            self.count_success(name)
        self.held_episodes = {}
        self.cycle_result["duration"] = round(time.time() - self.cycle_start_time, 2)
        self.cycle_result["memory_available_gb"] = round(
            psutil.virtual_memory().available / 1024**3, 2
        )
        self.cycle_results.append(self.cycle_result)
        self.stats["cycles"] += 1
        if self.cycle_result["stopped"]:
            self.stats["consecutive_successes"] = 0
        self.log(
            f"Cycle {self.cycle_idx} {'STOPPED' if self.cycle_result['stopped'] else 'done'} "
            f"in {self.cycle_result['duration']:.1f} s "
            f"({self.stats['cycles_without_stop']}/{self.stats['cycles']} without a stop)."
        )
        self.save_results()
        self.cycle_idx += 1
        if self.args.num_cycles is not None and self.cycle_idx >= self.args.num_cycles:
            self.quit_flag = True
            return
        self.start_cycle()

    # ---- recording ---------------------------------------------------------

    def setup_recording(self):
        self.recording_phase = None
        self.cycle_results = []
        if self.args.result_filename is None:
            self.args.result_filename = os.path.join(
                "logs",
                f"segment_results_{self.task_name}_{self.datetime_now:%Y%m%d_%H%M%S}.yaml",
            )
        self.save_queue = queue.Queue()
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()

    @property
    def phase_manager(self):
        # OperationDataMixin.record_data() reads the elapsed time of
        # self.phase_manager.phase; the segment runner has no PhaseManager,
        # the current phase plays that role.
        return self

    def start_segment_recording(self, phase):
        if not self.args.save_rollout:
            return
        self.data_manager.reset()
        # TIME starts at 0 when the policy (or operator) takes control, like
        # a teleop demo.
        phase.start_time = self.env.unwrapped.get_time()
        self.recording_phase = phase

    def is_recording(self):
        return (
            self.recording_phase is not None
            and self.recording_phase is self.phase
            and self.phase.is_recording_state
        )

    def record_extra_data(self):
        """Per-step data beyond OperationDataMixin's: the finger-tag reading."""
        reading = None
        if hasattr(self.env.unwrapped, "get_gripper_marker_reading"):
            reading = self.env.unwrapped.get_gripper_marker_reading()
        if reading is None:
            return
        width = reading["raw_width_m"] if reading["detected"] else None
        self.data_manager.append_single_data(
            DataKey.MEASURED_GRIPPER_MARKER_WIDTH,
            np.array([np.nan if width is None else width], dtype=np.float64),
        )
        self.data_manager.append_single_data(
            DataKey.GRIPPER_MARKER_DETECTED, np.int8(1 if reading["detected"] else 0)
        )

    def take_segment_recording(self, phase):
        """Detach the recorded attempt of `phase` (None if nothing recorded)."""
        if self.recording_phase is not phase:
            return None
        self.recording_phase = None
        all_data_seq = self.data_manager.all_data_seq
        self.data_manager.all_data_seq = {}
        if len(all_data_seq.get(DataKey.TIME, [])) == 0:
            return None
        meta_data = dict(self.data_manager.meta_data)
        meta_data.update(self.get_segment_meta_data(phase))
        meta_data.update(phase.check_info)
        return {
            "all_data_seq": all_data_seq,
            "meta_data": meta_data,
            "segment": phase.label,
            "cycle_idx": self.cycle_idx,
            "retry_idx": self.retry_count.get(phase.label, 0),
        }

    def commit_episode(self, episode, result, fail_reason=None):
        """Label and save a detached attempt (in the background: video
        encoding takes seconds)."""
        if episode is None:
            return
        episode["meta_data"]["segment_result"] = result
        episode["meta_data"]["fail_reason"] = fail_reason or ""
        filename = self.get_segment_filename(
            episode["segment"], episode["cycle_idx"], episode["retry_idx"], result
        )
        self.save_queue.put((filename, episode["all_data_seq"], episode["meta_data"]))

    def get_segment_meta_data(self, phase):
        name = phase.label
        return {
            "segment": name,
            "segment_result": phase.result or "",
            "fail_reason": phase.fail_reason or "",
            "retry_idx": self.retry_count.get(name, 0),
            "cycle_idx": self.cycle_idx,
            "checkpoint": ""
            if phase.policy_runner is None
            else phase.policy_runner.checkpoint,
            "iteration": self.args.iteration,
            "segment_mode": self.args.mode,
            "teleop_only": bool(phase.teleop_only),
        }

    def get_segment_filename(self, name, cycle_idx, retry_idx, result):
        dataset_dir = self.args.dataset_dir
        if dataset_dir is None:
            dataset_dir = os.path.join(PACKAGE_DIR, "dataset")
        return os.path.normpath(
            os.path.join(
                os.path.expanduser(dataset_dir),
                f"{self.task_name}_{name}_rollout_iter{self.args.iteration:03d}",
                f"{self.task_name}_{name}_{self.datetime_now:%Y%m%d_%H%M%S}_"
                f"c{cycle_idx:04d}_r{retry_idx}_{result}.rmb",
            )
        )

    def _save_worker(self):
        while True:
            item = self.save_queue.get()
            if item is None:
                return
            filename, all_data_seq, meta_data = item
            try:
                self.data_manager.save_data(
                    filename, all_data_seq, meta_data, increment_episode_idx=False
                )
                self.log(f"Saved {filename}")
            except Exception as e:  # noqa: BLE001 -- never kill the loop for a save
                self.log(f"Failed to save {filename}: {e}")

    def save_results(self):
        os.makedirs(
            os.path.dirname(os.path.abspath(self.args.result_filename)), exist_ok=True
        )
        with open(self.args.result_filename, "w") as f:
            yaml.safe_dump(
                {
                    "task": self.task_name,
                    "mode": self.args.mode,
                    "stats": self.stats,
                    "segments": self.segment_stats,
                    "cycles": self.cycle_results,
                },
                f,
                allow_unicode=True,
                sort_keys=False,
            )

    # ---- main loop -----------------------------------------------------------

    # Minimum available system memory [GB]; below it the current recording
    # is cut short and saved (see TeleopBase.MIN_AVAILABLE_MEMORY_GB).
    MIN_AVAILABLE_MEMORY_GB = 1.5

    def run(self):
        self.quit_flag = False
        self.reset_sequence()
        try:
            self.main_loop()
        except KeyboardInterrupt:
            self.log("Interrupted by Ctrl+C.")
        finally:
            self.shutdown()

    def main_loop(self):
        while not self.quit_flag:
            loop_start_time = time.time()

            # Re-anchor the IK warm start on the measured joints every tick
            # (see RolloutBase.main_loop / ArmManager.sync_to_measured).
            self.motion_manager.sync_arm_to_measured(self.obs)

            self.phase.pre_update()

            env_action = np.concatenate(
                [
                    self.motion_manager.get_command_data(key)
                    for key in self.env.unwrapped.command_keys_for_step
                ]
            )

            if self.is_recording():
                if self._is_memory_low():
                    self.log(
                        "Available memory is low -- recording of this attempt stops here."
                    )
                    self.recording_phase = None
                    self.data_manager.reset()
                else:
                    self.record_data()
                    self.record_extra_data()

            self.obs, self.reward, _, _, self.info = self.env.step(env_action)

            self.phase.post_update()

            camera_problem = self.check_camera()
            if camera_problem is not None:
                self.stop_for_operator(camera_problem, "camera")

            self.draw_image()
            self.key = cv2.waitKey(1)
            if self.key == 27:  # escape key
                self.log("Esc pressed -- stopping.")
                self.quit_flag = True
                break
            self.actions = self.poll_actions()
            self.phase.handle_actions(self.actions)

            if self.phase.check_transition():
                self.on_phase_done(self.phase)

            remaining = self.args.control_period - (time.time() - loop_start_time)
            if remaining > 0:
                time.sleep(remaining)

    def _is_memory_low(self):
        return (
            psutil.virtual_memory().available < self.MIN_AVAILABLE_MEMORY_GB * 1024**3
        )

    def check_camera(self):
        """A reason string if a camera has stopped delivering frames."""
        cameras = getattr(self.env.unwrapped, "pointcloud_cameras", {}) or {}
        now = time.time()
        for name, camera in cameras.items():
            last_frame_time = camera.get("last_frame_time")
            if last_frame_time is None:
                continue
            if now - last_frame_time > self.args.camera_timeout:
                return f"camera '{name}' has delivered no frame for {now - last_frame_time:.0f} s"
        return None

    def shutdown(self):
        # Stop the arm where it is before anything else.
        try:
            hold_at_measured(self, refresh_obs=False)
        except Exception as e:  # noqa: BLE001
            print(f"[{self.__class__.__name__}] Could not hold the arm: {e}")
        if self.recording_phase is not None:
            phase = self.recording_phase
            episode = self.take_segment_recording(phase)
            self.commit_episode(
                episode, phase.result or "interrupted", phase.fail_reason
            )
        for episode, _ in self.held_episodes.values():
            self.commit_episode(episode, "success")
        self.held_episodes = {}
        # The cycle in progress (possibly stopped and waiting) is recorded too.
        if self.cycle_result is not None and self.cycle_result["segments"]:
            self.cycle_result["incomplete"] = True
            self.cycle_result["duration"] = round(
                time.time() - self.cycle_start_time, 2
            )
            if isinstance(self.phase, WaitForOperatorPhase):
                self.cycle_result["stopped"] = True
                self.cycle_result["stop_reason"] = self.phase.reason
            self.cycle_results.append(self.cycle_result)
            self.cycle_result = None
        tracker = getattr(self.env.unwrapped, "gripper_marker_tracker", None)
        if tracker is not None and hasattr(tracker, "get_detect_time_ms"):
            detect_ms = tracker.get_detect_time_ms()
            if detect_ms is not None:
                self.log(f"Finger-tag detection: median {detect_ms:.0f} ms per frame.")
        self.save_queue.put(None)
        self.save_thread.join()
        self.save_results()
        self.log(f"Results: {self.args.result_filename}")
        for runner in self.policy_runners.values():
            runner.close()
        if self.teleop_device is not None:
            self.teleop_device.close()
        if self.key_listener is not None:
            self.key_listener.stop()
        self.env.close()

    # ---- display -----------------------------------------------------------

    def get_status_lines(self):
        lines = [
            f"{self.task_name}  [{self.args.mode.upper()}]  cycle {self.cycle_idx + 1}",
            f"segment: {self.phase.display_name}",
        ]
        if self.stats["cycles"] > 0:
            rate = 100.0 * self.stats["cycles_without_stop"] / self.stats["cycles"]
            lines.append(
                f"streak: {self.stats['consecutive_successes']}   "
                f"success: {self.stats['cycles_without_stop']}/{self.stats['cycles']} ({rate:.0f}%)"
            )
        else:
            lines.append("")
        return lines

    def get_camera_image(self):
        images = [
            image
            for image in self.info.get("rgb_images", {}).values()
            if image is not None
        ]
        if len(images) == 0:
            return None
        return images[0]

    def draw_image(self):
        width = self.args.window_width
        image = self.get_camera_image()
        if image is None:
            canvas = np.full((width * 10 // 16, width, 3), 40, dtype=np.uint8)
        else:
            height = int(width * image.shape[0] / image.shape[1])
            canvas = cv2.resize(image, (width, height))
        self.draw_overlay(canvas)

        line_height = 44 if self.args.mode == "exhibit" else 36
        lines = self.get_status_lines()
        panel = np.full((line_height * len(lines) + 16, width, 3), 255, dtype=np.uint8)
        for idx, line in enumerate(lines):
            draw_text(panel, line, (12, 8 + line_height * idx), size=line_height - 8)
        window_image = cv2.vconcat([canvas, panel])
        cv2.imshow("segments", cv2.cvtColor(window_image, cv2.COLOR_RGB2BGR))

    def draw_overlay(self, canvas):
        """Banners on the camera image (stop / intervention)."""
        if isinstance(self.phase, WaitForOperatorPhase):
            height, width = canvas.shape[:2]
            canvas[: height // 3] = (
                canvas[: height // 3] * 0.3 + np.array([200, 30, 30]) * 0.7
            ).astype(np.uint8)
            draw_text(
                canvas,
                "係員を呼んでください",
                (20, 16),
                size=max(width // 16, 24),
                color=(255, 255, 255),
                fallback_text="PLEASE CALL STAFF",
            )
            draw_text(
                canvas,
                self.phase.reason,
                (20, 16 + max(width // 16, 24) + 12),
                size=22,
                color=(255, 255, 255),
            )
