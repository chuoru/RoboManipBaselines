"""Run a task as a cycle of scripted and learned segments.

A task YAML (envs/configs/segments/<Task>.yaml) lists the segments of one
cycle, e.g. for the exhibition task:

    go_home (scripted) -> grasp (learned) -> lift (scripted) -> go_carry
    (scripted) -> place (learned) -> retreat (scripted) -> back_home

and the runner repeats that cycle. Only the parts of the task that vary
(where the cup is, where the coaster is) are learned; everything else is a
fixed move. Each learned segment has a fixed start pose (the scripted segment
before it always brings the arm there), a fixed end condition (the gripper
closed / opened) and a fixed observation, so its policy can be trained and
swapped on its own.

Switching rules (doc/segment_dagger_checklist.md, design D5):
  learned -> scripted: on the end condition, drop the rest of the action
    chunk and hold the measured pose; the next scripted segment starts with
    a relative retreat (where the grasp happened differs every time).
  scripted -> learned: wait for the arm to stand still, re-read the
    observation, re-anchor the IK on it, clear every policy buffer, then
    start inferring.

Usage: bin/RolloutSegments.py.
"""

import argparse
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
from ..utils.GlobalKeyListener import GlobalKeyListener
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

    def __init__(self, op, label, reason):
        super().__init__(op, label)
        self.reason = reason
        self.choice = None

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


class LearnedSegmentPhase(SegmentPhaseBase):
    """A learned segment: settle -> run the policy (or the operator, with
    --replace_learned_with_teleop) until the end condition -> result."""

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

    @property
    def display_name(self):
        return f"{self.label} [{self.state}]"

    def start(self):
        super().start()
        self.state = "settle"
        self.result = None
        self.fail_reason = None
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
                self.finish("success")
            elif self.timeout_s is not None and self.run_time > self.timeout_s:
                self.finish("failure", "timeout")

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
        who = "operator (SpaceMouse)" if self.human_control else "policy"
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
            self.finish("failure", "operator")
        elif "success" in actions:
            self.op.log(f"[{self.label}] Marked as success by the operator.")
            self.finish("success")

    def finish(self, result, fail_reason=None):
        # Learned -> scripted (D5): drop the rest of the chunk, hold here.
        if self.policy_runner is not None:
            self.policy_runner.hold_current_command(self.op)
        else:
            hold_at_measured(self.op, refresh_obs=False)
        self.state = "done"
        self.result = result
        self.fail_reason = fail_reason
        if result == "failure":
            self.op.log(f"[{self.label}] FAILED ({fail_reason}).")

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

        render_mode = None
        self.setup_env(render_mode=render_mode)
        self.demo_name = self.task_name

        self.motion_manager = self.MotionManagerClass(self.env)
        self.data_manager = self.DataManagerClass(self.env, demo_name=self.demo_name)
        self.data_manager.setup_camera_info()
        self.obs, self.info = self.env.reset(seed=self.args.seed)
        self.reward = 0.0

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
            "--replace_learned_with_teleop",
            action="store_true",
            help="drive every learned segment with the SpaceMouse instead of a "
            "policy (end condition or SUCCESS ends it) -- to try the whole "
            "cycle before any policy exists",
        )
        parser.add_argument(
            "--poses",
            type=str,
            default=None,
            help="named pose YAML to use instead of the task YAML's poses "
            "(e.g. ./envs/configs/poses/DryRunExample.yaml for a dry run)",
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
        for idx, spec in enumerate(self.segment_specs):
            if spec.get("type") not in ("scripted_move", "gripper", "learned", "check"):
                raise ValueError(
                    f"Segment '{spec['name']}' has an invalid type: {spec}"
                )
            if spec["type"] != "learned":
                continue
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
        for spec in self.segment_specs:
            if spec["type"] == "learned":
                self.segment_phases.append(self.make_learned_phase(spec))
            else:
                self.segment_phases.append(
                    make_primitive_phase(
                        self, spec, self.named_poses, label=spec["name"]
                    )
                )
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
        self.stats = {
            "cycles": 0,
            "cycles_without_stop": 0,
            "consecutive_successes": 0,
            "stops": 0,
        }
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
        }
        self.cycle_start_time = time.time()

    def next_phase(self):
        if self.pending_phases:
            self.phase = self.pending_phases.pop(0)
        else:
            self.phase = self.segment_phases[self.segment_idx]
        self.phase.start()

    def on_phase_done(self, phase):
        if (
            phase in self.segment_phases
            and phase is self.segment_phases[self.segment_idx]
        ):
            if isinstance(phase, LearnedSegmentPhase):
                self.on_learned_done(phase)
            else:
                self.advance_segment()
        elif isinstance(phase, WaitForOperatorPhase):
            self.on_operator_choice(phase)
        if self.quit_flag:
            return
        self.next_phase()

    def segment_record(self, name):
        return self.cycle_result["segments"].setdefault(
            name, {"result": None, "attempts": 0, "fail_reasons": []}
        )

    def on_learned_done(self, phase):
        record = self.segment_record(phase.label)
        record["attempts"] += 1
        record["result"] = phase.result
        self.finish_segment_recording(phase)
        if phase.result == "success":
            self.advance_segment()
        else:
            record["fail_reasons"].append(phase.fail_reason)
            self.fail_segment(phase, phase.fail_reason)

    def fail_segment(self, learned_phase, reason):
        """Run the learned segment's on_fail steps, then retry it -- or, past
        max_retries, stop at its start pose and wait for the operator."""
        name = learned_phase.label
        self.retry_count[name] = self.retry_count.get(name, 0) + 1
        self.segment_idx = self.segment_phases.index(learned_phase)
        self.pending_phases = list(self.on_fail_phases[name])
        if self.retry_count[name] > learned_phase.max_retries:
            self.pending_phases.append(
                WaitForOperatorPhase(
                    self,
                    "wait",
                    f"'{name}' failed {self.retry_count[name]} times (last: {reason})",
                )
            )
        else:
            self.log(
                f"Retry '{name}' ({self.retry_count[name]}/{learned_phase.max_retries})."
            )

    def on_operator_choice(self, phase):
        self.stats["stops"] += 1
        self.cycle_result["stopped"] = True
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
        self.cycle_result["duration"] = round(time.time() - self.cycle_start_time, 2)
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

    def record_extra_data(self):
        """Per-step data beyond OperationDataMixin's (extended later)."""
        pass

    def finish_segment_recording(self, phase):
        if self.recording_phase is not phase:
            return
        self.recording_phase = None
        if len(self.data_manager.all_data_seq.get(DataKey.TIME, [])) == 0:
            return
        meta_data = dict(self.data_manager.meta_data)
        meta_data.update(self.get_segment_meta_data(phase))
        filename = self.get_segment_filename(phase)
        all_data_seq = self.data_manager.all_data_seq
        self.data_manager.all_data_seq = {}
        # Video encoding takes seconds; do it off the control loop.
        self.save_queue.put((filename, all_data_seq, meta_data))

    def get_segment_meta_data(self, phase):
        name = phase.label
        return {
            "segment": name,
            "segment_result": phase.result,
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

    def get_segment_filename(self, phase):
        dataset_dir = self.args.dataset_dir
        if dataset_dir is None:
            dataset_dir = os.path.join(PACKAGE_DIR, "dataset")
        name = phase.label
        return os.path.normpath(
            os.path.join(
                os.path.expanduser(dataset_dir),
                f"{self.task_name}_{name}_rollout_iter{self.args.iteration:03d}",
                f"{self.task_name}_{name}_{self.datetime_now:%Y%m%d_%H%M%S}_"
                f"c{self.cycle_idx:04d}_r{self.retry_count.get(name, 0)}_{phase.result}.rmb",
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

            if self.recording_phase is not None and self.recording_phase is self.phase:
                if self._is_memory_low():
                    self.log("Available memory is low -- cutting the recording short.")
                    self.finish_segment_recording(self.phase)
                else:
                    self.record_data()
                    self.record_extra_data()

            self.obs, self.reward, _, _, self.info = self.env.step(env_action)

            self.phase.post_update()

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

    def shutdown(self):
        # Stop the arm where it is before anything else.
        try:
            hold_at_measured(self, refresh_obs=False)
        except Exception as e:  # noqa: BLE001
            print(f"[{self.__class__.__name__}] Could not hold the arm: {e}")
        if self.recording_phase is not None:
            phase = self.recording_phase
            if phase.result is None:
                phase.result = "interrupted"
            self.finish_segment_recording(phase)
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
            f"{self.task_name}  [{self.args.mode.upper()}]  cycle {self.cycle_idx}",
            f"segment: {self.phase.display_name}",
        ]
        if self.stats["cycles"] > 0:
            rate = 100.0 * self.stats["cycles_without_stop"] / self.stats["cycles"]
            lines.append(
                f"streak: {self.stats['consecutive_successes']}   "
                f"success: {self.stats['cycles_without_stop']}/{self.stats['cycles']} ({rate:.0f}%)"
            )
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
        panel = np.full((40 * 3 + 20, width, 3), 255, dtype=np.uint8)
        for idx, line in enumerate(self.get_status_lines()):
            cv2.putText(
                panel,
                line,
                (10, 35 + 40 * idx),
                cv2.FONT_HERSHEY_DUPLEX,
                0.9,
                (0, 0, 0),
                2,
            )
        window_image = cv2.vconcat([canvas, panel])
        cv2.imshow("segments", cv2.cvtColor(window_image, cv2.COLOR_RGB2BGR))
