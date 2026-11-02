"""Run several trained policies inside one process (one per learned segment).

The existing Rollout* classes (policy/*/Rollout*.py) hold the policy-specific
inference (observation buffers, action chunks, temporal ensembling) as
methods of the rollout OPERATION itself, which therefore can only ever run
one policy. A PolicyRunner wraps one of those classes WITHOUT running its
__init__ (which would parse argv, open the env and build the phase loop): it
creates a bare instance, gives it the attributes its inference methods read
(args, env, motion_manager, model_meta_info, obs, info, ...) and then calls
exactly the same methods RolloutPhase calls -- setup_model_meta_info(),
setup_policy(), reset_variables(), infer_policy(), set_command_data(). The
inference is therefore the existing code, not a copy of it, and produces the
same commands for the same inputs (tests/TestPolicyRunner.py).

Subclasses only name the Rollout class and say how to feed observations
without inferring (observe(), used while a human has taken over).
"""

import argparse
import os
import pickle
import time
from abc import ABC, abstractmethod

import torch

from .SegmentPhases import hold_at_measured


def load_model_meta_info(checkpoint):
    path = os.path.join(os.path.dirname(checkpoint), "model_meta_info.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


class PolicyRunnerBase(ABC):
    # Defaults for every args attribute the wrapped Rollout class may read
    # (RolloutBase.setup_args and the policy's set_additional_args). Values
    # match the command-line defaults, except that the runner's own loop is
    # paced at one policy step per tick (see load()).
    DEFAULT_ARGS = {
        "skip": None,
        "skip_draw": None,
        "max_policy_steps": -1,
        "action_interp": False,
        "no_auto_finish": True,
        "no_plot": True,
        "seed": 0,
        "drop_initial_actions": 0,
        "no_temp_ensem": False,
    }

    def __init__(self, segment_name, policy_args=None, policy_skip=None):
        self.segment_name = segment_name
        self.policy_args = dict(policy_args or {})
        self.policy_skip = policy_skip
        self.shell = None
        self.checkpoint = None
        self._observe_tick = 0

    @classmethod
    @abstractmethod
    def get_rollout_class(cls):
        """The policy's Rollout* class (imported lazily: importing it pulls
        in the policy's third-party code)."""

    @property
    def policy_name(self):
        return self.shell.policy_name

    @property
    def model_meta_info(self):
        return self.shell.model_meta_info

    def load(self, checkpoint, op):
        rollout_class = self.get_rollout_class()
        from ..utils.MiscUtils import remove_prefix

        policy_name = remove_prefix(rollout_class.__name__, "Rollout")
        shell_class = type(
            f"{rollout_class.__name__}Shell",
            (rollout_class,),
            {"policy_name": property(lambda _self: policy_name)},
        )
        shell = object.__new__(shell_class)
        shell.args = argparse.Namespace(
            **{**self.DEFAULT_ARGS, **self.policy_args, "checkpoint": checkpoint}
        )
        shell.env = op.env
        shell.motion_manager = op.motion_manager
        shell.demo_name = f"{op.demo_name}_{self.segment_name}"
        shell.datetime_now = op.datetime_now
        shell._policy_command_log_file = None
        shell._policy_command_log_writer = None
        shell._eef_pose_retargeter = None
        shell.inference_duration_list = []
        shell.obs = getattr(op, "obs", None)
        shell.info = getattr(op, "info", None)

        shell.setup_model_meta_info()
        # The segment loop runs at a fixed control period, one policy step
        # per tick -- the same thing RolloutBase.run() does when a
        # control_period is set and --skip was not given.
        shell.args.skip = 1 if self.policy_skip is None else int(self.policy_skip)
        shell.args.skip_draw = shell.args.skip
        shell.setup_policy()
        shell.setup_variables()
        shell.reset_variables()
        shell.rollout_time_idx = 0

        self.shell = shell
        self.checkpoint = checkpoint
        return self

    def _sync(self, op):
        self.shell.obs = op.obs
        self.shell.info = op.info

    def reset_buffers(self, op):
        """Forget everything from before: observation history, action chunk,
        interpolation state. Called when the policy (re)starts on a segment."""
        self._sync(op)
        self.shell.reset_variables()
        self.shell.setup_eef_pose_retargeter()
        self.shell.rollout_time_idx = 0
        self._observe_tick = 0

    def reset_action_buffers(self, op):
        """Drop the pending action chunk (and ensembling/interpolation state)
        but keep the observation history, which observe() kept up to date --
        used when the policy resumes after a human intervention."""
        self.reset_buffers(op)

    def observe(self, op):
        """Feed this tick's observation without inferring (while a human is
        in control). Policies with no observation history need nothing."""
        self._sync(op)

    def infer_and_command(self, op):
        """One policy tick: infer when due, then set this tick's command --
        what RolloutPhase.pre_update() does."""
        self._sync(op)
        shell = self.shell
        if shell.rollout_time_idx % shell.args.skip == 0:
            inference_start_time = time.time()
            with torch.inference_mode():
                shell.infer_policy()
            shell.inference_duration_list.append(time.time() - inference_start_time)
        shell.set_command_data()
        shell.rollout_time_idx += 1

    def hold_current_command(self, op):
        """Stop following the policy: drop the rest of the action chunk and
        command the arm to stay where it is."""
        self.reset_buffers(op)
        hold_at_measured(op, refresh_obs=False)

    def close(self):
        if self.shell is not None and self.shell._policy_command_log_file is not None:
            self.shell._policy_command_log_file.close()
            self.shell._policy_command_log_file = None


def get_policy_runner_class(policy_name):
    """Runner class for model_meta_info["policy"]["name"]."""
    if policy_name == "DiffusionPolicy":
        from robo_manip_baselines.policy.diffusion_policy.DiffusionPolicyRunner import (
            DiffusionPolicyRunner,
        )

        return DiffusionPolicyRunner
    elif policy_name == "Act":
        from robo_manip_baselines.policy.act.ActRunner import ActRunner

        return ActRunner
    raise ValueError(
        f"[get_policy_runner_class] No segment PolicyRunner for policy '{policy_name}' "
        "(supported: DiffusionPolicy, Act)."
    )


def make_policy_runner(
    segment_name, checkpoint, op, policy_args=None, policy_skip=None
):
    model_meta_info = load_model_meta_info(checkpoint)
    runner_class = get_policy_runner_class(model_meta_info["policy"]["name"])
    return runner_class(segment_name, policy_args, policy_skip).load(checkpoint, op)
