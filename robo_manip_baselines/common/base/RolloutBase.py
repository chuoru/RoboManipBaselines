import argparse
import csv
import datetime
import os
import pickle
import sys
import time
from abc import ABC, abstractmethod

import cv2
import matplotlib

# Select the Agg backend BEFORE importing pyplot, not just inside
# setup_plot(). Every policy's Rollout*.py overrides setup_plot() to build its
# own figure with plt.subplots() and pass it down to RolloutBase.setup_plot()
# -- so that plt.subplots() call runs FIRST, and matplotlib.use("agg") there
# is already too late. On a machine with a display, that first plt.subplots()
# initializes matplotlib's default interactive (Qt) backend, which then
# collides with the Qt that OpenCV loads for cv2.imshow: Qt reports
# "QObject::moveToThread: Current thread is not the object's thread" and then
# fails outright with "Could not load the Qt platform plugin xcb ... even
# though it was found", aborting the process. Rollout never needs an
# interactive figure anyway -- the figure is rasterized via FigureCanvasAgg
# and displayed through cv2.imshow (see setup_plot below).
matplotlib.use("agg")

import matplotlib.pylab as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import yaml
from matplotlib.backends.backend_agg import FigureCanvasAgg
from torchvision.transforms import v2

from ..data.DataKey import DataKey
from ..data.OperationDataMixin import OperationDataMixin
from ..manager.DataManager import DataManager
from ..manager.MotionManager import MotionManager
from ..manager.PhaseManager import PhaseManager
from ..utils.DataUtils import (
    convert_data_from_policy,
    convert_data_to_policy,
    normalize_data,
)
from ..utils.EefPoseRetargetUtils import EpisodeRelativeEefPoseRetargeter
from ..utils.MathUtils import get_pose_from_se3, get_se3_from_pose, set_random_seed
from ..utils.MiscUtils import remove_prefix, remove_suffix
from .PhaseBase import PhaseBase


class InitialRolloutPhase(PhaseBase):
    def start(self):
        super().start()

        if self.op.args.wait_before_start:
            print(f"[{self.op.__class__.__name__}] Press the 'n' key to proceed.")

    def check_transition(self):
        if self.op.args.wait_before_start:
            return self.op.key == ord("n")
        else:
            duration = 1.0  # [s]
            return self.get_elapsed_duration() > duration


class RolloutPhase(PhaseBase):
    def start(self):
        super().start()

        self.op.rollout_time_idx = 0
        self.success_time = None
        # Round-trip completion tracking: these tasks are demonstrated as
        # "leave the start pose, do the thing, come back", so in the
        # episode-relative convention the finish looks like the start (both
        # near the identity pose). Watching for depart-then-return is what
        # tells a finished rollout from one that never started -- see
        # check_transition().
        self.has_departed = False
        self.op.setup_eef_pose_retargeter()
        print(
            f"[{self.op.__class__.__name__}] Start policy rollout. Press the 'n' key to finish policy rollout."
        )

    def pre_update(self):
        if self.op.rollout_time_idx % self.op.args.skip == 0:
            inference_start_time = time.time()
            with torch.inference_mode():
                self.op.infer_policy()
            self.op.inference_duration_list.append(time.time() - inference_start_time)

        self.op.set_command_data()

    def post_update(self):
        if (not self.op.args.no_plot) and (
            self.op.rollout_time_idx % self.op.args.skip_draw == 0
        ):
            self.op.draw_plot()

        self.op.rollout_time_idx += 1

    def check_transition(self):
        elapsed_duration = self.get_elapsed_duration()

        transition_flag = False
        # Number of policy action steps consumed so far. infer_policy() pops
        # one action every args.skip env steps (see pre_update above).
        policy_step_idx = self.op.rollout_time_idx // self.op.args.skip

        if self.op.key == ord("n"):
            transition_flag = True
        elif self.op.check_task_finished(self):
            print(
                f"[{self.op.__class__.__name__}] Terminate the rollout phase "
                "because the end-effector left the start pose and has now "
                "returned to it -- the demonstrated round trip is complete."
            )
            transition_flag = True
        elif (self.op.args.max_policy_steps is not None) and (
            policy_step_idx >= self.op.args.max_policy_steps
        ):
            # A demonstration-trained policy has no notion of "done": it just
            # keeps mapping observations to actions. For a task recorded as a
            # round trip (reach, grasp, return), the end state is by
            # construction close to the start state -- and for an
            # episode-relative EEF convention it is *identical* by definition
            # (both are the identity pose). So the policy reads the finished
            # state as "start of episode" and cheerfully does the whole task
            # again, forever. On real hardware nothing stops it: reward stays
            # 0.0, so auto_exit's success path never fires either.
            #
            # Bound the rollout by how long the task actually took to
            # demonstrate (model_meta_info's max_episode_len, in policy
            # steps) instead -- see setup_model_meta_info().
            print(
                f"[{self.op.__class__.__name__}] Terminate the rollout phase "
                f"because the policy step limit was reached "
                f"({policy_step_idx} >= {self.op.args.max_policy_steps}). This "
                "is a runaway guard, not a task-length estimate -- if the task "
                "was still in progress, raise --max_policy_steps (or -1 to "
                "disable) and check why it is running so slowly."
            )
            transition_flag = True
        elif self.op.args.auto_exit:
            if (self.op.reward >= 1.0) and (self.success_time is None):
                self.success_time = elapsed_duration

            if self.success_time is not None:
                post_success_duration = 1.0  # [s]
                if elapsed_duration > self.success_time + post_success_duration:
                    print(
                        f"[{self.op.__class__.__name__}] Terminate the rollout phase because the environment has been terminated."
                    )
                    transition_flag = True
            else:
                if elapsed_duration > self.op.args.max_duration:
                    print(
                        f"[{self.op.__class__.__name__}] Terminate the rollout phase because the maximum duration has elapsed."
                    )
                    transition_flag = True

        if transition_flag:
            success_str = "success" if self.op.reward >= 1.0 else "failure"
            print(
                # Do not change the following print description, as it will be used
                # to automatically obtain the task success/failure result
                f"Rollout result: {success_str}",
                flush=True,
            )

            self.op.result["success"].append(bool(self.op.reward >= 1.0))
            self.op.result["reward"].append(float(self.op.reward))
            self.op.result["duration"].append(elapsed_duration)

            if self.op.args.save_last_image:
                self.op.save_rgb_image()

            return True
        else:
            return False


class EndRolloutPhase(PhaseBase):
    def start(self):
        super().start()

        msg = f"[{self.op.__class__.__name__}] Policy rollout is finished."
        if not self.op.args.auto_exit:
            msg += " Press the 'n' key to reset."
        print(msg)

    def check_transition(self):
        if (self.op.key == ord("n")) or self.op.args.auto_exit:
            if self.op.args.save_rollout:
                filename = self.op.get_data_filename()
                self.op.data_manager.save_data(filename)
                print(f"[{self.op.__class__.__name__}] Save the data as {filename}")
            else:
                self.op.data_manager.episode_idx += 1
            if self.op.data_manager.episode_idx == len(self.op.args.world_idx_list):
                self.op.quit_flag = True
            else:
                self.op.reset_flag = True

        return False


class RolloutBase(OperationDataMixin, ABC):
    require_task_desc = False
    MotionManagerClass = MotionManager
    DataManagerClass = DataManager

    def __init__(self):
        # Setup arguments
        self.setup_args()

        set_random_seed(self.args.seed)

        # Setup gym environment
        render_mode = None if self.args.no_render else "human"
        self.setup_env(render_mode=render_mode)
        self.demo_name = self.args.demo_name or remove_suffix(self.env.spec.name, "Env")
        if self.args.target_task is not None:
            self.env.unwrapped.target_task = self.args.target_task
        if self.args.world_random_factors is not None:
            self.env.unwrapped.world_random_factors = self.args.world_random_factors

        # Setup policy
        self.setup_model_meta_info()
        self.setup_policy()

        # Setup plot
        if not self.args.no_plot:
            self.setup_plot()

        # Setup motion manager
        self.motion_manager = self.MotionManagerClass(self.env)

        # Setup data manager
        task_desc = self.args.task_desc if self.require_task_desc else ""
        self.data_manager = self.DataManagerClass(
            self.env, demo_name=self.demo_name, task_desc=task_desc
        )
        self.data_manager.setup_camera_info()
        self.datetime_now = datetime.datetime.now()
        self.result = {key: [] for key in ("success", "reward", "duration")}

        # Diagnostic CSV of the policy's predicted command for each action key
        # (in the same physical units as the *_replay_log.csv files written by
        # misc/ReplayUmiOnFairino5.py, so the two are directly comparable)
        # alongside the measured state it was conditioned on. Always on: it's
        # a small per-tick CSV write, independent of any real robot I/O.
        self._policy_command_log_file = None
        self._policy_command_log_writer = None

        # Converts DataKey.MEASURED_EEF_POSE/COMMAND_EEF_POSE between the env's
        # ABSOLUTE end-effector pose (what ArmManager's IK actually needs) and
        # the EPISODE-RELATIVE convention UMI-collected training data uses for
        # those same keys (see envs/real/umi/RealUMIEnvBase.py and
        # misc/ReplayUmiOnFairino5.py's module docstring). None until
        # setup_eef_pose_retargeter() runs at the start of each rollout
        # episode; stays None (no-op) for policies that don't use these keys.
        self._eef_pose_retargeter = None

        # Setup phase manager
        phase_order = [
            InitialRolloutPhase(self),
            *self.get_pre_motion_phases(),
            RolloutPhase(self),
            EndRolloutPhase(self),
        ]
        self.phase_manager = PhaseManager(phase_order)

        self.setup_variables()

    def setup_args(self, parser=None, argv=None):
        if parser is None:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )

        parser.add_argument(
            "--checkpoint", type=str, required=True, help="checkpoint file"
        )

        parser.add_argument(
            "--world_idx",
            type=int,
            default=0,
            help="world index (if '--world_idx_list' option is specified, it takes precedence)",
        )
        parser.add_argument(
            "--world_idx_list",
            type=int,
            nargs="*",
            default=None,
            help="list of world indexes",
        )
        parser.add_argument(
            "--world_idx_repeat_count",
            type=int,
            default=1,
            help="number of times to repeat world indexes",
        )
        parser.add_argument(
            "--world_random_scale",
            nargs="+",
            type=float,
            default=None,
            help="random scale of simulation world (no randomness by default)",
        )
        parser.add_argument(
            "--world_random_factors",
            nargs="*",
            type=str,
            default=None,
            help="list of randomization factors applied to simulation world (no randomness by default)",
        )

        parser.add_argument(
            "--skip",
            type=int,
            help="step interval to infer policy",
        )
        parser.add_argument(
            "--skip_draw",
            type=int,
            help="step interval to draw the plot",
        )

        parser.add_argument("--seed", type=int, default=-1, help="random seed")

        parser.add_argument(
            "--no_render",
            action="store_true",
            help="whether to disable simulation rendering",
        )
        parser.add_argument(
            "--no_plot", action="store_true", help="whether to disable policy plot"
        )
        parser.add_argument(
            "--win_xy_plot",
            type=int,
            nargs=2,
            help="xy position of window to plot policy information",
        )

        parser.add_argument(
            "--wait_before_start",
            action="store_true",
            help="whether to wait a key input before starting motion",
        )
        parser.add_argument(
            "--auto_exit",
            action="store_true",
            help="whether to automatically exit from rollout",
        )
        parser.add_argument(
            "--max_duration",
            type=float,
            default=30.0,
            help=(
                "maximum rollout duration for automatic exit [s] "
                "(used only when '--auto_exit' option is enabled)"
            ),
        )
        parser.add_argument(
            "--time_scale",
            type=float,
            default=None,
            help=(
                "stretch the control period by this factor, slowing the whole "
                "rollout uniformly (2.0 = half speed). Overrides the config "
                "file's time_scale, so speeds can be swept without editing it. "
                "This -- not the velocity clamp -- is the right way to slow a "
                "closed-loop policy down: it scales the observation cadence and "
                "the arm's speed together, leaving the spatial change the policy "
                "sees between observations unchanged. Only environments that "
                "accept time_scale honour this."
            ),
        )
        parser.add_argument(
            "--action_interp",
            action="store_true",
            help=(
                "ramp the command between successive policy waypoints instead "
                "of holding each for --skip env steps. Off by default: measured "
                "on this stack it barely changed the commanded motion "
                "(peak/median step 3.2 -> 3.1, no idle ticks either way, since "
                "max_joint_pos_delta_deg already spreads each waypoint out) "
                "while costing up to one hold window of lag. See "
                "RolloutBase.get_interpolated_policy_action()."
            ),
        )
        parser.add_argument(
            "--no_auto_finish",
            action="store_true",
            help=(
                "do not end the rollout when the end effector completes the "
                "demonstrated round trip (leaves the start pose and returns "
                "to it). See RolloutBase.check_task_finished()."
            ),
        )
        parser.add_argument(
            "--max_policy_steps",
            type=int,
            default=None,
            help=(
                "runaway guard: end the rollout phase after this many policy "
                "action steps. NOT a task-length estimate -- normal completion "
                "is detected by check_task_finished(). A closed-loop rollout "
                "can need many times the demonstrated step count when the arm "
                "lags its command. Default: the longest training episode's "
                "length times MAX_POLICY_STEPS_MARGIN. Pass -1 to disable."
            ),
        )

        parser.add_argument(
            "--save_rollout",
            action="store_true",
            help="whether to save rollout data",
        )

        parser.add_argument(
            "--result_filename",
            type=str,
            default=None,
            help="File path (*.yaml) to save rollout results (default: do not save)",
        )

        parser.add_argument(
            "--save_last_image",
            action="store_true",
            help="whether to save the observation image of the last frame",
        )
        parser.add_argument(
            "--output_image_dir",
            type=str,
            default=".",
            help=(
                "directory to save the output image (default: current directory, "
                "used only when '--output_image_dir' option is enabled)."
            ),
        )

        parser.add_argument(
            "--demo_name", type=str, default="", help="demonstration name"
        )
        parser.add_argument(
            "--target_task", type=str, default=None, help="target task name"
        )
        if self.require_task_desc:
            parser.add_argument(
                "--task_desc", type=str, required=True, help="task description"
            )

        self.set_additional_args(parser)

        if argv is None:
            argv = sys.argv
        self.args = parser.parse_args(argv[1:])

        if self.args.world_idx_list is None:
            self.args.world_idx_list = [self.args.world_idx]
        self.args.world_idx_list *= self.args.world_idx_repeat_count

        if self.args.world_random_scale is not None:
            self.args.world_random_scale = np.array(self.args.world_random_scale)

        if self.args.seed < 0:
            self.args.seed = int(time.time()) % (2**32)

    def set_additional_args(self, parser):
        pass

    def setup_model_meta_info(self):
        checkpoint_dir = os.path.split(self.args.checkpoint)[0]
        model_meta_info_path = os.path.join(checkpoint_dir, "model_meta_info.pkl")
        with open(model_meta_info_path, "rb") as f:
            self.model_meta_info = pickle.load(f)
        print(
            f"[{self.__class__.__name__}] Load model meta info: {model_meta_info_path}"
        )

        # Set state and action information
        self.state_keys = self.model_meta_info["state"]["keys"]
        self.action_keys = self.model_meta_info["action"]["keys"]
        self.camera_names = self.model_meta_info["image"]["camera_names"]
        self.state_dim = len(self.model_meta_info["state"]["example"])
        self.action_dim = len(self.model_meta_info["action"]["example"])

        # Set skip if not specified
        if self.args.skip is None:
            self.args.skip = self.model_meta_info["data"]["skip"]
        if self.args.skip_draw is None:
            self.args.skip_draw = self.args.skip

        # Bound the rollout by the demonstrated task length unless told
        # otherwise -- see RolloutPhase.check_transition() for why a policy
        # with no "done" signal otherwise loops the task forever. Older
        # checkpoints may predate max_episode_len being recorded; leave the
        # limit off in that case rather than guessing.
        if self.args.max_policy_steps is not None and self.args.max_policy_steps < 0:
            self.args.max_policy_steps = None
        elif self.args.max_policy_steps is None:
            max_episode_len = self.model_meta_info["data"].get("max_episode_len")
            if max_episode_len is None:
                self.args.max_policy_steps = None
                print(
                    f"[{self.__class__.__name__}] model_meta_info has no "
                    "max_episode_len, so the rollout is not length-limited. "
                    "Pass --max_policy_steps to bound it."
                )
            else:
                # Deliberately generous. This is a runaway guard, NOT an
                # estimate of how long the task takes: a closed-loop rollout
                # advances through the demonstrated trajectory far more slowly
                # than the demonstration did whenever the arm lags its command
                # (rate limits, contact, a cautious max_joint_pos_delta_deg),
                # because the policy keeps re-planning from the state it
                # actually observes. Measured on this rig: a task demonstrated
                # in 85 policy steps needed >770 with a 0.5 deg/command cap.
                # Sizing this to max_episode_len cut the task off before the
                # arm had really started moving. Completion is detected
                # directly instead -- see check_task_finished().
                self.args.max_policy_steps = int(max_episode_len) * self.MAX_POLICY_STEPS_MARGIN

        # Threshold scale for check_task_finished(), taken from how far the
        # end effector actually travelled across the training demos.
        self._eef_travel_scale = None
        if DataKey.MEASURED_EEF_POSE in self.state_keys:
            state_range = self.model_meta_info["state"].get("range")
            if state_range is not None and len(state_range) >= 3:
                # First three state entries are the EEF translation (see
                # convert_data_to_policy / get_pose9_from_pose7).
                self._eef_travel_scale = float(np.linalg.norm(state_range[:3]))

    # Multiplier applied to the longest training episode to size the runaway
    # guard -- see setup_model_meta_info().
    MAX_POLICY_STEPS_MARGIN = 15

    # check_task_finished() thresholds, as fractions of the EEF travel seen in
    # training. Depart must be cleared before return is watched for, so a
    # rollout that has not started yet is never mistaken for a finished one.
    TASK_DEPART_FRACTION = 0.40
    TASK_RETURN_FRACTION = 0.12
    # Consecutive ticks the return condition must hold, so a trajectory that
    # merely passes near the start pose does not end the rollout.
    TASK_RETURN_HOLD_TICKS = 10

    def check_task_finished(self, phase):
        """True once the end effector has left the start pose and come back.

        These tasks are demonstrated as a round trip, so in the
        episode-relative convention a finished rollout looks exactly like an
        unstarted one -- both sit at the identity pose. That ambiguity is why
        the policy otherwise repeats the task forever (it reads the finished
        state as "start of episode"). Requiring a departure first
        disambiguates the two, and gives a real completion signal instead of
        guessing a step count, which cannot work when execution speed varies
        with how well the arm tracks its command.

        Returns False (never terminates) when the episode-relative convention
        is not in use, or when the training stats needed to size the
        thresholds are unavailable.
        """
        if self.args.no_auto_finish:
            return False
        if self._eef_pose_retargeter is None or self._eef_travel_scale is None:
            return False

        rel_pose = self.get_measured_data_for_policy(DataKey.MEASURED_EEF_POSE)
        displacement = float(np.linalg.norm(rel_pose[:3]))

        if not phase.has_departed:
            if displacement > self.TASK_DEPART_FRACTION * self._eef_travel_scale:
                phase.has_departed = True
                self._task_return_ticks = 0
            return False

        if displacement < self.TASK_RETURN_FRACTION * self._eef_travel_scale:
            self._task_return_ticks = getattr(self, "_task_return_ticks", 0) + 1
        else:
            self._task_return_ticks = 0

        return self._task_return_ticks >= self.TASK_RETURN_HOLD_TICKS

    @abstractmethod
    def setup_policy(self):
        pass

    def setup_env(self):
        raise NotImplementedError(
            f"[{self.__class__.__name__}] This method should be defined in the Operation class and inherited from it."
        )

    def setup_plot(self, fig_ax=None):
        matplotlib.use("agg")

        if fig_ax is None:
            self.fig, self.ax = plt.subplots(
                1, 1, figsize=(13.5, 6.0), dpi=60, squeeze=False
            )
        else:
            self.fig, self.ax = fig_ax

        for _ax in np.ravel(self.ax):
            _ax.cla()
            _ax.axis("off")

        self.canvas = FigureCanvasAgg(self.fig)
        self.canvas.draw()
        cv2.imshow(
            self.policy_name,
            cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR),
        )

        if self.args.win_xy_plot is not None:
            cv2.moveWindow(self.policy_name, *self.args.win_xy_plot)
        cv2.waitKey(1)

        if len(self.action_keys) > 0:
            self.action_plot_scale = np.concatenate(
                [
                    DataKey.get_plot_scale_for_policy(key, self.env)
                    for key in self.action_keys
                ]
            )
        else:
            self.action_plot_scale = np.zeros(0)

    def setup_variables(self):
        self.image_transforms = v2.Compose([v2.ToDtype(torch.float32, scale=True)])

    def reset_variables(self):
        self.policy_action_list = np.empty((0, self.action_dim))
        # Endpoints of the waypoint ramp -- see get_interpolated_policy_action().
        self._interp_from = None
        self._interp_to = None

    def get_pre_motion_phases(self):
        return []

    def print_policy_info(self):
        print(
            f"[{self.__class__.__name__}] Construct {self.policy_name} policy.\n"
            f"  - state dim: {self.state_dim}, action dim: {self.action_dim}, camera num: {len(self.camera_names)}\n"
            f"  - state keys: {self.state_keys}\n"
            f"  - action keys: {self.action_keys}\n"
            f"  - camera names: {self.camera_names}\n"
            f"  - skip: {self.args.skip}"
        )

    def load_ckpt(self, device="cuda"):
        print(f"[{self.__class__.__name__}] Load {self.args.checkpoint}")
        self.device = torch.device(device)
        self.policy.load_state_dict(
            torch.load(
                self.args.checkpoint, map_location=self.device, weights_only=True
            )
        )
        self.policy.to(self.device)
        self.policy.eval()

    def run(self):
        self.reset_flag = True
        self.quit_flag = False
        self.inference_duration_list = []

        while True:
            if self.reset_flag:
                self.reset()
                self.reset_flag = False

            # Re-anchor the IK warm-start to the arm's actual measured
            # position before computing this tick's command -- see
            # ArmManager.sync_to_measured()'s docstring for why this must
            # happen every tick on real hardware (prevents the commanded
            # trajectory from winding up away from what the arm can actually
            # follow under overwrite_command_for_safety's velocity clamp).
            self.motion_manager.sync_arm_to_measured(self.obs)

            self.phase_manager.pre_update()

            env_action = np.concatenate(
                [
                    self.motion_manager.get_command_data(key)
                    for key in self.env.unwrapped.command_keys_for_step
                ]
            )

            if self.args.save_rollout and self.phase_manager.is_phase("RolloutPhase"):
                self.record_data()

            self.obs, self.reward, _, _, self.info = self.env.step(env_action)

            self.phase_manager.post_update()

            if self.args.no_plot:
                self.key = -1
            else:
                self.key = cv2.waitKey(1)

            self.phase_manager.check_transition()

            if self.key == 27:  # escape key
                self.quit_flag = True
            if self.quit_flag:
                break

        if self.args.result_filename is not None:
            print(
                f"[{self.__class__.__name__}] Save the rollout results: {self.args.result_filename}"
            )
            with open(self.args.result_filename, "w") as result_file:
                yaml.dump(self.result, result_file)

        self.print_statistics()

        if self._policy_command_log_file is not None:
            self._policy_command_log_file.close()

        # self.env.close()

    def reset(self):
        # Reset plot
        if not self.args.no_plot:
            for _ax in np.ravel(self.ax):
                _ax.cla()
                _ax.axis("off")

            self.canvas = FigureCanvasAgg(self.fig)
            self.canvas.draw()
            cv2.imshow(
                self.policy_name,
                cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR),
            )

        # Reset motion manager
        self.motion_manager.reset()

        # Reset data manager
        self.data_manager.reset()

        # Reset environment
        self.env.unwrapped.world_random_scale = self.args.world_random_scale
        world_idx = self.args.world_idx_list[self.data_manager.episode_idx]
        self.data_manager.setup_env_world(world_idx)
        self.obs, self.info = self.env.reset(seed=self.args.seed)
        self.reward = 0
        msg = f"[{self.__class__.__name__}] Reset environment. demo_name: {self.demo_name}, world_idx: {self.data_manager.world_idx}, episode_idx: {self.data_manager.episode_idx}"
        if self.require_task_desc:
            msg += f", task desc: {self.args.task_desc}"
        print(msg)

        # Reset phase manager
        self.phase_manager.reset()

        # Reset variables
        self.reset_variables()

    @abstractmethod
    def infer_policy(self):
        pass

    def get_state(self):
        if len(self.state_keys) == 0:
            state = np.zeros(0, dtype=np.float32)
        else:
            state = np.concatenate(
                [
                    convert_data_to_policy(
                        self.get_measured_data_for_policy(state_key), state_key
                    )
                    for state_key in self.state_keys
                ]
            )

        state = normalize_data(state, self.model_meta_info["state"])
        state = torch.tensor(state[np.newaxis], dtype=torch.float32).to(self.device)

        return state

    def get_measured_data_for_policy(self, state_key):
        measured_data = self.motion_manager.get_data(state_key, self.obs)

        if state_key == DataKey.MEASURED_EEF_POSE and self._eef_pose_retargeter is not None:
            abs_se3 = get_se3_from_pose(measured_data)
            rel_se3 = self._eef_pose_retargeter.to_episode_relative(abs_se3)
            measured_data = get_pose_from_se3(rel_se3)

        return measured_data

    def setup_eef_pose_retargeter(self):
        self._eef_pose_retargeter = None

        if (DataKey.MEASURED_EEF_POSE not in self.state_keys) and (
            DataKey.COMMAND_EEF_POSE not in self.action_keys
        ):
            return

        # Reference pose the episode's own convention starts from (identity),
        # matching how UMI-collected training data is recorded -- see
        # EpisodeRelativeEefPoseRetargeter's docstring.
        ref_se3 = get_se3_from_pose(
            self.motion_manager.get_data(DataKey.MEASURED_EEF_POSE, self.obs)
        )
        self._eef_pose_retargeter = EpisodeRelativeEefPoseRetargeter(ref_se3)

    def get_images(self):
        # Assume all images are the same size
        images = np.stack(
            [self.info["rgb_images"][camera_name] for camera_name in self.camera_names],
            axis=0,
        )

        images = np.moveaxis(images, -1, -3)
        images = torch.tensor(images, dtype=torch.uint8)
        images = self.image_transforms(images)[torch.newaxis].to(self.device)

        return images

    @abstractmethod
    def draw_plot(self):
        pass

    def get_interpolated_policy_action(self):
        """Policy action for this tick, ramped toward the current waypoint.

        infer_policy() only advances the policy action once every args.skip
        env steps, so the commanded pose is a staircase: it jumps to the new
        waypoint, the arm converges on it well inside the hold window, and
        then sits still until the next one. At the trained cadence the steps
        are short enough not to matter, but they scale with --time_scale --
        at 8.0 each waypoint is held for skip * 0.02 * 8 = 0.48 s, which is
        plainly visible as move-stop-move-stop.

        Ramping the command linearly from the previous waypoint to the
        current one across the hold window turns that into continuous
        motion. It costs up to one window of lag, which is exactly the
        tradeoff that makes the motion smooth.

        The rotation is interpolated in the policy's own 6D representation
        and re-orthonormalised downstream by get_pose7_from_pose9(), so the
        result is always a valid rotation. The gripper is deliberately NOT
        interpolated: it is a binary tool-DO gripper here, and ramping the
        commanded percentage would just smear its open/close edge.
        """
        if not self.args.action_interp:
            return self.policy_action

        skip = self.args.skip
        if skip <= 1:
            return self.policy_action

        if self.rollout_time_idx % skip == 0:
            # New waypoint this tick: ramp from wherever the last window
            # finished (the previous waypoint), or from this one on the very
            # first window, where there is nothing to ramp from.
            self._interp_from = (
                self.policy_action.copy()
                if self._interp_to is None
                else self._interp_to
            )
            self._interp_to = self.policy_action.copy()

        if self._interp_to is None:
            return self.policy_action

        alpha = ((self.rollout_time_idx % skip) + 1) / skip
        action = (1.0 - alpha) * self._interp_from + alpha * self._interp_to

        # Keep non-pose components (gripper) on the un-ramped waypoint.
        action_idx = 0
        for key in self.action_keys:
            action_dim = DataKey.get_dim_for_policy(key, self.env)
            if key not in (DataKey.MEASURED_EEF_POSE, DataKey.COMMAND_EEF_POSE):
                action[action_idx : action_idx + action_dim] = self._interp_to[
                    action_idx : action_idx + action_dim
                ]
            action_idx += action_dim

        return action

    def set_command_data(self, action_keys=None):
        if action_keys is None:
            action_keys = self.action_keys

        is_skip = self.rollout_time_idx % self.args.skip != 0
        policy_action = self.get_interpolated_policy_action()
        action_idx = 0
        command_by_key = {}
        for key in action_keys:
            action_dim = DataKey.get_dim_for_policy(key, self.env)
            command = convert_data_from_policy(
                policy_action[action_idx : action_idx + action_dim], key
            )
            # Logged/kept in the policy's own episode-relative convention;
            # motion_manager_command is what ArmManager's IK actually needs.
            command_by_key[key] = command
            motion_manager_command = command
            if key == DataKey.COMMAND_EEF_POSE and self._eef_pose_retargeter is not None:
                rel_se3 = get_se3_from_pose(command)
                abs_se3 = self._eef_pose_retargeter.to_absolute(rel_se3)
                motion_manager_command = get_pose_from_se3(abs_se3)
            self.motion_manager.set_command_data(
                key,
                motion_manager_command,
                is_skip,
            )
            action_idx += action_dim

        self.log_policy_command_debug(command_by_key, is_skip)

    # Column suffixes for keys whose physical meaning is worth naming
    # explicitly (matches the *_replay_log.csv column names written by
    # misc/ReplayUmiOnFairino5.py). Any other key falls back to numeric
    # suffixes ("_0", "_1", ...).
    _POLICY_COMMAND_LOG_SUFFIXES = {
        DataKey.COMMAND_EEF_POSE: ["tx", "ty", "tz", "qw", "qx", "qy", "qz"],
        DataKey.MEASURED_EEF_POSE: ["tx", "ty", "tz", "qw", "qx", "qy", "qz"],
        DataKey.COMMAND_GRIPPER_JOINT_POS: ["pos"],
        DataKey.MEASURED_GRIPPER_JOINT_POS: ["pos"],
    }

    def _policy_command_log_columns(self, key, dim):
        suffixes = self._POLICY_COMMAND_LOG_SUFFIXES.get(key)
        if suffixes is None or len(suffixes) != dim:
            suffixes = [str(i) for i in range(dim)]
        return [f"{key}_{suffix}" for suffix in suffixes]

    def log_policy_command_debug(self, command_by_key, is_skip):
        if len(command_by_key) == 0:
            return

        measured_by_key = {
            key.replace("command_", "measured_", 1): self.get_measured_data_for_policy(
                key.replace("command_", "measured_", 1)
            )
            for key in command_by_key
        }

        if self._policy_command_log_writer is None:
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(
                log_dir,
                f"policy_command_log_{self.policy_name}_{self.demo_name}_"
                f"{self.datetime_now:%Y%m%d_%H%M%S}.csv",
            )
            self._policy_command_log_file = open(log_path, "w", newline="")
            self._policy_command_log_writer = csv.writer(self._policy_command_log_file)
            header = ["t", "rollout_time_idx", "is_skip"]
            for key, command in command_by_key.items():
                header += self._policy_command_log_columns(key, len(command))
            for key, measured in measured_by_key.items():
                header += self._policy_command_log_columns(key, len(measured))
            self._policy_command_log_writer.writerow(header)
            print(
                f"[{self.__class__.__name__}] Logging policy command/measured "
                f"state to {log_path}"
            )

        row = [self.env.unwrapped.get_time(), self.rollout_time_idx, is_skip]
        for command in command_by_key.values():
            row += [f"{v:.6f}" for v in command]
        for measured in measured_by_key.values():
            row += [f"{v:.6f}" for v in measured]
        self._policy_command_log_writer.writerow(row)
        self._policy_command_log_file.flush()

    def get_data_filename(self):
        filename = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "dataset",
                f"Rollout{self.policy_name}_{self.demo_name}_{self.datetime_now:%Y%m%d_%H%M%S}",
                f"{self.demo_name}_world{self.data_manager.world_idx:0>1}_{self.data_manager.episode_idx:0>3}.rmb",
            )
        )
        return filename

    def plot_images(self, axes):
        for camera_idx, camera_name in enumerate(self.camera_names):
            axes[camera_idx].imshow(self.info["rgb_images"][camera_name])
            axes[camera_idx].set_title(camera_name, fontsize=20)

    def get_action_plot_labels(self):
        """One label per action dimension, for plot_action's legend.

        Names come from the action keys themselves, so the legend matches
        whatever --action_keys the policy was trained with. Note the policy
        dimension is not always the raw data dimension: an EEF pose is stored
        as 7 (xyz + quaternion) but fed to the policy as 9 (xyz + 6D rotation),
        which is what get_dim_for_policy reports and what these lines actually
        are -- so a pose contributes labels xyz then rot0..rot5 rather than a
        quaternion's wxyz.
        """
        labels = []
        for key in self.action_keys:
            dim = DataKey.get_dim_for_policy(key, self.env)
            # Trim the redundant "command_"/"measured_" prefix; every action
            # key carries the same one, so it only eats legend width.
            name = remove_prefix(remove_prefix(key, "command_"), "measured_")
            if key in (DataKey.MEASURED_EEF_POSE, DataKey.COMMAND_EEF_POSE):
                num_eef = DataKey.get_num_eef(self.env)
                per_eef = ["x", "y", "z"] + [f"rot{i}" for i in range(6)]
                for eef_idx in range(num_eef):
                    prefix = f"eef{eef_idx}_" if num_eef > 1 else ""
                    labels += [f"{prefix}{component}" for component in per_eef]
            elif dim == 1:
                labels.append(name)
            else:
                labels += [f"{name}[{i}]" for i in range(dim)]
        return labels

    def plot_action(self, ax):
        history_size = 100
        lines = ax.plot(
            self.policy_action_list[-1 * history_size :] * self.action_plot_scale
        )
        ax.set_title("action", fontsize=20)
        ax.set_xlabel("step", fontsize=16)
        ax.set_xlim(0, history_size - 1)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)
        ax.axis("on")

        # Without this the plot is an anonymous bundle of lines -- which one is
        # the gripper, which is Z? Placed outside the axes so it never covers
        # the traces, and multi-column so a 10-dim action stays readable.
        labels = self.get_action_plot_labels()
        if len(labels) == len(lines):
            ax.legend(
                lines,
                labels,
                fontsize=9,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0.0,
                ncol=max(1, (len(labels) + 9) // 10),
                framealpha=0.9,
            )

    def print_statistics(self):
        print(f"[{self.__class__.__name__}] Statistics on policy inference")
        policy_model_size = self.calc_model_size()
        print(f"  - Policy model size [MB] | {policy_model_size / 1024**2:.2f}")
        gpu_memory_usage = torch.cuda.max_memory_reserved()
        print(f"  - GPU memory usage [GB] | {gpu_memory_usage / 1024**3:.3f}")
        inference_duration_arr = np.array(self.inference_duration_list)
        print(
            "  - Inference duration [s] | "
            f"mean: {inference_duration_arr.mean():.2e}, std: {inference_duration_arr.std():.2e} "
            f"min: {inference_duration_arr.min():.2e}, max: {inference_duration_arr.max():.2e}"
        )

    def save_rgb_image(self):
        image = cv2.hconcat(list(self.info["rgb_images"].values()))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        success_str = "success" if self.reward >= 1.0 else "failure"
        image_path = os.path.abspath(
            os.path.join(
                self.args.output_image_dir,
                (
                    f"Rollout{self.policy_name}_{self.demo_name}_world{self.data_manager.world_idx:0>1}_"
                    f"{self.data_manager.episode_idx:0>3}_{success_str}_{self.datetime_now:%Y%m%d_%H%M%S}.png"
                ),
            )
        )

        print(
            f"[{self.__class__.__name__}] Save the observation image of the last frame: {image_path}"
        )
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        cv2.imwrite(image_path, image)

    def calc_model_size(self):
        # https://discuss.pytorch.org/t/finding-model-size/130275/2
        param_size = 0
        for param in self.policy.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.policy.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return param_size + buffer_size
