"""Run the SAME real-hardware control loop as bin/Rollout.py (state read from
the real robot, actions sent to the real robot via ServoJ -- MOVES THE ARM),
but substitute the LIVE camera feed with the image sequence recorded for one
training episode, played back one frame per policy re-inference (i.e. every
model_meta_info["data"]["skip"] real steps -- the same cadence the recorded
sequence itself was sampled at). Everything else (state readback, action
execution, safety clamps, --wait_before_start, joint_vel_limit_scale, ...) is
the normal live rollout path, completely unchanged.

Why this test: misc/CheckDiffusionPolicyPrediction.py already showed
offline that (a) the checkpoint reproduces the recorded trajectory when fed
the recorded (state, image) pair, and (b) the recorded STATE sequence alone
nearly determines that trajectory (swapping the image for a frozen/black one
barely changes the prediction). But on real hardware, from the same start
pose, the arm does something qualitatively different from the recorded
trajectory (a fast back-and-forth oscillation / periodic gripper flutter),
not merely an inaccurate version of it. Since the robot's own proprioception
should track the recorded (episode-relative) state reasonably closely if
motion is even roughly working, and the checkpoint itself was already
validated offline, the one remaining variable this script isolates is the
SOURCE of the image feeding the policy.

Interpreting the result:
  - Arm moves approximately like the trained trajectory (reach, close
    gripper, retreat, once) => the LIVE camera pipeline (acquisition,
    format, color order, orientation, timing) is feeding the policy
    something broken. Fix the camera path, not the model or the data.
  - Arm STILL oscillates/flutters => the image was never the problem for
    THIS symptom; look at closed-loop state feedback / retargeting instead
    (e.g. the real trajectory of measured state departing from the recorded
    one, chunk-boundary handling, or the state-shortcut behavior compounding
    badly once the real state timeline isn't in lockstep with training).

SAFETY: identical hazards to bin/Rollout.py -- this moves the real arm.
Use --wait_before_start and a config with a conservative
joint_vel_limit_scale, same as any other real rollout.
"""

import argparse
import importlib
import importlib.util
import os
import sys

import cv2
import numpy as np
import torch
import yaml


def parse_argument():
    env_utils_spec = importlib.util.spec_from_file_location(
        "EnvUtils",
        os.path.join(os.path.dirname(__file__), "..", "common/utils/EnvUtils.py"),
    )
    env_utils_module = importlib.util.module_from_spec(env_utils_spec)
    env_utils_spec.loader.exec_module(env_utils_module)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
        fromfile_prefix_chars="@",
    )
    parser.add_argument("policy", type=str, help="policy (e.g. DiffusionPolicy)")
    parser.add_argument(
        "env",
        type=str,
        help="environment",
        choices=env_utils_module.get_env_names(
            operation_parent_module_str="robo_manip_baselines.envs.operation"
        ),
    )
    parser.add_argument("--config", type=str, help="configuration file")
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint file")
    parser.add_argument(
        "--replay_images_from",
        type=str,
        required=True,
        help="recorded episode (*.rmb/*.hdf5) to source camera images from, "
        "in place of the live camera",
    )
    parser.add_argument(
        "--wait_before_start",
        action="store_true",
        help="whether to wait for a key press before starting motion (passed "
        "through to Rollout)",
    )

    args, remaining_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv
    if args.wait_before_start:
        sys.argv += ["--wait_before_start"]
    if args.checkpoint:
        sys.argv += ["--checkpoint", args.checkpoint]

    return args


class RecordedImageSource:
    """Serves one recorded episode's camera frames, one per call to step(),
    at the same skip-strided cadence the episode itself was sampled at
    during training. Holds the last frame once the episode is exhausted."""

    def __init__(self, rmb_path, model_meta_info):
        from robo_manip_baselines.common import DataKey, RmbData

        skip = model_meta_info["data"]["skip"]
        image_size = model_meta_info["data"]["image_size"]
        camera_names = model_meta_info["image"]["camera_names"]

        with RmbData(rmb_path, image_size=image_size) as rmb_data:
            self.images_seq = {
                camera_name: rmb_data[DataKey.get_rgb_image_key(camera_name)][::skip]
                for camera_name in camera_names
            }
        self.episode_len = next(iter(self.images_seq.values())).shape[0]
        print(
            f"[RecordedImageSource] Loaded {rmb_path}: {self.episode_len} frames "
            f"(skip={skip})"
        )
        self.idx = 0
        self._warned_end = False

    def frame(self, camera_name):
        idx = min(self.idx, self.episode_len - 1)
        if self.idx >= self.episode_len and not self._warned_end:
            print(
                f"[RecordedImageSource] Reached end of recorded episode "
                f"({self.episode_len} frames); holding last frame."
            )
            self._warned_end = True
        return self.images_seq[camera_name][idx]

    def step(self):
        self.idx += 1

    def reset(self):
        self.idx = 0
        self._warned_end = False


def patch_rollout_with_recorded_images(rollout, replay_images_from):
    """Replace rollout.update_images_buf() with a copy of
    RolloutDiffusionPolicy.update_images_buf() that reads from a
    RecordedImageSource instead of self.info["rgb_images"] -- see that
    method for the original this mirrors. Everything else about the
    instance (state, action, env, safety) is untouched."""
    if not hasattr(rollout, "update_images_buf"):
        raise NotImplementedError(
            f"[RolloutWithRecordedImages] {type(rollout).__name__} has no "
            "update_images_buf() -- this script currently only supports "
            "DiffusionPolicy-style policies."
        )

    image_source = RecordedImageSource(replay_images_from, rollout.model_meta_info)

    def patched_update_images_buf():
        if len(rollout.camera_names) == 0:
            raise RuntimeError(
                f"[{type(rollout).__name__}] update_images_buf() requires image "
                "observations."
            )

        images = []
        for camera_name in rollout.camera_names:
            image = image_source.frame(camera_name)
            image = cv2.resize(image, rollout.model_meta_info["data"]["image_size"])
            image = np.moveaxis(image, -1, -3)
            image = torch.tensor(image, dtype=torch.uint8)
            image = rollout.image_transforms(image)
            image = image * 2.0 - 1.0
            images.append(image)

        if rollout.images_buf is None:
            rollout.images_buf = [
                [image for _ in range(rollout.model_meta_info["data"]["n_obs_steps"])]
                for image in images
            ]
        else:
            for single_images_buf, image in zip(rollout.images_buf, images):
                single_images_buf.pop(0)
                single_images_buf.append(image)

        image_source.step()

    rollout.update_images_buf = patched_update_images_buf

    # Also restart image playback from frame 0 at the start of each rollout
    # episode, same as the recorded/state buffers -- see RolloutBase.reset_variables.
    original_reset_variables = rollout.reset_variables

    def patched_reset_variables():
        original_reset_variables()
        image_source.reset()

    rollout.reset_variables = patched_reset_variables


def main():
    args = parse_argument()

    from robo_manip_baselines.common import camel_to_snake, remove_prefix

    operation_module = importlib.import_module(
        f"robo_manip_baselines.envs.operation.Operation{args.env}"
    )
    OperationEnvClass = getattr(operation_module, f"Operation{args.env}")

    policy_module = importlib.import_module(
        f"robo_manip_baselines.policy.{camel_to_snake(args.policy)}"
    )
    RolloutPolicyClass = getattr(policy_module, f"Rollout{args.policy}")

    class Rollout(OperationEnvClass, RolloutPolicyClass):
        @property
        def policy_name(self):
            return remove_prefix(RolloutPolicyClass.__name__, "Rollout")

    if args.config is None:
        config = {}
    else:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    rollout = Rollout(**config)
    patch_rollout_with_recorded_images(rollout, args.replay_images_from)
    rollout.run()


if __name__ == "__main__":
    main()
