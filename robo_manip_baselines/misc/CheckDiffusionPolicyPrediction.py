"""Offline sanity check for a trained DiffusionPolicy checkpoint: replay a
recorded episode's own images/states through the policy (exactly the same
preprocessing RolloutDiffusionPolicy uses at inference time) and compare the
predicted actions against what was actually recorded, with no robot involved.

Motivation: if a real rollout ignores what the camera sees (same repeating
motion regardless of the scene), there are several very different possible
causes -- (a) the policy itself never learned to condition on the image at
all (a shortcut-learning/training-data problem), or (b) the rollout-time
image/state pipeline is feeding it something broken (stale camera frames,
wrong retargeting, preprocessing mismatch -- a rollout-time bug).

IMPORTANT CAVEAT about the default (--image_mode real) check: reproducing
the recorded action from the recorded (state, image) pair is NOT proof the
model is using the image. Within a single demonstration, the recorded STATE
at time t is itself a strong predictor of the recorded action at time t (it
effectively encodes "how far into this one trajectory we are"), so a model
that has learned to ignore the image entirely and condition only on state
can still pass this check. Use --image_mode freeze/black to actually test
image sensitivity: they keep the recorded state sequence (and hence the
recorded action-vs-state correlation) intact but corrupt the image, either
by repeating the very first frame for the whole episode (freeze -- the
"stale/frozen camera" failure mode) or replacing it with a constant image
(black -- removes all image information outright). If predictions barely
change between real/freeze/black, the policy is not meaningfully using the
image and the fix belongs in training/data, not the rollout code.

This is a teacher-forced, open-loop replay: every observation window's
STATE comes from the recording (never from a previous prediction), matching
how DiffusionPolicyDataset builds training windows -- see
policy/diffusion_policy/DiffusionPolicyDataset.py and DpStyleDatasetMixin.
It only validates "does the model reproduce recorded behavior / react to the
image given recorded state", not closed-loop rollout stability.
"""

import argparse
import os

import matplotlib.pylab as plt
import numpy as np
import torch
from torchvision.transforms import v2

from robo_manip_baselines.common import (
    DataKey,
    RmbData,
    convert_data_from_policy,
    convert_data_to_policy,
    denormalize_data,
    get_skipped_data_seq,
    normalize_data,
)


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("rmb_path", type=str, help="recorded episode (*.rmb/*.hdf5)")
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint file")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--image_mode",
        type=str,
        default="all",
        choices=["real", "freeze", "black", "all"],
        help="'real': recorded images (state-alone confound, see module "
        "docstring). 'freeze': every timestep uses the episode's FIRST frame "
        "(simulates a stale/frozen camera). 'black': constant image (removes "
        "all image information). 'all': run and compare all three -- the "
        "decisive check.",
    )
    parser.add_argument(
        "--save_plot",
        type=str,
        default=None,
        help="output PNG path (default: '<rmb_basename>_policy_check.png' "
        "next to rmb_path)",
    )
    return parser.parse_args()


def build_policy(model_meta_info, checkpoint_path, device):
    if "backbone" not in model_meta_info["policy"]:
        model_meta_info["policy"]["backbone"] = "cnn"
    if "scheduler" not in model_meta_info["policy"]:
        model_meta_info["policy"]["scheduler"] = "ddpm"

    if model_meta_info["policy"]["scheduler"] == "ddpm":
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

        noise_scheduler = DDPMScheduler(
            **model_meta_info["policy"]["noise_scheduler_args"]
        )
    elif model_meta_info["policy"]["scheduler"] == "ddim":
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler

        noise_scheduler = DDIMScheduler(
            **model_meta_info["policy"]["noise_scheduler_args"]
        )
    else:
        raise ValueError(
            f"Invalid scheduler: {model_meta_info['policy']['scheduler']}"
        )

    if len(model_meta_info["image"]["camera_names"]) == 0:
        raise ValueError(
            "This script only handles image-conditioned policies (camera_names is empty)."
        )
    if model_meta_info["policy"]["backbone"] != "cnn":
        raise ValueError(
            f"Only the 'cnn' backbone is currently supported by this script, "
            f"got: {model_meta_info['policy']['backbone']}"
        )

    from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
        DiffusionUnetHybridImagePolicy,
    )

    policy = DiffusionUnetHybridImagePolicy(
        noise_scheduler=noise_scheduler, **model_meta_info["policy"]["args"]
    )
    policy.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    policy.to(device)
    policy.eval()
    return policy


def load_episode(rmb_path, model_meta_info):
    skip = model_meta_info["data"]["skip"]
    image_size = model_meta_info["data"]["image_size"]
    state_keys = model_meta_info["state"]["keys"]
    action_keys = model_meta_info["action"]["keys"]
    camera_names = model_meta_info["image"]["camera_names"]

    with RmbData(rmb_path, image_size=image_size) as rmb_data:
        # State: same per-key conversion (pose7 -> pose9) as RolloutBase.get_state,
        # concatenated in the same key order as model_meta_info["state"]["keys"].
        state_seq = np.concatenate(
            [
                convert_data_to_policy(
                    get_skipped_data_seq(rmb_data[key][:], key, skip), key
                )
                for key in state_keys
            ],
            axis=1,
        )

        # Action kept in RAW physical units (pose7 / percent-closed) for a
        # direct, human-readable comparison against the model's (denormalized,
        # convert_data_from_policy'd) predictions -- see main().
        action_raw_seq = {
            key: get_skipped_data_seq(rmb_data[key][:], key, skip)
            for key in action_keys
        }

        images_seq = {
            camera_name: rmb_data[DataKey.get_rgb_image_key(camera_name)][::skip]
            for camera_name in camera_names
        }

    return state_seq, action_raw_seq, images_seq


def main():
    args = parse_argument()

    checkpoint_dir = os.path.split(args.checkpoint)[0]
    model_meta_info_path = os.path.join(checkpoint_dir, "model_meta_info.pkl")
    import pickle

    with open(model_meta_info_path, "rb") as f:
        model_meta_info = pickle.load(f)
    print(f"[CheckDiffusionPolicyPrediction] Load model meta info: {model_meta_info_path}")

    skip = model_meta_info["data"]["skip"]
    horizon = model_meta_info["data"]["horizon"]
    n_obs_steps = model_meta_info["data"]["n_obs_steps"]
    n_action_steps = model_meta_info["data"]["n_action_steps"]
    state_keys = model_meta_info["state"]["keys"]
    action_keys = model_meta_info["action"]["keys"]
    camera_names = model_meta_info["image"]["camera_names"]
    print(
        f"  - skip: {skip}, horizon: {horizon}, n_obs_steps: {n_obs_steps}, "
        f"n_action_steps: {n_action_steps}\n"
        f"  - state keys: {state_keys}\n"
        f"  - action keys: {action_keys}\n"
        f"  - camera names: {camera_names}"
    )

    device = torch.device(args.device)
    policy = build_policy(model_meta_info, args.checkpoint, device)
    print(f"[CheckDiffusionPolicyPrediction] Loaded {args.checkpoint}")

    state_seq, action_raw_seq, images_seq = load_episode(args.rmb_path, model_meta_info)
    episode_len = state_seq.shape[0]
    print(
        f"[CheckDiffusionPolicyPrediction] {args.rmb_path}: {episode_len} "
        f"skipped timesteps ({episode_len * skip} raw steps)"
    )

    state_norm = normalize_data(state_seq, model_meta_info["state"])

    image_transforms = v2.Compose([v2.ToDtype(torch.float32, scale=True)])

    def to_model_image(frame_hwc_uint8):
        # HWC uint8 -> CHW float32 in [-1, 1], matching training's
        # DiffusionPolicyDataset.augment_data / RolloutDiffusionPolicy.update_images_buf.
        frame = np.moveaxis(frame_hwc_uint8, -1, -3)
        frame = torch.tensor(frame, dtype=torch.uint8)
        frame = image_transforms(frame)
        return frame * 2.0 - 1.0

    def get_image_window(camera_name, t0, t1, mode):
        if mode == "real":
            window = images_seq[camera_name][t0:t1]
        elif mode == "freeze":
            # Every timestep repeats the episode's first frame -- the "camera
            # stopped updating" failure mode, with state still progressing.
            first_frame = images_seq[camera_name][0]
            window = np.stack([first_frame] * (t1 - t0), axis=0)
        elif mode == "black":
            window = np.zeros_like(images_seq[camera_name][t0:t1])
        else:
            raise ValueError(f"Unknown image_mode: {mode}")
        return to_model_image(window)

    def predict_episode(image_mode):
        # Non-overlapping chunks, each predicted from a teacher-forced
        # (recorded-state) observation window -- mirrors how RolloutPhase
        # re-infers once its n_action_steps buffer is empty, except state here
        # always comes from the recording instead of the robot's own
        # (possibly drifting) state. Only the image is substituted per mode.
        predicted_by_t = {}
        t0 = n_obs_steps - 1
        while t0 + n_action_steps <= episode_len:
            obs_state = torch.tensor(
                state_norm[t0 - n_obs_steps + 1 : t0 + 1], dtype=torch.float32
            )[None].to(device)
            input_data = {"state": obs_state}
            for camera_name in camera_names:
                input_data[DataKey.get_rgb_image_key(camera_name)] = get_image_window(
                    camera_name, t0 - n_obs_steps + 1, t0 + 1, image_mode
                )[None].to(device)

            with torch.inference_mode():
                action_chunk = policy.predict_action(input_data)["action"][0]
            action_chunk = action_chunk.cpu().numpy().astype(np.float64)

            for i in range(action_chunk.shape[0]):
                t = t0 + 1 + i
                if t >= episode_len:
                    break
                action_denorm = denormalize_data(
                    action_chunk[i], model_meta_info["action"]
                )
                action_idx = 0
                action_by_key = {}
                for key in action_keys:
                    key_dim = (
                        9
                        if key in (DataKey.MEASURED_EEF_POSE, DataKey.COMMAND_EEF_POSE)
                        else 1
                    )
                    action_by_key[key] = convert_data_from_policy(
                        action_denorm[action_idx : action_idx + key_dim], key
                    )
                    action_idx += key_dim
                predicted_by_t[t] = action_by_key

            t0 += n_action_steps

        return predicted_by_t

    image_modes = ["real", "freeze", "black"] if args.image_mode == "all" else [args.image_mode]
    predicted_by_mode = {}
    for image_mode in image_modes:
        predicted_by_mode[image_mode] = predict_episode(image_mode)
        n_predicted = len(predicted_by_mode[image_mode])
        print(
            f"[CheckDiffusionPolicyPrediction] [{image_mode}] Predicted "
            f"{n_predicted}/{episode_len} timesteps"
        )

    ts = sorted(predicted_by_mode[image_modes[0]].keys())
    eef_key = DataKey.COMMAND_EEF_POSE if DataKey.COMMAND_EEF_POSE in action_keys else None
    gripper_key = (
        DataKey.COMMAND_GRIPPER_JOINT_POS
        if DataKey.COMMAND_GRIPPER_JOINT_POS in action_keys
        else None
    )

    def trans_rot_err(pose_a, pose_b):
        trans_err = np.linalg.norm(pose_a[:, :3] - pose_b[:, :3], axis=1)
        dot = np.clip(np.abs(np.sum(pose_a[:, 3:] * pose_b[:, 3:], axis=1)), -1.0, 1.0)
        rot_err_deg = np.rad2deg(2.0 * np.arccos(dot))
        return trans_err, rot_err_deg

    # Metrics vs. the recorded ground truth, per image_mode.
    print("\n[CheckDiffusionPolicyPrediction] Prediction error vs. RECORDED action:")
    for image_mode in image_modes:
        predicted_by_t = predicted_by_mode[image_mode]
        print(f"  [{image_mode}]")
        if eef_key is not None:
            recorded = action_raw_seq[eef_key][ts]
            predicted = np.stack([predicted_by_t[t][eef_key] for t in ts], axis=0)
            trans_err, rot_err_deg = trans_rot_err(recorded, predicted)
            print(
                f"    {eef_key}: translation RMSE {np.sqrt(np.mean(trans_err**2)):.4f} m, "
                f"rotation RMSE {np.sqrt(np.mean(rot_err_deg**2)):.2f} deg"
            )
        if gripper_key is not None:
            recorded = action_raw_seq[gripper_key][ts][:, 0]
            predicted = np.array([predicted_by_t[t][gripper_key][0] for t in ts])
            print(
                f"    {gripper_key}: RMSE {np.sqrt(np.mean((recorded - predicted) ** 2)):.2f} %"
            )

    # The decisive comparison: how much do predictions themselves change
    # across image_mode, given the IDENTICAL recorded state sequence? Small
    # numbers here mean the image isn't meaningfully influencing the output.
    if len(image_modes) > 1:
        print(
            "\n[CheckDiffusionPolicyPrediction] Prediction difference BETWEEN "
            "image_modes (same recorded state each time -- small numbers mean "
            "the image is NOT meaningfully influencing the output):"
        )
        baseline_mode = "real" if "real" in image_modes else image_modes[0]
        for image_mode in image_modes:
            if image_mode == baseline_mode:
                continue
            if eef_key is not None:
                pred_a = np.stack(
                    [predicted_by_mode[baseline_mode][t][eef_key] for t in ts], axis=0
                )
                pred_b = np.stack(
                    [predicted_by_mode[image_mode][t][eef_key] for t in ts], axis=0
                )
                trans_err, rot_err_deg = trans_rot_err(pred_a, pred_b)
                print(
                    f"  [{baseline_mode} vs {image_mode}] {eef_key}: "
                    f"translation RMSE {np.sqrt(np.mean(trans_err**2)):.4f} m, "
                    f"rotation RMSE {np.sqrt(np.mean(rot_err_deg**2)):.2f} deg"
                )
            if gripper_key is not None:
                pred_a = np.array(
                    [predicted_by_mode[baseline_mode][t][gripper_key][0] for t in ts]
                )
                pred_b = np.array(
                    [predicted_by_mode[image_mode][t][gripper_key][0] for t in ts]
                )
                print(
                    f"  [{baseline_mode} vs {image_mode}] {gripper_key}: "
                    f"RMSE {np.sqrt(np.mean((pred_a - pred_b) ** 2)):.2f} %"
                )

    # Plot: tx, ty, tz, rotation angular error (vs. recorded), gripper -- one
    # curve per image_mode plus the recorded ground truth.
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    t_axis = np.array(ts) * skip
    mode_colors = {"real": "tab:orange", "freeze": "tab:green", "black": "tab:purple"}

    if eef_key is not None:
        recorded = action_raw_seq[eef_key][ts]
        for i, label in enumerate(["tx", "ty", "tz"]):
            axes[i].plot(t_axis, recorded[:, i], label="recorded", color="tab:blue")
            for image_mode in image_modes:
                predicted = np.stack(
                    [predicted_by_mode[image_mode][t][eef_key] for t in ts], axis=0
                )
                axes[i].plot(
                    t_axis,
                    predicted[:, i],
                    label=f"predicted ({image_mode})",
                    color=mode_colors.get(image_mode),
                    linestyle="--",
                )
            axes[i].set_ylabel(f"{label} [m]")
            axes[i].legend(loc="upper right", fontsize=7)
            axes[i].grid(True, alpha=0.3)

        for image_mode in image_modes:
            predicted = np.stack(
                [predicted_by_mode[image_mode][t][eef_key] for t in ts], axis=0
            )
            _, rot_err_deg = trans_rot_err(recorded, predicted)
            axes[3].plot(
                t_axis,
                rot_err_deg,
                label=f"vs. recorded ({image_mode})",
                color=mode_colors.get(image_mode),
            )
        axes[3].set_ylabel("rotation error [deg]")
        axes[3].legend(loc="upper right", fontsize=7)
        axes[3].grid(True, alpha=0.3)

    if gripper_key is not None:
        recorded = action_raw_seq[gripper_key][ts][:, 0]
        axes[4].plot(t_axis, recorded, label="recorded", color="tab:blue")
        for image_mode in image_modes:
            predicted = np.array(
                [predicted_by_mode[image_mode][t][gripper_key][0] for t in ts]
            )
            axes[4].plot(
                t_axis,
                predicted,
                label=f"predicted ({image_mode})",
                color=mode_colors.get(image_mode),
                linestyle="--",
            )
        axes[4].set_ylabel("gripper [% closed]")
        axes[4].legend(loc="upper right", fontsize=7)
        axes[4].grid(True, alpha=0.3)

    axes[-1].set_xlabel("raw episode timestep")
    fig.suptitle(os.path.basename(args.rmb_path))
    fig.tight_layout()

    save_plot = args.save_plot
    if save_plot is None:
        root = args.rmb_path.rstrip("/")
        if root.endswith(".rmb") or root.endswith(".hdf5"):
            root = os.path.splitext(root)[0]
        save_plot = f"{root}_policy_check.png"
    fig.savefig(save_plot, dpi=120)
    print(f"\n[CheckDiffusionPolicyPrediction] Saved plot: {save_plot}")


if __name__ == "__main__":
    main()
