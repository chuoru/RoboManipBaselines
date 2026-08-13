"""Convert RoboManipBaselines-recorded episodes (.rmb/.hdf5) into a UMI-style
zarr ReplayBuffer -- the same schema written by
https://github.com/real-stanford/universal_manipulation_interface's
scripts_slam_pipeline/07_generate_replay_buffer.py and read by that repo's
diffusion_policy/dataset/umi_dataset.py (UmiDataset). This lets locally
recorded teleop data be trained directly with UMI's own train.py and its
umi/*.yaml task configs, instead of (or in addition to) this project's own
DiffusionPolicyDataset (which already trains straight from .rmb/.hdf5 files
via RmbData -- see policy/diffusion_policy/DiffusionPolicyDataset.py -- and
needs no conversion at all if that trainer is all you need).

Per-episode keys written to the replay buffer (matching UMI's own naming; one
robot arm = "robot0", a second arm for dual-arm setups would be "robot1", etc.
-- see --robot_prefix):
  robot0_eef_pos             (T, 3)  float32 -- DataKey.MEASURED_EEF_POSE[:, 0:3]
  robot0_eef_rot_axis_angle  (T, 3)  float32 -- rotation vector (pin.log3 of the
                                                 measured quaternion), NOT a
                                                 quaternion -- this is UMI's
                                                 on-disk rotation convention
  robot0_gripper_width       (T, 1)  float32 -- DataKey.MEASURED_GRIPPER_JOINT_POS,
                                                 passed through as-is; NOTE this
                                                 project's gripper units are
                                                 env-specific (e.g. percent-closed
                                                 for the Fairino grippers, meters
                                                 for others) rather than UMI's
                                                 fixed meters-of-opening, so
                                                 mixing this with real UMI
                                                 recordings needs a unit-matched
                                                 rescale first
  robot0_demo_start_pose     (T, 6)  float32 -- first frame's [pos, axis-angle],
                                                 broadcast across the episode
  robot0_demo_end_pose       (T, 6)  float32 -- same, using the last frame
  camera{N}_rgb              (T, H, W, 3) uint8 -- resized to --image_size, in
                                                    the order of the .rmb file's
                                                    own camera_names attribute

UMI's UmiDataset reconstructs "action" and the relative/absolute pose
representations it actually trains on at load time from these same keys (see
that class's __init__ and _sample_to_data), so they are intentionally not
duplicated here.

Usage:
    python ./misc/ConvertRmbDataToUmiZarr.py ./dataset/MujocoFairino5Cable_<date> \\
        --output ./dataset/MujocoFairino5Cable_<date>.umi.zarr
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pinocchio as pin
from tqdm import tqdm

sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../third_party/diffusion_policy")
)
from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: E402

from robo_manip_baselines.common import DataKey, RmbData, find_rmb_files  # noqa: E402


def eef_pose_to_pos_axis_angle(measured_eef_pose):
    """DataKey.MEASURED_EEF_POSE is (T, 7): tx,ty,tz,qw,qx,qy,qz (see
    common/utils/MathUtils.py's get_pose_from_se3). Returns (T,3) position and
    (T,3) rotation vectors (UMI's on-disk rotation representation)."""
    pos = measured_eef_pose[:, 0:3].astype(np.float32)
    axis_angle = np.stack(
        [
            pin.log3(pin.Quaternion(*quat_wxyz).toRotationMatrix())
            for quat_wxyz in measured_eef_pose[:, 3:7]
        ]
    ).astype(np.float32)
    return pos, axis_angle


def resize_images(images, out_res):
    out_w, out_h = out_res
    if images.shape[1:3] == (out_h, out_w):
        return images
    return np.stack(
        [
            cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
            for image in images
        ]
    )


class ConvertRmbDataToUmiZarr:
    def __init__(self, path, output, image_size, compressor, robot_prefix):
        self.rmb_path_list = find_rmb_files(path)
        self.output = output
        self.image_size = tuple(int(x) for x in image_size.split(","))
        self.compressor = compressor
        self.robot_prefix = robot_prefix

    def load_episode_data(self, rmb_data):
        measured_eef_pose = rmb_data[DataKey.MEASURED_EEF_POSE][:]
        eef_pos, eef_rot_axis_angle = eef_pose_to_pos_axis_angle(measured_eef_pose)

        gripper_width = rmb_data[DataKey.MEASURED_GRIPPER_JOINT_POS][:].astype(
            np.float32
        )
        if gripper_width.ndim == 1:
            gripper_width = gripper_width[:, None]

        demo_pose = np.concatenate([eef_pos, eef_rot_axis_angle], axis=1)
        demo_start_pose = np.broadcast_to(demo_pose[0], demo_pose.shape).astype(
            np.float32
        )
        demo_end_pose = np.broadcast_to(demo_pose[-1], demo_pose.shape).astype(
            np.float32
        )

        episode_data = {
            f"{self.robot_prefix}_eef_pos": eef_pos,
            f"{self.robot_prefix}_eef_rot_axis_angle": eef_rot_axis_angle,
            f"{self.robot_prefix}_gripper_width": gripper_width,
            f"{self.robot_prefix}_demo_start_pose": demo_start_pose,
            f"{self.robot_prefix}_demo_end_pose": demo_end_pose,
        }

        for camera_idx, camera_name in enumerate(rmb_data.attrs["camera_names"]):
            images = rmb_data[DataKey.get_rgb_image_key(camera_name)][:]
            episode_data[f"camera{camera_idx}_rgb"] = resize_images(
                images, self.image_size
            )

        return episode_data

    def convert(self):
        import zarr

        replay_buffer = ReplayBuffer.create_empty_zarr(storage=zarr.MemoryStore())

        for rmb_path in tqdm(self.rmb_path_list, desc="episodes"):
            with RmbData(rmb_path) as rmb_data:
                episode_data = self.load_episode_data(rmb_data)
            replay_buffer.add_episode(data=episode_data, compressors=None)

        print(
            f"[{self.__class__.__name__}] Converted {replay_buffer.n_episodes} "
            f"episodes, {replay_buffer.n_steps} total steps -> {self.output}"
        )
        replay_buffer.save_to_path(self.output, compressors=self.compressor)
        print(f"[{self.__class__.__name__}] Wrote zarr dataset to {self.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=str, help="directory or file of .rmb/.hdf5 episodes")
    parser.add_argument(
        "--output", type=str, required=True, help="output .zarr directory path"
    )
    parser.add_argument(
        "--image_size",
        type=str,
        default="224,224",
        help="output image resolution as 'width,height' (UMI's own default)",
    )
    parser.add_argument(
        "--compressor",
        type=str,
        default="disk",
        choices=["default", "disk"],
        help="zarr compressor preset (see ReplayBuffer.resolve_compressor): "
        "'disk' (zstd, better ratio) or 'default' (lz4, faster)",
    )
    parser.add_argument(
        "--robot_prefix",
        type=str,
        default="robot0",
        help="key prefix for the arm (e.g. 'robot0', 'robot1' for a second arm "
        "in a dual-arm recording converted separately)",
    )
    args = parser.parse_args()

    converter = ConvertRmbDataToUmiZarr(
        path=args.path,
        output=args.output,
        image_size=args.image_size,
        compressor=args.compressor,
        robot_prefix=args.robot_prefix,
    )
    converter.convert()
