import numpy as np
import torch

from robo_manip_baselines.common import (
    DataKey,
    DatasetBase,
    RmbData,
    convert_data_to_policy,
    get_relative_eef_pose9,
    get_skipped_data_seq,
    get_skipped_single_data,
)


class ActDataset(DatasetBase):
    """Dataset to train ACT policy."""

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, episode_idx):
        skip = self.model_meta_info["data"]["skip"]
        chunk_size = self.model_meta_info["data"]["chunk_size"]

        with RmbData(self.filenames[episode_idx], self.enable_rmb_cache) as rmb_data:
            episode_len = rmb_data[DataKey.TIME][::skip].shape[0]
            start_time_idx = np.random.choice(episode_len)

            # Load state
            if len(self.model_meta_info["state"]["keys"]) == 0:
                state = np.zeros(0, dtype=np.float64)
            else:
                state = np.concatenate(
                    [
                        convert_data_to_policy(
                            get_skipped_single_data(
                                rmb_data[key], start_time_idx * skip, key, skip
                            ),
                            key,
                        )
                        for key in self.model_meta_info["state"]["keys"]
                    ]
                )

            # Load action
            relative_eef_action = self.model_meta_info["data"].get(
                "relative_eef_action", False
            )
            if relative_eef_action:
                # Pose at the observation (inference) time.
                ref_pose = get_skipped_single_data(
                    rmb_data[DataKey.MEASURED_EEF_POSE],
                    start_time_idx * skip,
                    DataKey.MEASURED_EEF_POSE,
                    skip,
                )
            action = np.concatenate(
                [
                    get_relative_eef_pose9(
                        get_skipped_data_seq(
                            rmb_data[key][start_time_idx * skip :], key, skip
                        )[:chunk_size],
                        ref_pose,
                    )
                    if relative_eef_action and key == DataKey.COMMAND_EEF_POSE
                    else convert_data_to_policy(
                        get_skipped_data_seq(
                            rmb_data[key][start_time_idx * skip :],
                            key,
                            skip,
                        )[:chunk_size],
                        key,
                    )
                    for key in self.model_meta_info["action"]["keys"]
                ],
                axis=1,
            )

            # Load images
            image_keys = [
                DataKey.get_rgb_image_key(camera_name)
                for camera_name in self.model_meta_info["image"]["camera_names"]
            ]
            # Read just the one frame needed. The cached variant used to index
            # [::skip] first, which decodes and caches the WHOLE episode video
            # per entry -- at 1280x800 that exhausts RAM within a few episodes.
            images = np.stack(
                [rmb_data[key][start_time_idx * skip] for key in image_keys],
                axis=0,
            )

        # Chunk action
        action_len = action.shape[0]
        action_chunked = np.zeros((chunk_size, action.shape[1]), dtype=np.float64)
        action_chunked[:action_len] = action[:chunk_size]
        is_pad = np.zeros(chunk_size, dtype=bool)
        is_pad[action_len:] = True

        # Pre-convert data
        state, action_chunked, images = self.pre_convert_data(
            state, action_chunked, images
        )

        # Convert to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = torch.tensor(action_chunked, dtype=torch.float32)
        images_tensor = torch.tensor(images, dtype=torch.uint8)
        is_pad_tensor = torch.tensor(is_pad, dtype=torch.bool)

        # Augment data
        state_tensor, action_tensor, images_tensor = self.augment_data(
            state_tensor, action_tensor, images_tensor
        )

        # Sort in the order of policy inputs and outputs
        return state_tensor, images_tensor, action_tensor, is_pad_tensor
