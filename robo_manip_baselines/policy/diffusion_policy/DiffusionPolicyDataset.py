import numpy as np
import torch

from robo_manip_baselines.common import (
    DataKey,
    DatasetBase,
    DpStyleDatasetMixin,
    RmbData,
    convert_data_to_policy,
    get_relative_eef_pose9,
    get_skipped_data_seq,
)


class DiffusionPolicyDataset(DatasetBase, DpStyleDatasetMixin):
    """Dataset to train diffusion policy."""

    def setup_variables(self):
        self.setup_dp_style_chunk()

    def __len__(self):
        return len(self.chunk_info_list)

    def __getitem__(self, chunk_idx):
        skip = self.model_meta_info["data"]["skip"]
        horizon = self.model_meta_info["data"]["horizon"]
        image_size = self.model_meta_info["data"]["image_size"]
        episode_idx, start_time_idx = self.chunk_info_list[chunk_idx]

        with RmbData(
            self.filenames[episode_idx], self.enable_rmb_cache, image_size=image_size
        ) as rmb_data:
            episode_len = rmb_data[DataKey.TIME][::skip].shape[0]
            time_idxes = np.clip(
                np.arange(start_time_idx, start_time_idx + horizon), 0, episode_len - 1
            )

            # Load state
            if len(self.model_meta_info["state"]["keys"]) == 0:
                state = np.zeros(0, dtype=np.float64)
            else:
                state = np.concatenate(
                    [
                        convert_data_to_policy(
                            get_skipped_data_seq(rmb_data[key][:], key, skip)[
                                time_idxes
                            ],
                            key,
                        )
                        for key in self.model_meta_info["state"]["keys"]
                    ],
                    axis=1,
                )

            # Load action
            relative_eef_action = self.model_meta_info["data"].get(
                "relative_eef_action", False
            )
            if relative_eef_action:
                # Pose at the observation (inference) time = the last
                # observation step of this window.
                obs_time_idx = time_idxes[self.model_meta_info["data"]["n_obs_steps"] - 1]
                ref_pose = get_skipped_data_seq(
                    rmb_data[DataKey.MEASURED_EEF_POSE][:], DataKey.MEASURED_EEF_POSE, skip
                )[obs_time_idx]
            action = np.concatenate(
                [
                    get_relative_eef_pose9(
                        get_skipped_data_seq(rmb_data[key][:], key, skip)[time_idxes],
                        ref_pose,
                    )
                    if relative_eef_action and key == DataKey.COMMAND_EEF_POSE
                    else convert_data_to_policy(
                        get_skipped_data_seq(rmb_data[key][:], key, skip)[time_idxes],
                        key,
                    )
                    for key in self.model_meta_info["action"]["keys"]
                ],
                axis=1,
            )

            # Load images
            if len(self.model_meta_info["image"]["camera_names"]) == 0:
                images = None
            else:
                # Decode only the window this sample needs. Indexing the
                # video with [::skip] first decoded (and, with
                # enable_rmb_cache, cached) the WHOLE episode per sample, which
                # at 1280x800 OOM-killed a 4-worker run on a 15 GB machine.
                # time_idxes is a clipped arange, so it is non-decreasing.
                window_start = int(time_idxes[0])
                window_end = int(time_idxes[-1])
                images = np.stack(
                    [
                        rmb_data[DataKey.get_rgb_image_key(camera_name)][
                            window_start * skip : window_end * skip + 1 : skip
                        ][time_idxes - window_start]
                        for camera_name in self.model_meta_info["image"]["camera_names"]
                    ],
                    axis=0,
                )

        if images is not None:
            # Check image size
            current_h, current_w = images.shape[2:4]
            target_w, target_h = self.model_meta_info["data"]["image_size"]
            if (current_w, current_h) != (target_w, target_h):
                raise ValueError(
                    f"[{self.__class__.__name__}] Image size mismatch: ({current_w}, {current_h}) != ({target_w}, {target_h})"
                )

        # Pre-convert data
        state, action, images = self.pre_convert_data(state, action, images)

        # Convert to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.float32)
        if images is None:
            images_tensor = None
        else:
            images_tensor = torch.tensor(images, dtype=torch.uint8)

        # Augment data
        state_tensor, action_tensor, images_tensor = self.augment_data(
            state_tensor, action_tensor, images_tensor
        )

        # Convert to data structure of policy input and output
        if len(self.model_meta_info["image"]["camera_names"]) == 0:
            data = {"obs": state_tensor, "action": action_tensor}
        else:
            data = {"obs": {}, "action": action_tensor}
            if len(self.model_meta_info["state"]["keys"]) > 0:
                data["obs"]["state"] = state_tensor
            for camera_idx, camera_name in enumerate(
                self.model_meta_info["image"]["camera_names"]
            ):
                data["obs"][DataKey.get_rgb_image_key(camera_name)] = images_tensor[
                    camera_idx
                ]

        return data

    def augment_data(self, state, action, images):
        state, action, images = super().augment_data(state, action, images)

        if images is not None:
            # Adjust to a range from -1 to 1 to match the original implementation
            images = images * 2.0 - 1.0

        return state, action, images
