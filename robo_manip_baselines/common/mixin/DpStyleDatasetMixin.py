import numpy as np

from robo_manip_baselines.common import DataKey, RmbData
from robo_manip_baselines.common.utils.InterventionUtils import (
    get_d8_masks,
    get_window_weight,
)


class DpStyleDatasetMixin:
    "This mixin is intended to be used in classes that inherit from DatasetBase."

    def setup_dp_style_chunk(self):
        self.chunk_info_list = []
        # Sampling weight per chunk (only filled with a rollout data filter,
        # see TrainBase --rollout_data_filter).
        self.sample_weights = None
        skip = self.model_meta_info["data"]["skip"]
        horizon = self.model_meta_info["data"]["horizon"]
        # Set pad_before and pad_after to values one less than n_obs_steps and n_action_steps, respectively
        # Ref: https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/config/task/pusht_image.yaml#L36-L37
        pad_before = self.model_meta_info["data"]["n_obs_steps"] - 1
        pad_after = self.model_meta_info["data"]["n_action_steps"] - 1
        filter_config = self.model_meta_info["data"].get("rollout_data_filter")
        if filter_config is not None:
            self.sample_weights = []

        for episode_idx, filename in enumerate(self.filenames):
            with RmbData(filename) as rmb_data:
                episode_len = rmb_data[DataKey.TIME][::skip].shape[0]
                masks = None
                if filter_config is not None:
                    masks = get_d8_masks(rmb_data, filter_config, skip)
                for start_time_idx in range(
                    -1 * pad_before, episode_len - (horizon - 1) + pad_after
                ):
                    if filter_config is not None:
                        if masks is None:
                            weight = 1.0  # a plain demo: every window
                        else:
                            # Same window __getitem__ reads; every step of it
                            # must be usable (D8).
                            time_idxes = np.clip(
                                np.arange(start_time_idx, start_time_idx + horizon),
                                0,
                                episode_len - 1,
                            )
                            weight = get_window_weight(*masks, time_idxes)
                            if weight <= 0.0:
                                continue
                        self.sample_weights.append(weight)
                    self.chunk_info_list.append((episode_idx, start_time_idx))

        if filter_config is not None:
            weights = np.asarray(self.sample_weights)
            print(
                f"[{self.__class__.__name__}] Rollout data filter "
                f"'{filter_config['type']}': {len(weights)} samples from "
                f"{len(self.filenames)} episodes, {int(np.sum(weights != 1.0))} of them "
                "from interventions (weights "
                f"{sorted(set(np.round(weights, 3).tolist()))})"
            )
