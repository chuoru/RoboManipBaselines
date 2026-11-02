"""--rollout_data_filter d8 in the DP-style and ACT datasets: which windows
are drawn from demos and from intervention attempts, and with what weight."""

import importlib.util
import os
import shutil
import tempfile
import unittest

import numpy as np

from robo_manip_baselines.common import DataKey, DataManager

POLICY_DIR = os.path.join(os.path.dirname(__file__), "..", "policy")


def load_class(rel_path, name):
    # Load the dataset module by file: the policy packages' __init__ imports
    # their trainers, which need the third-party policy code.
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(POLICY_DIR, rel_path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


ActDataset = load_class("act/ActDataset.py", "ActDataset")
DiffusionPolicyDataset = load_class(
    "diffusion_policy/DiffusionPolicyDataset.py", "DiffusionPolicyDataset"
)

FILTER = {
    "type": "d8",
    "pre_intervention_drop_sec": 1.0,
    "boundary_drop_ticks": 2,
    "intervention_weight": 2.5,
    "autonomous_weight": 0.0,
}


def write_episode(path, num_steps, take_over=None, result="success"):
    seq = {
        DataKey.TIME: list(0.2 * np.arange(num_steps)),
        DataKey.MEASURED_GRIPPER_JOINT_POS: [np.zeros(1) for _ in range(num_steps)],
    }
    meta = {"segment": "grasp"}
    if take_over is not None:
        intervention = np.zeros(num_steps, dtype=np.int8)
        intervention[take_over:] = 1
        active = np.zeros(num_steps, dtype=np.int8)
        active[take_over + 1 :] = 1
        seq[DataKey.INTERVENTION] = list(intervention)
        seq[DataKey.INTERVENTION_INPUT_ACTIVE] = list(active)
        meta["segment_result"] = result
    DataManager(None).save_data(path, seq, meta, increment_episode_idx=False)


def make_meta(filter_config, **data):
    return {
        "data": {"skip": 1, "rollout_data_filter": filter_config, **data},
        "image": {
            "aug_erasing_scale": 0.0,
            "aug_color_scale": 0.0,
            "aug_affine_scale": 0.0,
            "aug_std": 0.0,
        },
    }


class TestRolloutDataFilter(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.demo = os.path.join(self.tmp_dir, "demo.rmb")
        self.success = os.path.join(self.tmp_dir, "success.rmb")
        self.failure = os.path.join(self.tmp_dir, "failure.rmb")
        self.autonomous = os.path.join(self.tmp_dir, "autonomous.rmb")
        write_episode(self.demo, 20)
        # taken over at 10, moving from 11, until the end (30 steps)
        write_episode(self.success, 30, take_over=10)
        write_episode(self.failure, 30, take_over=10, result="failure")
        write_episode(self.autonomous, 30, take_over=30)  # never taken over
        self.filenames = [self.demo, self.success, self.failure, self.autonomous]

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def dp_meta(self, filter_config):
        return make_meta(filter_config, horizon=4, n_obs_steps=2, n_action_steps=2)

    def test_dp_without_filter_is_unchanged(self):
        dataset = DiffusionPolicyDataset(self.filenames, self.dp_meta(None))
        self.assertIsNone(dataset.sample_weights)
        # every start from -pad_before to len - horizon + pad_after
        expected = sum(n - 4 + 1 + 1 + 1 for n in (20, 30, 30, 30))
        self.assertEqual(len(dataset.chunk_info_list), expected)

    def test_dp_d8_windows_and_weights(self):
        dataset = DiffusionPolicyDataset(self.filenames, self.dp_meta(FILTER))
        by_episode = {}
        for (episode_idx, start), weight in zip(
            dataset.chunk_info_list, dataset.sample_weights
        ):
            by_episode.setdefault(episode_idx, []).append((start, weight))
        # demo: every window, weight 1
        self.assertEqual(len(by_episode[0]), 20 - 4 + 3)
        self.assertTrue(all(w == 1.0 for _, w in by_episode[0]))
        # success: usable from max(11, 10 + 2) = 12 to the end; windows whose
        # 4 (clipped) steps all lie in 12..29
        starts = [s for s, _ in by_episode[1]]
        self.assertEqual(starts, list(range(12, 28)))
        self.assertTrue(all(w == 2.5 for _, w in by_episode[1]))
        # failed / never taken over: nothing
        self.assertNotIn(2, by_episode)
        self.assertNotIn(3, by_episode)

    def test_act_d8_valid_starts(self):
        meta = make_meta(FILTER, chunk_size=5)
        dataset = ActDataset(self.filenames, meta)
        self.assertEqual(dataset.filenames, [self.demo, self.success])
        np.testing.assert_array_equal(dataset.valid_start_idxes[0], np.arange(20))
        np.testing.assert_array_equal(dataset.valid_start_idxes[1], np.arange(12, 30))
        self.assertEqual(dataset.sample_weights, [1.0, 2.5])

    def test_act_without_filter_is_unchanged(self):
        dataset = ActDataset(self.filenames, make_meta(None, chunk_size=5))
        self.assertIsNone(dataset.valid_start_idxes)
        self.assertEqual(dataset.filenames, self.filenames)


if __name__ == "__main__":
    unittest.main()
