import importlib.util
import os
import shutil
import tempfile
import unittest

import numpy as np

from robo_manip_baselines.common import DataKey, DataManager, RmbData


def load_resample_module():
    path = os.path.join(os.path.dirname(__file__), "..", "misc", "ResampleRmbData.py")
    spec = importlib.util.spec_from_file_location("ResampleRmbData", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resample = load_resample_module()


class TestResampleIndices(unittest.TestCase):
    def test_nearest_on_uniform_grid(self):
        time_seq = np.array([0.0, 0.07, 0.15, 0.21, 0.42, 0.5, 0.61])
        indices, grid = resample.get_resample_indices(time_seq, 5.0)
        np.testing.assert_allclose(grid, [0.0, 0.2, 0.4, 0.6])
        np.testing.assert_array_equal(indices, [0, 3, 4, 6])

    def test_slower_source_repeats_frames(self):
        time_seq = np.array([0.0, 0.5, 1.0])
        indices, grid = resample.get_resample_indices(time_seq, 5.0)
        self.assertEqual(len(grid), 6)
        np.testing.assert_array_equal(indices, [0, 0, 1, 1, 2, 2])

    def test_tie_goes_to_earlier(self):
        # 0.5 is exactly halfway between 0.25 and 0.75 (exact in binary).
        indices, _ = resample.get_resample_indices(np.array([0.0, 0.25, 0.75]), 2.0)
        np.testing.assert_array_equal(indices, [0, 1])

    def test_empty(self):
        indices, grid = resample.get_resample_indices(np.zeros(0), 5.0)
        self.assertEqual(len(indices), 0)
        self.assertEqual(len(grid), 0)


class TestResampleEpisode(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_round_trip(self):
        rng = np.random.default_rng(0)
        num_steps = 23
        time_seq = np.cumsum(rng.uniform(0.05, 0.15, num_steps)) - 0.05
        all_data_seq = {
            DataKey.TIME: list(time_seq),
            DataKey.MEASURED_GRIPPER_JOINT_POS: [
                np.array([100.0 * (i > 10)]) for i in range(num_steps)
            ],
            "intervention": [np.int8(i % 2) for i in range(num_steps)],
            DataKey.get_rgb_image_key("hand"): [
                np.full((24, 32, 3), 10 * i, dtype=np.uint8) for i in range(num_steps)
            ],
        }
        in_path = os.path.join(self.tmp_dir, "in", "ep.rmb")
        DataManager(None).save_data(in_path, all_data_seq, {"segment": "grasp"})

        out_path = os.path.join(self.tmp_dir, "out", "ep.rmb")
        num_in, num_out, _ = resample.resample_episode(in_path, out_path, 5.0)
        self.assertEqual(num_in, num_steps)
        indices, grid = resample.get_resample_indices(time_seq, 5.0)

        with RmbData(out_path) as rmb_data:
            np.testing.assert_allclose(rmb_data[DataKey.TIME][:], grid)
            np.testing.assert_array_equal(
                rmb_data[resample.RESAMPLE_SOURCE_IDX_KEY][:], indices
            )
            np.testing.assert_array_equal(
                rmb_data[DataKey.MEASURED_GRIPPER_JOINT_POS][:, 0],
                100.0 * (indices > 10),
            )
            np.testing.assert_array_equal(rmb_data["intervention"][:], indices % 2)
            images = rmb_data[DataKey.get_rgb_image_key("hand")][:]
            self.assertEqual(len(images), num_out)
            # Lossy video: compare the mean brightness of each frame.
            np.testing.assert_allclose(
                images.mean(axis=(1, 2, 3)), 10.0 * indices, atol=3.0
            )
            self.assertEqual(rmb_data.attrs["segment"], "grasp")
            self.assertEqual(float(rmb_data.attrs["resampled_hz"]), 5.0)


if __name__ == "__main__":
    unittest.main()
