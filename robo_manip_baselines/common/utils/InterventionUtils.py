"""Intervention labels of segment-runner recordings (DataKey.INTERVENTION,
DataKey.INTERVENTION_INPUT_ACTIVE) and, for training on them, which time
steps may be used (design D8)."""

import numpy as np

from ..data.DataKey import DataKey


def get_intervention_seq(rmb_data):
    """Per-step 0/1 intervention flag of an episode; all 0 when the episode
    has no such key (teleop demos, older recordings)."""
    num_steps = len(rmb_data[DataKey.TIME][:])
    if DataKey.INTERVENTION not in rmb_data.keys():
        return np.zeros(num_steps, dtype=np.int8)
    return np.asarray(rmb_data[DataKey.INTERVENTION][:], dtype=np.int8).reshape(
        num_steps
    )


def get_intervention_input_active_seq(rmb_data):
    """Per-step 0/1 "operator is moving the device" flag (falls back to the
    intervention flag itself when not recorded)."""
    num_steps = len(rmb_data[DataKey.TIME][:])
    if DataKey.INTERVENTION_INPUT_ACTIVE not in rmb_data.keys():
        return get_intervention_seq(rmb_data)
    return np.asarray(
        rmb_data[DataKey.INTERVENTION_INPUT_ACTIVE][:], dtype=np.int8
    ).reshape(num_steps)
