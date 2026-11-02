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


def _runs(flags):
    """[(start, end)) index ranges where flags is true."""
    flags = np.asarray(flags, dtype=bool)
    if len(flags) == 0:
        return []
    edges = np.diff(np.concatenate([[0], flags.astype(np.int8), [0]]))
    return list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))


def compute_d8_masks(
    time_seq,
    intervention,
    input_active,
    segment_success,
    pre_intervention_drop_sec=1.0,
    boundary_drop_ticks=2,
    intervention_weight=2.5,
    autonomous_weight=0.0,
):
    """Which steps of a segment-runner episode a policy may learn from, and
    with what weight (design D8). Returns (usable: bool[T], weight: float[T]).

    Along the time axis an attempt with an intervention looks like
        (1) autonomous -> (2) just before the take-over -> (3) taken over, the
        operator not moving yet -> (4) the operator correcting -> (5) autonomous
    and only (4) is learned from, with intervention_weight:
      - (1), (5): the policy's own actions; unused (autonomous_weight > 0
        would use them, from successful attempts only, minus (2))
      - (2): pre_intervention_drop_sec before a take-over: whatever made the
        operator step in -- never used
      - (3): from the take-over until intervention_input_active first
        becomes 1 (the reaction delay: the arm just stands still); dropped,
        and boundary_drop_ticks more steps at both ends of every
        intervention, where control switches hands
      - (4) of an attempt that still failed in the end is dropped as well:
        a correction that did not lead to success is not worth imitating.
    """
    time_seq = np.asarray(time_seq, dtype=np.float64)
    intervention = np.asarray(intervention, dtype=bool)
    input_active = np.asarray(input_active, dtype=bool)
    num_steps = len(time_seq)
    usable = np.zeros(num_steps, dtype=bool)
    weight = np.zeros(num_steps, dtype=np.float64)

    if segment_success:
        for start, end in _runs(intervention):
            active = np.where(input_active[start:end])[0]
            if len(active) == 0:
                continue  # the operator never moved: nothing to learn
            first = start + int(active[0])
            lo = max(first, start + boundary_drop_ticks)
            # The end of the episode is not a hand-over: keep the last steps
            # when the operator finished the segment.
            hi = end if end == num_steps else end - boundary_drop_ticks
            if hi > lo:
                usable[lo:hi] = True
                weight[lo:hi] = intervention_weight

        if autonomous_weight > 0.0:
            autonomous = ~intervention
            for start, _ in _runs(intervention):
                # (2): drop the window before each take-over.
                autonomous &= ~(
                    (time_seq >= time_seq[start] - pre_intervention_drop_sec)
                    & (np.arange(num_steps) < start)
                )
            for start, end in _runs(intervention):
                autonomous[max(start - boundary_drop_ticks, 0) : start] = False
                autonomous[end : end + boundary_drop_ticks] = False
            autonomous &= ~usable
            usable |= autonomous
            weight[autonomous] = autonomous_weight

    return usable, weight


def get_d8_masks(rmb_data, filter_config, skip=1):
    """(usable, weight) of one episode at the training step spacing (every
    skip-th step), or None when the episode is a plain demo (no
    intervention labels -- every step usable with weight 1)."""
    if DataKey.INTERVENTION not in rmb_data.keys():
        return None
    time_seq = rmb_data[DataKey.TIME][:]
    intervention = get_intervention_seq(rmb_data)
    input_active = get_intervention_input_active_seq(rmb_data)
    result = rmb_data.attrs.get("segment_result", "")
    if isinstance(result, bytes):
        result = result.decode()
    usable, weight = compute_d8_masks(
        time_seq,
        intervention,
        input_active,
        segment_success=(result == "success"),
        pre_intervention_drop_sec=filter_config.get("pre_intervention_drop_sec", 1.0),
        boundary_drop_ticks=filter_config.get("boundary_drop_ticks", 2),
        intervention_weight=filter_config.get("intervention_weight", 2.5),
        autonomous_weight=filter_config.get("autonomous_weight", 0.0),
    )
    return usable[::skip], weight[::skip]


def get_window_weight(usable, weight, time_idxes):
    """Weight of a training sample whose action window covers time_idxes:
    0 unless every step of the window is usable (design D8), else the mean
    step weight."""
    if not np.all(usable[time_idxes]):
        return 0.0
    return float(np.mean(weight[time_idxes]))
