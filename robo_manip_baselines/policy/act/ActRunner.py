from robo_manip_baselines.common.base.PolicyRunnerBase import PolicyRunnerBase


class ActRunner(PolicyRunnerBase):
    """ACT for one learned segment -- see PolicyRunnerBase.

    ACT conditions on the current observation only, so observe() has nothing
    to buffer. Its temporal-ensembling history (policy_action_buf_history)
    holds chunks inferred from earlier states; reset_variables(), called by
    reset_buffers()/reset_action_buffers(), clears it together with the
    current chunk, so after an intervention ACT does not average its new
    chunk with ones from before the human moved the arm."""

    @classmethod
    def get_rollout_class(cls):
        from .RolloutAct import RolloutAct

        return RolloutAct
