from robo_manip_baselines.common.base.PolicyRunnerBase import PolicyRunnerBase


class DiffusionPolicyRunner(PolicyRunnerBase):
    """Diffusion policy for one learned segment -- see PolicyRunnerBase."""

    @classmethod
    def get_rollout_class(cls):
        from .RolloutDiffusionPolicy import RolloutDiffusionPolicy

        return RolloutDiffusionPolicy

    def reset_action_buffers(self, op):
        # Keep state_buf/images_buf (kept current by observe() during the
        # intervention); drop only the chunk and the interpolation ramp, and
        # restart the tick count so the very next tick infers.
        self._sync(op)
        shell = self.shell
        shell.policy_action_buf = None
        shell._interp_from = None
        shell._interp_to = None
        shell.rollout_time_idx = 0
        self._observe_tick = 0

    def observe(self, op):
        # Push the observation into the n_obs_steps history exactly as
        # infer_policy() does, and at the same spacing (once per policy step,
        # i.e. every skip ticks), so the history the policy sees on resuming
        # has the spacing it was trained on.
        self._sync(op)
        shell = self.shell
        if self._observe_tick % shell.args.skip == 0:
            if len(shell.state_keys) > 0:
                shell.update_state_buf()
            if len(shell.camera_names) > 0:
                shell.update_images_buf()
        self._observe_tick += 1
