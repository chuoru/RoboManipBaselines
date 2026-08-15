from os import path

import mujoco
import numpy as np

from ...real.fairino5.RealFairino5DemoEnv import RealFairino5DemoEnv
from .MujocoFairino5EnvBase import MujocoFairino5EnvBase


class MujocoFairino5CableEnv(MujocoFairino5EnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        MujocoFairino5EnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/fairino/env_fairino_cable.xml",
            ),
            # Start from the SAME joint configuration the real arm starts
            # from (RealFairino5DemoEnv.READY_POSE_DEG), so this env is a
            # usable dry run for the real one.
            #
            # This matters because UMI replay (misc/ReplayUmiOnFairino5.py)
            # and its offline reachability check
            # (misc/CheckUmiFairino5Reachability.py) both retarget a recorded
            # demo as motion RELATIVE to the arm's initial pose. A different
            # starting configuration therefore sweeps a completely different
            # region of the workspace: with this env's previous, MuJoCo-only
            # posture a demo replayed cleanly in sim while the same demo
            # measured against the real ready pose demanded J4=182deg
            # (limit 84deg) and violated joint limits on 14% of frames. "It
            # worked in --sim" has to mean something about the real arm, and
            # it cannot unless both start from the same place.
            #
            # The previous value was a posture tuned for this env's own cable
            # task (gripper ~10cm above the table, reaching horizontally,
            # every joint >=30deg from its limit, no interpenetration with
            # the table/cable/poles). That tuning is deliberately dropped:
            # this env exists as practice/scaffolding for the real FR5 work,
            # so matching the real arm outranks its own task's ergonomics.
            np.concatenate(
                [np.deg2rad(RealFairino5DemoEnv.READY_POSE_DEG), np.zeros(2)]
            ),
            **kwargs,
        )

        self.original_push_block_pos = self.model.body("push_block").pos.copy()
        self.push_block_pos_offsets = np.array(
            [
                [-0.03, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.03, 0.0, 0.0],
                [0.06, 0.0, 0.0],
                [0.09, 0.0, 0.0],
                [0.12, 0.0, 0.0],
            ]
        )  # [m]

        self.goal_region_half_size = 0.06  # [m], matches env_fairino_cable.xml
        self.push_block_half_size = 0.03  # [m], matches env_fairino_cable.xml

    def _get_reward(self):
        # Task: push push_block into the taped-off goal_region square. Reward
        # 1.0 if the block overlaps the goal region's outline at all (not
        # just when the block's center lies inside it) -- i.e. the two
        # axis-aligned squares' bounding boxes intersect, which happens
        # whenever the center-to-center distance is within the sum of their
        # half-sizes on each axis.
        push_block_pos = self.data.body("push_block").xpos.copy()
        goal_region_pos = self.data.body("goal_region").xpos.copy()
        overlap_thre = self.goal_region_half_size + self.push_block_half_size

        if (
            abs(push_block_pos[0] - goal_region_pos[0]) <= overlap_thre
            and abs(push_block_pos[1] - goal_region_pos[1]) <= overlap_thre
        ):
            return 1.0

        return 0.0

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.push_block_pos_offsets)

        push_block_pos = (
            self.original_push_block_pos + self.push_block_pos_offsets[world_idx]
        )
        if self.world_random_scale is not None:
            push_block_pos += np.random.uniform(
                low=-1.0 * self.world_random_scale, high=self.world_random_scale, size=3
            )
        # push_block has a freejoint (dynamic body), so its initial pose must
        # be set via init_qpos (like MujocoXarm7PushtEnv's tblock), not via
        # model.body().pos -- that only affects the body's static compile-
        # time frame, not the freejoint's qpos0 that env.reset() restores.
        push_block_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "push_block"
        )
        push_block_qpos_addr = self.model.jnt_qposadr[push_block_joint_id]
        self.init_qpos[push_block_qpos_addr : push_block_qpos_addr + 3] = push_block_pos

        return world_idx
