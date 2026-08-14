from os import path

import mujoco
import numpy as np

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
            # Arm joints tuned (numerically, against real MuJoCo collision geometry
            # -- see misc/ for the search script) so the gripper's "pinch" site:
            #  - sits ~10cm above the table top (table top z=0.815 in env_fairino_
            #    cable.xml -> pinch z=0.915)
            #  - is oriented parallel to the table (reaching horizontally, not
            #    pointing down), facing straight forward along the base's +X axis
            #    (no yaw). NOTE: the "pinch" site's reach direction is its local
            #    *Y* axis, not Z -- confirmed both by the site's body-fixed offset
            #    (pos="0 0.0895 0.0023" in fairino5_v6_body.xml, overwhelmingly
            #    along Y) and by rendering -- an earlier attempt that leveled the
            #    local Z axis instead visibly pointed the gripper straight down.
            #  - joint1 (shoulder yaw) is ~90deg away from an earlier candidate
            #    that reached the same region with joint3 (elbow) only ~3deg from
            #    its limit; this posture instead keeps every joint at least
            #    ~30deg from its limit (joint4 the tightest, ~6deg), leaving
            #    headroom to actually extend the arm further across the table
            #    during teleop instead of starting near-maxed-out.
            #  - has zero interpenetration with the table, itself, the cable, and
            #    the poles at rest (previously the class default of contype=0/
            #    conaffinity=0 on all robot meshes hid this entirely -- with
            #    collision enabled, the old pose -- which, despite its "pointing
            #    straight down" description, actually already reached
            #    horizontally over the table -- had the forearm buried up to
            #    22.7cm into the table)
            # This sits ~12cm away (in the table plane) from the table's center.
            np.array(
                [
                    7.631944889126715e-05,
                    -2.4288940584970162,
                    -2.163997949351957,
                    1.3792951116167333,
                    1.570720007328029,
                    3.238504968409533e-10,
                    *np.zeros(2),
                ]
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
