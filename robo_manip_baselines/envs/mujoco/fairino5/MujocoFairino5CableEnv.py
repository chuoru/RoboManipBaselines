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

        self.original_pole_pos = self.model.body("poles").pos.copy()
        self.pole_pos_offsets = np.array(
            [
                [-0.03, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.03, 0.0, 0.0],
                [0.06, 0.0, 0.0],
                [0.09, 0.0, 0.0],
                [0.12, 0.0, 0.0],
            ]
        )  # [m]

        self.cable_body_ids = None

    def _get_reward(self):
        # Get grid position list of cable
        if self.cable_body_ids is None:
            self.cable_body_ids = []
            for body_id in range(self.model.nbody):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                if name is not None and name.startswith("cable_B"):
                    self.cable_body_ids.append(body_id)
        cable_grid_pos_list = np.array(
            [self.data.xpos[body_id] for body_id in self.cable_body_ids]
        )

        # Get position of poles
        pole1_pos = self.data.geom("pole1").xpos.copy()
        pole2_pos = self.data.geom("pole2").xpos.copy()

        # Check cable height
        z_thre = pole1_pos[2] + 0.01  # [m]
        if cable_grid_pos_list[:, 2].max() > z_thre:
            return 0.0

        # Check cable end
        cable_end_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cable_end"
        )
        cable_end_pos = self.data.xpos[cable_end_body_id].copy()
        x_thre = pole2_pos[0]
        y_thre = pole1_pos[1] - 0.05
        if cable_end_pos[0] < x_thre or cable_end_pos[1] > y_thre:
            return 0.0

        # Check if the cable passes through the poles
        cable_grid_xy_list = cable_grid_pos_list[:, :2]
        pole1_xy = pole1_pos[:2]
        pole2_xy = pole2_pos[:2]
        pole_dir = pole2_xy - pole1_xy

        def check_ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        for i in range(len(cable_grid_xy_list) - 1):
            cable_grid1_pos = cable_grid_xy_list[i]
            cable_grid2_pos = cable_grid_xy_list[i + 1]
            if (
                check_ccw(cable_grid1_pos, pole1_xy, pole2_xy)
                != check_ccw(cable_grid2_pos, pole1_xy, pole2_xy)
            ) and (
                check_ccw(cable_grid1_pos, cable_grid2_pos, pole1_xy)
                != check_ccw(cable_grid1_pos, cable_grid2_pos, pole2_xy)
            ):
                cable_dir = cable_grid2_pos - cable_grid1_pos
                cable_pole_cross = (
                    pole_dir[0] * cable_dir[1] - pole_dir[1] * cable_dir[0]
                )
                if cable_pole_cross > 0:
                    return 1.0

        return 0.0

    def modify_world(self, world_idx=None, cumulative_idx=None):
        if world_idx is None:
            world_idx = cumulative_idx % len(self.pole_pos_offsets)

        pole_pos = self.original_pole_pos + self.pole_pos_offsets[world_idx]
        if self.world_random_scale is not None:
            pole_pos += np.random.uniform(
                low=-1.0 * self.world_random_scale, high=self.world_random_scale, size=3
            )
        self.model.body("poles").pos = pole_pos

        return world_idx
