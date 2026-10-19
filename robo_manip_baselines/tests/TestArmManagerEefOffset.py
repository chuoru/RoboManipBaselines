import os
import unittest

import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import ArmConfig
from robo_manip_baselines.common.body.ArmManager import ArmManager
from robo_manip_baselines.common.utils.MathUtils import get_rel_pose_from_se3, get_se3_from_rel_pose

URDF = os.path.join(
    os.path.dirname(__file__), "..", "envs", "assets", "common", "robots", "fairino5_v6", "fairino5_v6.urdf"
)
INIT_DEG = [-65.0, -93.0, -142.0, 56.0, 64.0, -1.0]


def make_manager(offset_pose):
    config = ArmConfig(
        arm_urdf_path=URDF,
        arm_root_pose=None,
        ik_eef_joint_id=6,
        arm_joint_idxes=np.arange(6),
        gripper_joint_idxes=np.array([6]),
        gripper_joint_idxes_in_gripper_joint_pos=np.array([0]),
        eef_idx=0,
        init_arm_joint_pos=np.radians(INIT_DEG),
        init_gripper_joint_pos=np.zeros(1),
        ik_eef_offset_pose=offset_pose,
    )
    return ArmManager(None, config)


class TestArmManagerEefOffset(unittest.TestCase):
    OFFSET = np.array([0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0])  # FR5 joint-6 origin -> flange face

    def test_current_se3_is_offset_point(self):
        manager = make_manager(self.OFFSET)
        joint_se3 = manager.pin_data.oMi[6]
        np.testing.assert_allclose(
            manager.current_se3.translation,
            joint_se3.translation + joint_se3.rotation @ np.array([0, 0, 0.1]),
            atol=1e-12,
        )
        np.testing.assert_allclose(manager.current_se3.rotation, joint_se3.rotation, atol=1e-12)

    def test_relative_rotation_is_about_offset_point(self):
        # A pure rotation of the controlled point must leave its position fixed
        # (it rotates about the flange face, not about the joint-6 origin).
        manager = make_manager(self.OFFSET)
        start = manager.current_se3.copy()
        manager.target_se3 = start * get_se3_from_rel_pose(np.r_[0, 0, 0, np.radians(15), 0, 0])
        for _ in range(50):
            manager.inverse_kinematics()
        np.testing.assert_allclose(manager.current_se3.translation, start.translation, atol=1e-5)
        rel = get_rel_pose_from_se3(start.actInv(manager.current_se3))
        np.testing.assert_allclose(rel[3:], [np.radians(15), 0, 0], atol=1e-5)

    def test_without_offset_unchanged(self):
        manager = make_manager(None)
        np.testing.assert_allclose(
            manager.current_se3.homogeneous, manager.pin_data.oMi[6].homogeneous, atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
