import unittest

import mujoco
import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import MadgwickAhrsFilter
from robo_manip_baselines.misc.ValidateImuAhrsFilterInMujoco import (
    SCENE_XML,
    gyro_only_step,
    quat_to_rotmat,
)

# Convention used throughout this test (verified against
# pin.Quaternion(...).toRotationMatrix()'s actual behavior, see
# test_static_convergence): R_wb is body-to-world (v_world = R_wb @ v_body),
# matching Pinocchio's own convention. Angular velocity in the BODY frame is
# what MadgwickAhrsFilter.update()'s gyro argument expects (matches the
# standard dq/dt = 0.5 * q (x) [0, omega_body] kinematic equation the filter
# implements).


def quat_to_rotmat(q):
    return pin.Quaternion(q[0], q[1], q[2], q[3]).toRotationMatrix()


class TestImuAhrsFilter(unittest.TestCase):
    def test_static_convergence(self):
        """With zero angular velocity and a constant accelerometer reading,
        the filter should converge its predicted gravity direction to match
        the measurement -- this isolates and validates just the
        accelerometer-correction term's sign/direction (independent of any
        assumption about how a moving ground-truth trajectory is
        generated)."""
        target_rotvec = np.array([0.4, -0.6, 0.9])
        R_target = pin.exp3(target_rotvec)
        gravity_world = np.array([0.0, 0.0, 1.0])
        accel_target = R_target.T @ gravity_world

        ahrs = MadgwickAhrsFilter(beta=0.2, initial_quat=[1.0, 0.0, 0.0, 0.0])
        for _ in range(3000):
            q = ahrs.update(gyro=np.zeros(3), accel=accel_target, dt=1.0 / 200.0)

        R_est = quat_to_rotmat(q)
        predicted_accel = R_est.T @ gravity_world
        self.assertLess(np.linalg.norm(predicted_accel - accel_target), 0.01)

    def test_noise_free_dynamic_tracking(self):
        """With exact (noise-free) gyro+accel matching a rotating ground
        truth, the filter should track that ground truth closely and stay
        bounded (not just "not diverge over 10s", which a stationary filter
        would also satisfy) -- checked at the end of a full rotation
        trajectory."""
        dt = 1.0 / 200.0
        true_omega_world = np.array([0.5, -0.3, 0.2])
        gravity_world = np.array([0.0, 0.0, 1.0])
        ahrs = MadgwickAhrsFilter(beta=0.05)
        r_wb = np.eye(3)

        for _ in range(2000):  # 10 s
            omega_body = r_wb.T @ true_omega_world
            r_wb = r_wb @ pin.exp3(omega_body * dt)
            accel_body = r_wb.T @ gravity_world
            q = ahrs.update(omega_body, accel_body, dt)

        r_est = quat_to_rotmat(q)
        err_deg = np.rad2deg(np.linalg.norm(pin.log3(r_wb.T @ r_est)))
        self.assertLess(err_deg, 1.0)

    def test_beats_raw_gyro_integration_under_bias(self):
        """With a constant gyro bias + noise, AHRS-filtered orientation
        error must grow markedly slower than plain gyro integration's --
        the entire reason to use this filter over dead-reckoning. Both
        estimators see the exact same noisy measurements each step; only
        the update rule differs."""
        dt = 1.0 / 200.0
        true_omega_world = np.array([0.5, -0.3, 0.2])
        gravity_world = np.array([0.0, 0.0, 1.0])
        gyro_bias = np.array([0.02, -0.015, 0.01])
        rng = np.random.default_rng(3)

        ahrs = MadgwickAhrsFilter(beta=0.05)
        r_wb = np.eye(3)
        q_gyro_only = np.array([1.0, 0.0, 0.0, 0.0])

        def gyro_only_step(q, omega_body, dt):
            qdot = 0.5 * np.array(
                [
                    -q[1] * omega_body[0] - q[2] * omega_body[1] - q[3] * omega_body[2],
                    q[0] * omega_body[0] + q[2] * omega_body[2] - q[3] * omega_body[1],
                    q[0] * omega_body[1] - q[1] * omega_body[2] + q[3] * omega_body[0],
                    q[0] * omega_body[2] + q[1] * omega_body[1] - q[2] * omega_body[0],
                ]
            )
            q_new = q + qdot * dt
            return q_new / np.linalg.norm(q_new)

        for _ in range(6000):  # 30 s
            omega_body_true = r_wb.T @ true_omega_world
            r_wb = r_wb @ pin.exp3(omega_body_true * dt)
            accel_body_true = r_wb.T @ gravity_world

            gyro_meas = omega_body_true + gyro_bias + rng.normal(0, 0.002, 3)
            accel_meas = accel_body_true + rng.normal(0, 0.01, 3)

            q_ahrs = ahrs.update(gyro_meas, accel_meas, dt)
            q_gyro_only = gyro_only_step(q_gyro_only, gyro_meas, dt)

        r_ahrs = quat_to_rotmat(q_ahrs)
        r_gyro_only = quat_to_rotmat(q_gyro_only)
        err_ahrs = np.rad2deg(np.linalg.norm(pin.log3(r_wb.T @ r_ahrs)))
        err_gyro_only = np.rad2deg(np.linalg.norm(pin.log3(r_wb.T @ r_gyro_only)))

        self.assertLess(err_ahrs, 0.6 * err_gyro_only)

    def test_mujoco_tumbling_body(self):
        """Same comparison as the two tests above, but driven by a REAL
        MuJoCo rigid-body physics simulation (see
        misc/ValidateImuAhrsFilterInMujoco.py, which this reuses) instead of
        a hand-derived kinematic trajectory -- a torque-free tumbling box on
        a ball joint, so its asymmetric inertia produces genuine
        precession/nutation, not just constant-rate spin. Short duration
        (5s) here for test speed; that script's __main__ runs the same
        scene for longer with plotting for a more thorough by-hand check.
        """
        model = mujoco.MjModel.from_xml_string(SCENE_XML)
        data = mujoco.MjData(model)
        dt = model.opt.timestep
        data.qvel[:3] = [1.2, 0.6, -0.4]
        mujoco.mj_forward(model, data)

        ahrs = MadgwickAhrsFilter(beta=0.08)
        gyro_only_q = np.array([1.0, 0.0, 0.0, 0.0])
        rng = np.random.default_rng(42)
        gyro_bias = np.array([0.01, -0.008, 0.006])

        n_steps = int(5.0 / dt)
        for step in range(n_steps):
            mujoco.mj_step(model, data)
            true_quat = data.sensor("quat").data.copy()
            gyro_meas = (
                data.sensor("gyro").data
                + gyro_bias
                + rng.normal(0, 0.003, 3)
            )
            accel_meas = data.sensor("accel").data + rng.normal(0, 0.05, 3)

            q_ahrs = ahrs.update(gyro_meas, accel_meas, dt)
            gyro_only_q = gyro_only_step(gyro_only_q, gyro_meas, dt)

        r_true = quat_to_rotmat(true_quat)
        err_ahrs = np.rad2deg(
            np.linalg.norm(pin.log3(r_true.T @ quat_to_rotmat(q_ahrs)))
        )
        err_gyro_only = np.rad2deg(
            np.linalg.norm(pin.log3(r_true.T @ quat_to_rotmat(gyro_only_q)))
        )
        self.assertLess(err_ahrs, err_gyro_only)


if __name__ == "__main__":
    unittest.main()
