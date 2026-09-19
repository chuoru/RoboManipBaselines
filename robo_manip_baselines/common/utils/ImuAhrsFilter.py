"""IMU-only orientation estimation (no magnetometer), for fusing a gyroscope
with an accelerometer so that ROTATION tracking does not drift the way plain
gyro integration does (see MadgwickAhrsFilter's docstring for why). This is
the "self-localization from IMU data" building block that is testable and
useful independent of the Insta360 CameraSDK/ORB-SLAM3 bridge (see
envs/real/insta360_bridge/) -- it only needs gyro+accel samples, real or
Mujoco-synthesized (see tests/TestImuAhrsFilter.py for a from-scratch Mujoco
validation).

IMPORTANT SCOPE NOTE: this estimates ORIENTATION only. As established
earlier (see envs/real/insta360_bridge/README.md's "Known unresolved risk"
and the M5Stack/Insta360 planning discussion), IMU data alone cannot give a
useful POSITION estimate -- double-integrating accelerometer noise/bias
diverges within seconds. Position still needs a visual (or other absolute)
reference; see VisualRelativePoseEstimator.py, which combines this filter's
orientation with camera-based translation tracking.
"""

import numpy as np


def _quat_multiply(q1, q2):
    """Hamilton product of two (w, x, y, z) quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


class MadgwickAhrsFilter:
    """Madgwick's IMU-only AHRS filter (S. Madgwick, "An efficient
    orientation filter for inertial and inertial/magnetic sensor arrays",
    2010) -- the widely-used standard algorithm for fusing a gyroscope with
    an accelerometer into a drift-corrected orientation estimate.

    Why this beats plain gyro integration: integrating angular velocity
    alone (q_{t+1} = q_t + 0.5 * q_t (x) [0, gyro] * dt) accumulates gyro
    bias/noise without bound -- a constant 0.5 deg/s bias alone reaches 30
    degrees of error in one minute, with no way to tell "the sensor rotated"
    from "the bias drifted". Madgwick's filter adds a correction step each
    update that nudges the estimate toward whatever orientation would make
    the ROTATED reference gravity vector match the measured accelerometer
    reading -- since gravity's direction in the world frame is known and
    fixed, this bounds roll/pitch error to the accelerometer's own noise
    floor instead of the gyro's integrated bias. (Yaw about the vertical
    axis is NOT observable from gravity alone and will still drift slowly --
    a magnetometer would be needed to correct that too; not implemented
    here, as the Insta360 IMU's magnetometer availability is unconfirmed,
    same category of unknown as its accelerometer -- see
    envs/real/insta360_bridge/README.md.)

    Convention: quaternion (w, x, y, z) represents the rotation from the
    EARTH/world frame to the SENSOR/body frame (i.e. it rotates a
    world-frame vector into the sensor frame) -- e.g. the reference gravity
    direction [0, 0, 1] (world +Z up) rotates into what the accelerometer
    should read when stationary. accel is expected normalized (unit
    vector); gyro in rad/s.
    """

    def __init__(self, beta=0.1, initial_quat=None):
        """beta: filter gain -- higher trusts the accelerometer correction
        more (faster convergence, noisier steady-state); lower trusts the
        gyro integration more (smoother, slower to correct drift). 0.1 is
        Madgwick's own commonly-cited starting point for a MEMS IMU."""
        self.beta = beta
        self.quat = (
            np.array([1.0, 0.0, 0.0, 0.0])
            if initial_quat is None
            else np.array(initial_quat, dtype=np.float64)
        )
        self.quat /= np.linalg.norm(self.quat)

    def update(self, gyro, accel, dt):
        """Advance the filter by one sample. gyro: (3,) [rad/s]. accel: (3,)
        [any consistent unit -- normalized internally; pass all-zero to skip
        the accelerometer correction for this step, e.g. during a known
        high-vibration/high-acceleration interval where gravity is not the
        dominant signal]. Returns the updated (w, x, y, z) quaternion."""
        q1, q2, q3, q4 = self.quat
        gx, gy, gz = gyro

        # Gyro-only rate of change: qDot = 0.5 * q (x) [0, gyro].
        qdot_gyro = 0.5 * _quat_multiply(self.quat, np.array([0.0, gx, gy, gz]))

        accel_norm = np.linalg.norm(accel)
        if accel_norm > 1e-8:
            ax, ay, az = np.asarray(accel, dtype=np.float64) / accel_norm

            # Gradient of the error between the rotated reference gravity
            # direction [0,0,1] and the measured (normalized) accelerometer
            # reading -- closed-form Jacobian from Madgwick's paper (eq.
            # 21-25), avoiding a general-purpose autodiff/optimizer for
            # what has to run every IMU sample in real time.
            f = np.array(
                [
                    2.0 * (q2 * q4 - q1 * q3) - ax,
                    2.0 * (q1 * q2 + q3 * q4) - ay,
                    2.0 * (0.5 - q2 * q2 - q3 * q3) - az,
                ]
            )
            j = np.array(
                [
                    [-2.0 * q3, 2.0 * q4, -2.0 * q1, 2.0 * q2],
                    [2.0 * q2, 2.0 * q1, 2.0 * q4, 2.0 * q3],
                    [0.0, -4.0 * q2, -4.0 * q3, 0.0],
                ]
            )
            gradient = j.T @ f
            gradient_norm = np.linalg.norm(gradient)
            if gradient_norm > 1e-8:
                gradient /= gradient_norm
                qdot_gyro = qdot_gyro - self.beta * gradient

        self.quat = self.quat + qdot_gyro * dt
        self.quat /= np.linalg.norm(self.quat)
        return self.quat.copy()
