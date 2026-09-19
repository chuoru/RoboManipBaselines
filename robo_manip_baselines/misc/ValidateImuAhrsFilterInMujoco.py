"""Validates common/utils/ImuAhrsFilter.py's MadgwickAhrsFilter against a
REAL MuJoCo rigid-body physics simulation, as an "IMU self-localization"
building block that can be developed/tested without any Insta360 hardware
or SDK (see envs/real/insta360_bridge/README.md's "Known unresolved risk" --
the SDK application is still pending; this validates the estimator itself
independent of that).

Scene: a single rigid body on a ball joint (so its center stays fixed in
space, isolating pure rotation -- like someone rotating a handheld gripper
in place) is given an initial angular velocity and left to tumble
torque-free under MuJoCo's own dynamics (its box geometry has an asymmetric
inertia tensor, so this is genuine precession/nutation, not just constant-
rate spin). MuJoCo's own `accelerometer`/`gyro` sensors (see this script's
one-time confirmation below of their real convention: "specific force",
i.e. gravity's REACTION shows up when supported/stationary and vanishes in
free-fall -- exactly like a real IMU) provide the sensor stream, with
injected bias/noise to stand in for a real sensor's imperfections.
MuJoCo's own ground-truth `framequat` sensor is compared against both the
AHRS filter's estimate and plain gyro-only integration (dead reckoning) to
show the drift correction this filter provides.

Usage:
    python ./misc/ValidateImuAhrsFilterInMujoco.py --duration 20 --plot
"""

import argparse

import mujoco
import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import MadgwickAhrsFilter

SCENE_XML = """
<mujoco>
  <option gravity="0 0 -9.81" timestep="0.005" integrator="RK4"/>
  <worldbody>
    <body name="imu_body" pos="0 0 1">
      <joint name="ball" type="ball" damping="0.0"/>
      <geom type="box" size="0.06 0.04 0.03" mass="1.0"/>
      <site name="imu_site" pos="0 0 0"/>
    </body>
  </worldbody>
  <sensor>
    <accelerometer name="accel" site="imu_site"/>
    <gyro name="gyro" site="imu_site"/>
    <framequat name="quat" objtype="site" objname="imu_site"/>
  </sensor>
</mujoco>
"""


def quat_to_rotmat(q):
    return pin.Quaternion(q[0], q[1], q[2], q[3]).toRotationMatrix()


def gyro_only_step(q, omega_body, dt):
    """Plain gyro dead-reckoning (no correction), for comparison -- same
    dq/dt = 0.5 * q (x) [0, omega_body] kinematic equation
    MadgwickAhrsFilter's gyro term uses, just without the accelerometer
    correction."""
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


def run(duration_sec, gyro_bias, gyro_noise_std, accel_noise_std, beta, seed):
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    data.qvel[:3] = [1.2, 0.6, -0.4]
    mujoco.mj_forward(model, data)

    ahrs = MadgwickAhrsFilter(beta=beta)
    gyro_only_q = np.array([1.0, 0.0, 0.0, 0.0])
    rng = np.random.default_rng(seed)

    n_steps = int(duration_sec / dt)
    log = {"t": [], "ahrs_err_deg": [], "gyro_only_err_deg": []}

    for step in range(n_steps):
        mujoco.mj_step(model, data)
        true_quat = data.sensor("quat").data.copy()
        gyro_true = data.sensor("gyro").data.copy()
        accel_true = data.sensor("accel").data.copy()

        gyro_meas = gyro_true + gyro_bias + rng.normal(0, gyro_noise_std, 3)
        accel_meas = accel_true + rng.normal(0, accel_noise_std, 3)

        q_ahrs = ahrs.update(gyro_meas, accel_meas, dt)
        gyro_only_q = gyro_only_step(gyro_only_q, gyro_meas, dt)

        r_true = quat_to_rotmat(true_quat)
        err_ahrs = np.rad2deg(
            np.linalg.norm(pin.log3(r_true.T @ quat_to_rotmat(q_ahrs)))
        )
        err_gyro_only = np.rad2deg(
            np.linalg.norm(pin.log3(r_true.T @ quat_to_rotmat(gyro_only_q)))
        )

        log["t"].append(step * dt)
        log["ahrs_err_deg"].append(err_ahrs)
        log["gyro_only_err_deg"].append(err_gyro_only)

    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=20.0, help="[s]")
    parser.add_argument("--gyro_bias", type=float, nargs=3, default=[0.01, -0.008, 0.006])
    parser.add_argument("--gyro_noise_std", type=float, default=0.003, help="[rad/s]")
    parser.add_argument("--accel_noise_std", type=float, default=0.05, help="[m/s^2]")
    parser.add_argument("--beta", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot_output", type=str, default="./imu_ahrs_validation.png")
    args = parser.parse_args()

    log = run(
        args.duration,
        np.array(args.gyro_bias),
        args.gyro_noise_std,
        args.accel_noise_std,
        args.beta,
        args.seed,
    )

    print(
        f"{'t [s]':>8} {'AHRS err [deg]':>16} {'gyro-only err [deg]':>20}"
    )
    for i in range(0, len(log["t"]), max(1, len(log["t"]) // 20)):
        print(
            f"{log['t'][i]:8.1f} {log['ahrs_err_deg'][i]:16.2f} "
            f"{log['gyro_only_err_deg'][i]:20.2f}"
        )

    tail = max(1, len(log["t"]) // 4)
    mean_ahrs = float(np.mean(log["ahrs_err_deg"][-tail:]))
    mean_gyro_only = float(np.mean(log["gyro_only_err_deg"][-tail:]))
    print()
    print(f"Mean error over the final quarter of the run:")
    print(f"  AHRS filter:        {mean_ahrs:.2f} deg")
    print(f"  Gyro-only (raw):    {mean_gyro_only:.2f} deg")
    print(f"  Improvement factor: {mean_gyro_only / max(mean_ahrs, 1e-6):.2f}x")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(log["t"], log["ahrs_err_deg"], label="AHRS filter")
        ax.plot(log["t"], log["gyro_only_err_deg"], label="gyro-only (raw)")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("orientation error [deg]")
        ax.set_title("IMU orientation tracking: AHRS filter vs. raw gyro integration")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot_output)
        print(f"Saved plot to {args.plot_output}")
