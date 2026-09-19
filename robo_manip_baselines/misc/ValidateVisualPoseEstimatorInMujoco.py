"""Validates common/utils/VisualRelativePoseEstimator.py against REAL MuJoCo
rendering, as the translation half of an "IMU + camera self-localization"
building block developed without any Insta360 hardware/SDK (see
envs/real/insta360_bridge/README.md's "Known unresolved risk"). Mirrors
misc/ValidateImuAhrsFilterInMujoco.py's role for the orientation half.

Renders a textured MuJoCo scene (checkerboard floor + several distinct
colored primitives, enough visual texture for optical flow to have
something to track) from a scripted, KINEMATICALLY-driven camera trajectory
(qpos set directly each frame, not physics-driven -- this script only cares
about vision, not dynamics), and compares the estimator's per-frame relative
pose against MuJoCo's own ground-truth camera pose (cam_xpos/cam_xmat).

Runs two trajectories to make the estimator's real, measured limitation
visible (see VisualRelativePoseEstimator.py's docstring for the full
explanation): a translation-dominant "handheld-like" path, where
translation direction tracks well, and a rotation-heavy circular path,
where it does not (rotation recovery itself stays accurate in both).

Usage:
    python ./misc/ValidateVisualPoseEstimatorInMujoco.py --plot
"""

import argparse

import mujoco
import numpy as np
import pinocchio as pin

from robo_manip_baselines.common import VisualRelativePoseEstimator

SCENE_XML = """
<mujoco>
  <asset>
    <texture type="2d" name="grid" builtin="checker" rgb1=".2 .3 .4" rgb2=".9 .9 .9" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="12 12" reflectance="0"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" diffuse="1 1 1"/>
    <geom type="plane" size="5 5 0.1" material="grid"/>
    <geom type="box" pos="0.6 0.3 0.2" size="0.1 0.1 0.2" rgba="0.8 0.2 0.2 1"/>
    <geom type="box" pos="-0.4 0.7 0.15" size="0.15 0.08 0.15" rgba="0.2 0.8 0.2 1"/>
    <geom type="sphere" pos="0.3 -0.6 0.1" size="0.1" rgba="0.2 0.2 0.8 1"/>
    <geom type="box" pos="-0.7 -0.4 0.25" size="0.1 0.1 0.25" rgba="0.8 0.8 0.2 1"/>
    <geom type="box" pos="0.0 0.9 0.1" size="0.3 0.05 0.1" rgba="0.8 0.5 0.1 1"/>
    <geom type="box" pos="0.9 -0.2 0.15" size="0.05 0.3 0.15" rgba="0.5 0.1 0.8 1"/>
    <body name="cam_body" pos="0 0 1.0">
      <inertial pos="0 0 0" mass="0.01" diaginertia="1e-5 1e-5 1e-5"/>
      <freejoint/>
      <camera name="eye" fovy="60"/>
    </body>
  </worldbody>
</mujoco>
"""


def render_camera_trajectory(positions, rpys, width=640, height=480):
    """positions/rpys: same-length lists of (3,) world position / (roll,
    pitch,yaw). Returns (frames, gt_positions, gt_rotmats) with MuJoCo's own
    ground truth (cam_xpos/cam_xmat) for each, driven kinematically (qpos
    set directly, mj_forward only -- no physics stepping)."""
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "eye")

    frames, gt_positions, gt_rotmats = [], [], []
    for pos, rpy in zip(positions, rpys):
        rotmat = pin.utils.rpyToMatrix(np.array(rpy))
        quat_xyzw = pin.Quaternion(rotmat).coeffs()
        data.qpos[0:3] = pos
        data.qpos[3:7] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="eye")
        frames.append(renderer.render().copy())
        gt_positions.append(data.cam_xpos[cam_id].copy())
        gt_rotmats.append(data.cam_xmat[cam_id].reshape(3, 3).copy())

    fovy_deg = model.cam_fovy[cam_id]
    fy = height / (2 * np.tan(np.deg2rad(fovy_deg) / 2))
    camera_matrix = np.array(
        [[fy, 0, width / 2.0], [0, fy, height / 2.0], [0, 0, 1]]
    )
    return frames, gt_positions, gt_rotmats, camera_matrix


def evaluate(frames, gt_positions, gt_rotmats, camera_matrix):
    estimator = VisualRelativePoseEstimator(
        camera_matrix, max_corners=400, quality_level=0.005, min_distance=6.0,
        min_tracked_points=15,
    )
    rot_errs_deg, cos_sims = [], []
    for i, frame in enumerate(frames):
        result = estimator.update(frame)
        if i == 0 or result is None:
            continue
        r1, r2 = gt_rotmats[i - 1], gt_rotmats[i]
        p1, p2 = gt_positions[i - 1], gt_positions[i]
        rotation_rel_true = r2.T @ r1
        translation_rel_true = r2.T @ (p1 - p2)
        if np.linalg.norm(translation_rel_true) < 1e-6:
            continue
        rot_errs_deg.append(
            np.rad2deg(
                np.linalg.norm(pin.log3(result["rotation"].T @ rotation_rel_true))
            )
        )
        cos_sims.append(
            np.dot(
                result["translation_direction"],
                translation_rel_true / np.linalg.norm(translation_rel_true),
            )
        )
    return np.array(rot_errs_deg), np.array(cos_sims)


def handheld_like_trajectory(n_steps):
    positions, rpys = [], []
    for i in range(n_steps):
        t = i / n_steps
        positions.append(
            [
                0.3 * np.sin(2 * np.pi * t * 2),
                0.2 * np.sin(2 * np.pi * t * 3) + 0.1,
                1.0 + 0.1 * np.sin(2 * np.pi * t),
            ]
        )
        rpys.append(
            [
                0.1 * np.sin(2 * np.pi * t * 1.5),
                0.5 + 0.1 * np.cos(2 * np.pi * t),
                np.pi / 2 + 0.15 * np.sin(2 * np.pi * t * 0.7),
            ]
        )
    return positions, rpys


def rotation_heavy_circular_trajectory(n_steps, radius=0.3):
    positions, rpys = [], []
    for i in range(n_steps):
        theta = 2 * np.pi * i / n_steps
        positions.append([radius * np.cos(theta), radius * np.sin(theta), 1.0])
        rpys.append([0.0, 0.3, theta + np.pi])
    return positions, rpys


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--plot_output", type=str, default="./visual_pose_estimator_validation.png"
    )
    args = parser.parse_args()

    scenarios = {
        "handheld-like (translation-dominant)": handheld_like_trajectory(
            args.n_steps
        ),
        "circular (rotation-heavy)": rotation_heavy_circular_trajectory(
            args.n_steps
        ),
    }

    all_results = {}
    for label, (positions, rpys) in scenarios.items():
        frames, gt_positions, gt_rotmats, camera_matrix = render_camera_trajectory(
            positions, rpys
        )
        rot_errs_deg, cos_sims = evaluate(
            frames, gt_positions, gt_rotmats, camera_matrix
        )
        all_results[label] = (rot_errs_deg, cos_sims)
        print(f"\n=== {label} ===")
        print(f"  n_valid: {len(rot_errs_deg)} / {len(frames) - 1}")
        print(
            f"  rotation error [deg]:      mean={rot_errs_deg.mean():.2f}  "
            f"median={np.median(rot_errs_deg):.2f}"
        )
        print(
            f"  translation cos-similarity: mean={cos_sims.mean():.3f}  "
            f"median={np.median(cos_sims):.3f}"
        )

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for label, (rot_errs_deg, cos_sims) in all_results.items():
            axes[0].plot(rot_errs_deg, label=label)
            axes[1].plot(cos_sims, label=label)
        axes[0].set_title("rotation error [deg]")
        axes[0].set_xlabel("frame")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)
        axes[1].set_title("translation direction cosine similarity")
        axes[1].set_xlabel("frame")
        axes[1].axhline(0.0, color="gray", linewidth=0.8)
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot_output)
        print(f"\nSaved plot to {args.plot_output}")
