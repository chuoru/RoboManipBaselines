"""Record a UMI demo (RealUMIDemoEnv, same as bin/Teleop.py RealUMIDemo) while
simultaneously driving a MuJoCo FR5 (MujocoFairino5CableEnv) in real time with
the SAME retargeting math as misc/ReplayUmiOnFairino5.py's --vive_config path,
so the recorded demo and the mirrored arm motion can be visually compared.

WHY: misc/ReplayUmiOnFairino5.py's checkpoint+PI replay was found to diverge
from the recorded demo (position error grew monotonically, e.g. 2mm -> 15mm ->
34mm -> 56mm... across consecutive checkpoints starting from a segment with a
fast/large wrist rotation), on both MuJoCo (--sim) and real hardware. This
script answers the open question that raised: is the retargeted trajectory
itself unreachable (a genuine kinematics/singularity problem), or is
ReplayUmiOnFairino5.py's checkpoint/PI control loop itself buggy?

It does this by driving the MuJoCo arm with a SIMPLE per-frame IK step
(ArmManager.inverse_kinematics(), a single damped-least-squares Newton step --
the same mechanism live teleop uses every frame, see ArmManager.
set_command_eef_pose()) continuously in lockstep with the UMI recording, at
the UMI loop's own (real-time) rate. There is no "checkpoint", no
PI-controller, no closed-loop re-anchoring to a "measured" pose read back from
the env, and no --max_iters_per_checkpoint budget to run out of -- just one
small IK correction per frame, exactly like normal teleoperation.

If the MuJoCo arm tracks the demo smoothly here, that PROVES the recorded
trajectory is kinematically reachable (no singularity/joint-limit problem),
and isolates the earlier divergence to ReplayUmiOnFairino5.py's
checkpoint+PI+adaptive_ik_step control loop specifically. If it does NOT
track smoothly here either, the problem is in the retargeting math or the
recorded data itself.

UPDATE: running this mirror against real recordings surfaced exactly the
latter, twice. Both rotation and translation are now retargeted as pure
EEF/TCP-LOCAL reproductions -- matching how ViveInputDevice.set_command_data
itself drives a robot in live single-robot teleop -- which needs no
vive_world_to_base_frame_rotation (room-vs-base calibration) at all. See
_mirror_step()'s comment and misc/ReplayUmiOnFairino5.py's docstring
rotation/translation notes for the full derivation and the wrong
intermediate versions this went through (rotation composed as a room-frame
quantity scrambled axes; translation retargeted as one batch delta from t=0
broke once the demo also rotated, since the commanded direction needs to
track the TCP's evolving orientation mid-demo, not a fixed room-frame one).

Usage:
    python ./misc/TeleopUmiWithMujocoMirror.py \\
        --config ./envs/configs/RealUMIDemo.yaml \\
        --input_device vive --input_device_config ./teleop/configs/ViveUMI.yaml

Saves, alongside the normal RealUMIDemo .rmb dataset (same as bin/Teleop.py):
  - The MuJoCo mirror's rendered frames as an .mp4 video next to the .rmb
    file (<demo>_world<W>_<E>_mujoco_mirror.mp4), for after-the-fact visual
    review (e.g. by Claude) of what the mirrored arm did during recording.

pos_scale is read from the SAME --input_device_config YAML passed for the
Vive tracker (e.g. teleop/configs/ViveUMI.yaml) -- the identical value
misc/ReplayUmiOnFairino5.py's --vive_config would load for the same file, so
this mirror and a later replay are directly comparable.
"""

import argparse

import gymnasium as gym
import numpy as np
import pinocchio as pin
import videoio
import yaml

from robo_manip_baselines.common import ArmManager
from robo_manip_baselines.envs.operation.OperationRealUMIDemo import (
    OperationRealUMIDemo,
)
from robo_manip_baselines.teleop import TeleopBase

MIRROR_ENV_ID = "robo_manip_baselines/MujocoFairino5CableEnv-v0"


class TeleopUmiWithMujocoMirror(OperationRealUMIDemo, TeleopBase):
    def set_additional_args(self, parser):
        parser.add_argument(
            "--no_mirror",
            action="store_true",
            help="disable the MuJoCo mirror (record UMI data only, same as "
            "plain bin/Teleop.py RealUMIDemo)",
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.mirror_enabled = not self.args.no_mirror
        if self.mirror_enabled:
            self._setup_mirror()

    def _setup_mirror(self):
        # render_mode="human" opens a live MuJoCo viewer window (WindowViewer,
        # auto-rendered every env.step() -- see MujocoEnvBase.step()) so the
        # arm's motion can be watched live while operating the UMI rig.
        # Video frames for saving are captured separately via the per-camera
        # OffScreenViewer objects (env.cameras[name]["viewer"], see
        # MujocoEnvBase._get_info()) in _mirror_step() below -- NOT via
        # env.render(), whose generic rgb_array/depth_array path is broken
        # here (MujocoEnvBase nulls out mujoco_renderer.width/height, which
        # OffScreenViewer's constructor requires as real ints; WindowViewer
        # for "human" mode does not have this problem).
        self.mirror_env = gym.make(MIRROR_ENV_ID, render_mode="human")
        self.mirror_env.reset()
        self.mirror_arm_manager = ArmManager(
            self.mirror_env.unwrapped, self.mirror_env.unwrapped.body_config_list[0]
        )
        self.mirror_init_se3 = self.mirror_arm_manager.current_se3.copy()
        camera_names = self.mirror_env.unwrapped.camera_names
        if len(camera_names) == 0:
            raise RuntimeError(
                f"[{self.__class__.__name__}] {MIRROR_ENV_ID} has no cameras "
                "to record a mirror video from."
            )
        self.mirror_camera_name = (
            "front" if "front" in camera_names else camera_names[0]
        )

        # Same source of truth misc/ReplayUmiOnFairino5.py's --vive_config
        # reads -- keeps this mirror and a later replay directly comparable.
        # Only pos_scale is used: retargeting is a pure EEF/TCP-local
        # reproduction for both rotation and translation (see _mirror_step's
        # comment), which needs no vive_world_to_base_frame_rotation
        # (room-vs-base) calibration at all.
        with open(self.args.input_device_config, "r") as f:
            vive_config = yaml.safe_load(f)
        self.mirror_pos_scale = vive_config.get("pos_scale", 1.0)
        print(
            f"[{self.__class__.__name__}] MuJoCo mirror enabled "
            f"(pos_scale={self.mirror_pos_scale})"
        )

        self._reset_mirror_episode()

    def _reset_mirror_episode(self):
        self.mirror_umi_se3_0 = None
        self.mirror_umi_prev_se3 = None
        self.mirror_fr5_translation = None
        self.mirror_video_frames = []
        if self.mirror_enabled:
            self.mirror_arm_manager.reset()

    def reset(self):
        super().reset()
        if self.mirror_enabled:
            self._reset_mirror_episode()

    def record_data(self):
        super().record_data()
        if self.mirror_enabled:
            self._mirror_step()

    def _mirror_step(self):
        umi_body_manager = self.motion_manager.body_manager_list[0]
        umi_se3_t = umi_body_manager.target_se3.copy()
        if self.mirror_umi_se3_0 is None:
            self.mirror_umi_se3_0 = umi_se3_t.copy()
            self.mirror_umi_prev_se3 = umi_se3_t.copy()
            self.mirror_fr5_translation = self.mirror_init_se3.translation.copy()

        # Rotation: umi_se3_t.rotation (with mirror_umi_se3_0.rotation ~= I)
        # is an EEF/TCP-LOCAL delta -- ViveInputDevice.set_command_data
        # computes it as eef_se3_at_enable.rotation @
        # (vive_to_eef_frame_rotation @ delta_vive_rotation @
        # vive_to_eef_frame_rotation.T), i.e. composed via RIGHT-multiply as
        # a local-frame delta relative to enable-time orientation (see that
        # method's own comment). Reapplying it onto FR5's own init
        # orientation uses the same right-multiply composition.
        delta_umi_rotation = self.mirror_umi_se3_0.rotation.T @ umi_se3_t.rotation
        target_rotation = self.mirror_init_se3.rotation @ delta_umi_rotation

        # Translation: umi_se3_t.translation is an ACCUMULATION of per-frame
        # EEF-local increments, each re-projected into the (evolving) target
        # frame by THAT frame's own target_rotation before being added on --
        # see ViveInputDevice.set_command_data's translation comment ("push
        # the tracker forward always means push the TCP forward along its
        # own CURRENT Z, even while simultaneously rotating"). So this
        # frame's local increment is recovered by un-rotating the raw
        # increment with THIS frame's own umi_se3_t.rotation, then
        # re-accumulated through FR5's own (already correctly retargeted)
        # target_rotation -- reproducing "push forward along the TCP's own
        # current, possibly-tilted, Z" on FR5 exactly as it happened on the
        # UMI rig. No vive_world_to_base_frame_rotation (room-vs-base) is
        # involved: this is a pure local-frame reproduction, same as
        # rotation.
        raw_translation_delta = umi_se3_t.translation - self.mirror_umi_prev_se3.translation
        translation_delta_eef_local = umi_se3_t.rotation.T @ raw_translation_delta
        self.mirror_fr5_translation = self.mirror_fr5_translation + self.mirror_pos_scale * (
            target_rotation @ translation_delta_eef_local
        )
        self.mirror_umi_prev_se3 = umi_se3_t.copy()
        target_translation = self.mirror_fr5_translation.copy()

        # Single damped-least-squares IK step toward the new target -- the
        # same per-frame mechanism normal live teleop uses (ArmManager.
        # set_command_eef_pose()), NOT the checkpoint+PI loop from
        # ReplayUmiOnFairino5.py.
        self.mirror_arm_manager.set_command_eef_pose(
            pin.SE3(target_rotation, target_translation)
        )
        self.mirror_arm_manager.set_command_gripper_joint_pos(
            umi_body_manager.gripper_joint_pos
        )

        mirror_action = np.concatenate(
            [
                self.mirror_arm_manager.arm_joint_pos,
                self.mirror_arm_manager.gripper_joint_pos,
            ]
        )
        self.mirror_env.step(mirror_action)

        camera = self.mirror_env.unwrapped.cameras[self.mirror_camera_name]
        camera["viewer"].make_context_current()
        frame = camera["viewer"].render(render_mode="rgb_array", camera_id=camera["id"])
        self.mirror_video_frames.append(frame)

    def save_data(self):
        import os

        # Recompute the exact same path TeleopBase.save_data() will use, so
        # the mirror video sits right next to the .rmb it corresponds to.
        filename = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "dataset",
                f"{self.demo_name}_{self.datetime_now:%Y%m%d_%H%M%S}",
                f"{self.demo_name}_world{self.data_manager.world_idx:0>1}_{self.data_manager.episode_idx:0>3}.{self.args.file_format}",
            )
        )

        super().save_data()

        if self.mirror_enabled and len(self.mirror_video_frames) > 0:
            video_filename = os.path.splitext(filename)[0] + "_mujoco_mirror.mp4"
            # Write at the rate this episode was ACTUALLY recorded at, not
            # the MuJoCo env's nominal render_fps: one mirror frame is
            # produced per teleop loop iteration, and that loop runs at
            # whatever rate the UMI rig allows (~8.4 Hz measured, dominated
            # by Vive tracker reads) -- far slower than env.dt's nominal
            # ~31 fps. Tagging the file 31 fps made it play back ~3.7x too
            # fast, which reads as "the operator moved much faster than they
            # did" and makes it useless as a reference to compare a replay
            # against. DataManager.calc_fps is the same computation the .rmb
            # camera videos use, so both videos share one timebase.
            fps = self.data_manager.calc_fps(self.data_manager.all_data_seq)
            videoio.videosave(
                video_filename, np.array(self.mirror_video_frames), fps=fps
            )
            print(
                f"[{self.__class__.__name__}] Save the MuJoCo mirror video as "
                f"{video_filename} ({len(self.mirror_video_frames)} frames"
                f"{'' if fps is None else f', {fps:.2f} fps'})"
            )

    def run(self):
        try:
            super().run()
        finally:
            if self.mirror_enabled:
                try:
                    self.mirror_env.close()
                except AttributeError:
                    # gymnasium's MujocoEnv.close() unconditionally closes
                    # its generic (human/rgb_array) viewer even though this
                    # script only ever renders via the per-camera
                    # OffScreenViewer objects (see _mirror_step()) -- that
                    # generic viewer is never created here, so close() hits
                    # a None. Harmless; only the cleanup call itself fails.
                    pass


def parse_argument():
    import sys

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False
    )
    parser.add_argument("--config", type=str, help="configuration file")
    args, remaining_argv = parser.parse_known_args()
    # Strip --config from sys.argv before TeleopBase's own setup_args() parses
    # it -- same reason bin/Teleop.py's TeleopMain does this: --config is
    # this meta-parser's own argument, not one TeleopBase.setup_args() knows.
    sys.argv = [sys.argv[0]] + remaining_argv
    return args


def main():
    args = parse_argument()
    if args.config is None:
        config = {}
    else:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)

    teleop = TeleopUmiWithMujocoMirror(**config)
    teleop.run()


if __name__ == "__main__":
    main()
