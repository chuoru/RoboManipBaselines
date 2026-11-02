"""Scripted building blocks of a segmented task (see
envs/configs/segments/*.yaml and common/base/SegmentRunnerBase.py).

A segmented task alternates SCRIPTED parts (fixed moves between named poses,
retreats, gripper open/close) with short LEARNED parts that absorb the
variation in the scene. The scripted parts are phases built from small YAML
specs:

    {joint_pos: home}            blocking move to a named pose
    {joint_pos: home, vel: 20}   ... with the MoveJ speed percentage
    {rel_xyz: [0, 0, 0.08]}      move the end effector by [dx, dy, dz] [m]
                                 in the robot base frame, orientation kept
    {gripper: open}              open / close the gripper and wait

These phases work in any op that has env, motion_manager and obs (the
segment runner, and TeleopBase for --pre_segments).

Every move here happens outside env.step() (or replaces the normal command
for its duration), so each phase ends by re-anchoring the motion manager on
the MEASURED state (hold_at_measured): IK warm start, IK target and hence the
next command all start from where the arm really is. Skipping that is what
produced a 13.8 deg lurch after the MoveJ to the reset pose (see
OperationRealFairino5Demo.MoveToInitPhase).
"""

import os

import numpy as np
import pinocchio as pin
import yaml

from ..body.ArmManager import ArmManager
from ..data.DataKey import DataKey
from .PhaseBase import PhaseBase

# Base directory of the package (robo_manip_baselines/), against which the
# relative paths in segment/pose YAML files are resolved if they do not
# exist relative to the current directory.
PACKAGE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_path(path, base_dir=None):
    """Resolve a path from a YAML file: absolute as is; otherwise relative to
    the current directory if it exists there, else to base_dir (e.g. the
    YAML file's own directory), else to the package directory."""
    if path is None:
        return None
    path = os.path.expanduser(str(path))
    if os.path.isabs(path) or os.path.exists(path):
        return path
    for root in (base_dir, PACKAGE_DIR):
        if root is None:
            continue
        candidate = os.path.normpath(os.path.join(root, path))
        if os.path.exists(candidate):
            return candidate
    return path


def get_arm_managers(op):
    return [
        body_manager
        for body_manager in op.motion_manager.body_manager_list
        if isinstance(body_manager, ArmManager)
    ]


def hold_at_measured(op, refresh_obs=True):
    """Make the current command equal to the measured state of the arm.

    refresh_obs: read a fresh observation first (needed after a blocking
    move, whose result op.obs does not contain yet). The gripper command is
    left as is."""
    if refresh_obs:
        op.obs = op.env.unwrapped._get_obs()
    op.motion_manager.sync_arm_to_measured(op.obs)
    for arm_manager in get_arm_managers(op):
        # sync_arm_to_measured only moves the IK warm start; the IK TARGET
        # (what teleop devices integrate on and what the next IK step chases)
        # must move too, or the arm is pulled back to the old target.
        arm_manager.target_se3 = arm_manager.current_se3.copy()


def get_measured_gripper(op):
    return float(
        op.motion_manager.get_measured_data(DataKey.MEASURED_GRIPPER_JOINT_POS, op.obs)[
            0
        ]
    )


def get_gripper_close_threshold(op):
    """Percent-closed at/above which the gripper counts as closed: the
    tool_do threshold when the env has one (that is what decides what the
    hardware does), else the middle of the range."""
    return float(getattr(op.env.unwrapped, "tool_do_close_threshold", 50.0))


def solve_ik(arm_manager, target_se3, init_arm_joint_pos, max_iters=200, tol=1e-6):
    """Full IK solve for target_se3 by repeating ArmManager's damped Newton
    step (which normally does one step per control tick). The manager's own
    state is restored afterwards. Returns (arm_joint_pos, converged)."""
    saved_arm_joint_pos = arm_manager.arm_joint_pos.copy()
    saved_target_se3 = arm_manager.target_se3.copy()
    try:
        arm_manager.arm_joint_pos = np.array(init_arm_joint_pos, dtype=np.float64)
        arm_manager.forward_kinematics()
        arm_manager.target_se3 = target_se3
        converged = False
        for _ in range(max_iters):
            arm_manager.inverse_kinematics()
            error = pin.log(arm_manager.current_se3.actInv(target_se3)).vector
            if np.dot(error, error) < tol**2:
                converged = True
                break
        result = arm_manager.arm_joint_pos.copy()
    finally:
        arm_manager.arm_joint_pos = saved_arm_joint_pos
        arm_manager.forward_kinematics()
        arm_manager.target_se3 = saved_target_se3
    return result, converged


def load_named_poses(path):
    """{name: arm joint positions [rad] or None} from a pose YAML (see
    envs/configs/poses/). A pose may also be the string "init" (the env's
    reset pose). Unrecorded poses are left as None and rejected when used."""
    with open(path, "r") as f:
        config = yaml.safe_load(f) or {}
    poses = config.get("poses", config)
    unit = config.get("unit", "rad") if "poses" in config else "rad"
    named_poses = {}
    for name, value in poses.items():
        if value is None or isinstance(value, str):
            named_poses[name] = value
        else:
            value = np.array(value, dtype=np.float64)
            named_poses[name] = np.deg2rad(value) if unit == "deg" else value
    return named_poses


def get_named_pose(op, named_poses, name):
    if name not in named_poses:
        raise KeyError(
            f"[{op.__class__.__name__}] Unknown named pose '{name}'. Known: "
            f"{sorted(named_poses)}"
        )
    value = named_poses[name]
    if isinstance(value, str) and value == "init":
        init_qpos = op.env.unwrapped.init_qpos
        arm_joint_idxes = op.env.unwrapped.body_config_list[0].arm_joint_idxes
        return np.array(init_qpos[arm_joint_idxes], dtype=np.float64)
    if value is None or isinstance(value, str):
        raise ValueError(
            f"[{op.__class__.__name__}] Named pose '{name}' is not recorded yet. "
            "Record it with misc/RecordNamedPose.py before running."
        )
    return value


class ScriptedPhaseBase(PhaseBase):
    """A scripted step. label: what to show on the screen."""

    def __init__(self, op, label):
        super().__init__(op)
        self.label = label

    @property
    def display_name(self):
        return self.label

    def handle_actions(self, actions):
        """Operator actions are ignored while a scripted step runs."""
        pass


class JointMovePhase(ScriptedPhaseBase):
    """Move to a named pose. Blocking MoveJ on envs that provide
    move_to_joint_pos() (the real FR5); otherwise the command is streamed
    along a joint-space ramp (simulation)."""

    STREAM_JOINT_VEL = np.deg2rad(20.0)  # [rad/s] for the streaming fallback

    def __init__(self, op, label, pose_name, named_poses, vel=10.0):
        super().__init__(op, label)
        self.pose_name = pose_name
        self.named_poses = named_poses
        self.vel = vel

    def start(self):
        super().start()
        self.target = get_named_pose(self.op, self.named_poses, self.pose_name)
        env = self.op.env.unwrapped
        self.done = False
        if hasattr(env, "move_to_joint_pos"):
            print(f"[{self.op.__class__.__name__}] Move to '{self.pose_name}'.")
            env.move_to_joint_pos(self.target, vel=self.vel)
            hold_at_measured(self.op)
            self.done = True
        else:
            arm_joint_idxes = env.body_config_list[0].arm_joint_idxes
            self.from_pos = np.array(
                env.get_joint_pos_from_obs(self.op.obs)[arm_joint_idxes]
            )
            self.duration = max(
                np.max(np.abs(self.target - self.from_pos)) / self.STREAM_JOINT_VEL,
                0.5,
            )

    def pre_update(self):
        if self.done:
            return
        alpha = min(self.get_elapsed_duration() / self.duration, 1.0)
        # Smoothstep: zero velocity at both ends.
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        arm_joint_pos = (1.0 - alpha) * self.from_pos + alpha * self.target
        for arm_manager in get_arm_managers(self.op):
            arm_manager.set_command_joint_pos(
                arm_joint_pos, arm_manager.gripper_joint_pos
            )

    def check_transition(self):
        if self.done:
            return True
        if self.get_elapsed_duration() >= self.duration + 0.5:
            hold_at_measured(self.op, refresh_obs=False)
            return True
        return False


class RelativeMovePhase(ScriptedPhaseBase):
    """Move the end effector by dxyz [m] (robot base frame) from where it is
    now, orientation unchanged.

    mode "movej" (default on the real FR5): solve IK for the goal and make one
    blocking MoveJ there. For the short vertical retreats this is used for,
    the joint-space path stays within millimetres of the straight line.
    mode "stream": stream a straight-line Cartesian ramp through the normal
    ServoJ command path (and every safety limiter of env.step()). Used
    automatically when the env has no move_to_joint_pos()."""

    # Largest joint change a relative move may need. A few-centimetre retreat
    # needs a few degrees; much more means IK found a different branch.
    MAX_JOINT_DELTA = np.deg2rad(45.0)

    def __init__(self, op, label, dxyz, vel=10.0, speed=0.05, mode="movej"):
        super().__init__(op, label)
        self.dxyz = np.array(dxyz, dtype=np.float64)
        self.vel = vel
        self.speed = speed  # [m/s] for mode "stream"
        if mode not in ("movej", "stream"):
            raise ValueError(f"[{self.__class__.__name__}] Invalid mode: {mode}")
        self.mode = mode

    def start(self):
        super().start()
        env = self.op.env.unwrapped
        hold_at_measured(self.op)
        arm_manager = get_arm_managers(self.op)[0]
        self.from_se3 = arm_manager.current_se3.copy()
        self.goal_se3 = self.from_se3.copy()
        self.goal_se3.translation = self.from_se3.translation + self.dxyz
        self.done = False

        mode = self.mode if hasattr(env, "move_to_joint_pos") else "stream"
        if mode == "movej":
            measured = arm_manager.arm_joint_pos.copy()
            target, converged = solve_ik(arm_manager, self.goal_se3, measured)
            if not converged:
                raise RuntimeError(
                    f"[{self.op.__class__.__name__}] IK did not converge for the "
                    f"relative move {self.dxyz.tolist()} -- target out of reach?"
                )
            if np.max(np.abs(target - measured)) > self.MAX_JOINT_DELTA:
                raise RuntimeError(
                    f"[{self.op.__class__.__name__}] Relative move "
                    f"{self.dxyz.tolist()} would need a "
                    f"{np.rad2deg(np.max(np.abs(target - measured))):.0f} deg joint "
                    "change; refusing (IK branch flip?)."
                )
            print(
                f"[{self.op.__class__.__name__}] Relative move "
                f"{np.round(self.dxyz, 3).tolist()} m."
            )
            env.move_to_joint_pos(target, vel=self.vel)
            hold_at_measured(self.op)
            self.done = True
        else:
            self.duration = max(np.linalg.norm(self.dxyz) / self.speed, 0.2)

    def pre_update(self):
        if self.done:
            return
        alpha = min(self.get_elapsed_duration() / self.duration, 1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        target_se3 = self.from_se3.copy()
        target_se3.translation = self.from_se3.translation + alpha * self.dxyz
        for arm_manager in get_arm_managers(self.op):
            arm_manager.set_command_eef_pose(target_se3)

    # Stream mode: how close [m] the measured end effector must get to the
    # goal before the move counts as done, and how long [s] past the planned
    # duration to wait for that before moving on regardless.
    STREAM_GOAL_TOLERANCE = 0.005
    STREAM_EXTRA_WAIT = 3.0

    def check_transition(self):
        if self.done:
            return True
        elapsed = self.get_elapsed_duration()
        if elapsed < self.duration:
            return False
        measured_pose = self.op.motion_manager.get_measured_data(
            DataKey.MEASURED_EEF_POSE, self.op.obs
        )[:3]
        error = np.linalg.norm(measured_pose - self.goal_se3.translation)
        if error < self.STREAM_GOAL_TOLERANCE or (
            elapsed > self.duration + self.STREAM_EXTRA_WAIT
        ):
            if error >= self.STREAM_GOAL_TOLERANCE:
                print(
                    f"[{self.op.__class__.__name__}] Relative move ended "
                    f"{error * 1000:.0f} mm short of its goal."
                )
            hold_at_measured(self.op, refresh_obs=False)
            return True
        return False


class GripperPhase(ScriptedPhaseBase):
    """Open or close the gripper and wait for it."""

    def __init__(self, op, label, close, settle_sec=None):
        super().__init__(op, label)
        self.close = close
        self.settle_sec = settle_sec

    def start(self):
        super().start()
        env = self.op.env.unwrapped
        gripper_dim = DataKey.get_dim(DataKey.COMMAND_GRIPPER_JOINT_POS, self.op.env)
        if hasattr(env, "set_gripper_blocking"):
            print(
                f"[{self.op.__class__.__name__}] "
                f"{'Close' if self.close else 'Open'} the gripper."
            )
            value = env.set_gripper_blocking(self.close, settle_sec=self.settle_sec)
            self.duration = 0.0
        else:
            value = (
                env.action_space.high[-1] if self.close else env.action_space.low[-1]
            )
            self.duration = 0.5 if self.settle_sec is None else self.settle_sec
        # The next env.step() sends the motion manager's gripper command; it
        # must agree, or it immediately undoes this.
        self.op.motion_manager.set_command_data(
            DataKey.COMMAND_GRIPPER_JOINT_POS, np.full(gripper_dim, value)
        )
        hold_at_measured(self.op)

    def check_transition(self):
        return self.get_elapsed_duration() >= self.duration


def make_primitive_phase(op, spec, named_poses, label=None):
    """One scripted phase from a primitive spec (see the module docstring).
    Also accepts a full segment entry ({name, type: scripted_move|gripper,
    ...})."""
    spec = dict(spec)
    seg_type = spec.pop("type", None)
    name = spec.pop("name", None)
    if label is None:
        label = name
    if "joint_pos" in spec:
        pose_name = spec["joint_pos"]
        return JointMovePhase(
            op,
            label or f"move to {pose_name}",
            pose_name,
            named_poses,
            vel=spec.get("vel", 10.0),
        )
    if "rel_xyz" in spec:
        return RelativeMovePhase(
            op,
            label or f"move {spec['rel_xyz']}",
            spec["rel_xyz"],
            vel=spec.get("vel", 10.0),
            speed=spec.get("speed", 0.05),
            mode=spec.get("mode", "movej"),
        )
    if "gripper" in spec or seg_type == "gripper":
        value = spec.get("gripper")
        if value is None:
            value = "open" if spec.get("open", True) else "close"
        if value not in ("open", "close"):
            raise ValueError(f"Invalid gripper value (open/close): {value}")
        return GripperPhase(
            op,
            label or f"gripper {value}",
            close=(value == "close"),
            settle_sec=spec.get("settle_sec"),
        )
    raise ValueError(f"Invalid scripted step: {spec} (type: {seg_type})")
