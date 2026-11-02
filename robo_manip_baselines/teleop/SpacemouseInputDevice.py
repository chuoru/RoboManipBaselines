import time

import numpy as np
import pinocchio as pin

from .InputDeviceBase import InputDeviceBase


class SpacemouseInputDevice(InputDeviceBase):
    """Spacemouse for teleoperation input device.

    Every option added after the original pos_scale/rpy_scale/gripper_scale/
    device_params defaults to the original behaviour, so existing configs are
    unaffected. See teleop/configs/SpaceMouseRealFairino5.yaml for a
    SpaceMouse Compact + FR5 setup that uses them.
    """

    # Order of the six axes in every per-axis option (axis_sign, axis_enable)
    # and in the internal axis vector: the device's own x/y/z/roll/pitch/yaw.
    AXIS_NAMES = ("x", "y", "z", "roll", "pitch", "yaw")

    def __init__(
        self,
        arm_manager,
        pos_scale=1e-2,
        rpy_scale=5e-3,
        gripper_scale=5.0,
        device_params={},
        # Per-axis deadzone on the raw axis value (|v| < deadzone -> 0). The
        # SpaceMouse cap rarely rests at exactly 0, and without a deadzone a
        # hand resting on it slowly drags the target.
        deadzone=0.0,
        # EMA over the axis values (1.0 = off). Smooths the jitter of a hand
        # pressing the cap; lower = smoother but laggier.
        smoothing_alpha=1.0,
        # Per-axis sign flip / enable mask, in AXIS_NAMES order. E.g. disable
        # roll and pitch ([1, 1, 1, 0, 0, 1]) so a top-down grasp can never be
        # tilted by accident.
        axis_sign=None,
        axis_enable=None,
        # "world": rotation deltas are applied in the robot base frame (the
        # original behaviour). "tool": in the end-effector's own frame.
        rotation_frame="world",
        # "incremental": the original behaviour (button 0 closes, the last
        # button opens, by gripper_scale per tick while held). "toggle": a
        # press of gripper_button (rising edge) flips the command between
        # gripper_open_value and gripper_closed_value -- for a binary gripper
        # (FR5 tool_do), where intermediate commands only smear the edge.
        gripper_mode="incremental",
        gripper_button=0,
        gripper_open_value=0.0,
        gripper_closed_value=100.0,
        # Button whose rising edge toggles precision mode (pos/rot speed x
        # precision_scale) for the final alignment. None = no precision mode.
        precision_button=None,
        precision_scale=1.0 / 3.0,
        # Velocity mode: when set, a full deflection moves the target at this
        # many m/s (rad/s) independent of the loop rate, instead of
        # pos_scale (rpy_scale) per tick. The per-tick scales tie the speed to
        # the loop period, which differs between teleop (~10-20 Hz) and a
        # 5 Hz rollout loop that the same device is used to intervene in.
        max_lin_vel=None,
        max_ang_vel=None,
        # Upper bound [s] on the time step used in velocity mode, so one slow
        # tick (camera stall) cannot turn into a large jump of the target.
        max_dt=0.25,
    ):
        super().__init__()

        self.arm_manager = arm_manager
        self.pos_scale = pos_scale
        self.rpy_scale = rpy_scale
        self.gripper_scale = gripper_scale
        self.device_params = device_params
        self.deadzone = deadzone
        self.smoothing_alpha = smoothing_alpha
        self.axis_sign = (
            np.ones(6) if axis_sign is None else np.array(axis_sign, dtype=np.float64)
        )
        self.axis_enable = (
            np.ones(6)
            if axis_enable is None
            else np.array(axis_enable, dtype=np.float64)
        )
        if self.axis_sign.shape != (6,) or self.axis_enable.shape != (6,):
            raise ValueError(
                f"[{self.__class__.__name__}] axis_sign/axis_enable need 6 values "
                f"({', '.join(self.AXIS_NAMES)})."
            )
        if rotation_frame not in ("world", "tool"):
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid rotation_frame: {rotation_frame}"
            )
        self.rotation_frame = rotation_frame
        if gripper_mode not in ("incremental", "toggle"):
            raise ValueError(
                f"[{self.__class__.__name__}] Invalid gripper_mode: {gripper_mode}"
            )
        self.gripper_mode = gripper_mode
        self.gripper_button = gripper_button
        self.gripper_open_value = gripper_open_value
        self.gripper_closed_value = gripper_closed_value
        self.precision_button = precision_button
        self.precision_scale = precision_scale
        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel
        self.max_dt = max_dt

        self.precision_mode = False
        self.reset_filter()

    def connect(self):
        if self.connected:
            return

        import pyspacemouse

        # An empty device_params opens the first device found. Passing
        # unrelated keys (e.g. a Linux hidraw path) on Windows is what usually
        # fails there, so do not pass anything that was not configured.
        try:
            spacemouse = pyspacemouse.open(**self.device_params)
        except Exception as e:
            raise RuntimeError(self._connect_error_message(e)) from e
        # pyspacemouse returns None (rather than raising) when no device
        # matched or the HID device could not be opened.
        if spacemouse is None:
            raise RuntimeError(self._connect_error_message(None))
        self.spacemouse = spacemouse
        self.connected = True

    def _connect_error_message(self, error):
        msg = (
            f"[{self.__class__.__name__}] Failed to open the SpaceMouse "
            f"(device_params: {self.device_params})"
        )
        if error is not None:
            msg += f": {error}"
        msg += (
            ". Check that it is plugged in and that device_params.device names "
            "a model pyspacemouse knows (e.g. 'SpaceMouse Compact'). On Windows "
            "pyspacemouse needs hidapi.dll on the PATH (or next to python.exe), "
            "and the 3Dconnexion driver (3DxWare) can hold the device -- quit "
            "it, or disable it for python.exe, if opening keeps failing. On "
            "Linux, the user needs read access to /dev/hidraw* (udev rule)."
        )
        return msg

    def read(self):
        if not self.connected:
            raise RuntimeError(f"[{self.__class__.__name__}] Device is not connected.")

        # Empirically, you can call read repeatedly to get the latest device state
        for i in range(10):
            self.state = self.spacemouse.read()

    def reset_filter(self):
        """Forget the EMA state and button edges, e.g. when the device takes
        over from a policy, so stale deflection from the last time it was used
        is not replayed."""
        self._filtered_axes = None
        self._last_axes = np.zeros(6)
        self._prev_buttons = None
        self._last_command_time = None

    def input_norm(self):
        """Norm of the latest processed (deadzoned, filtered, masked) axis
        values, in [0, sqrt(6)]. 0 while the operator is not touching the
        cap -- used to tell a human correction from the reaction delay right
        after an intervention starts."""
        return float(np.linalg.norm(self._last_axes))

    def _get_axes(self):
        raw = np.array(
            [
                self.state.x,
                self.state.y,
                self.state.z,
                self.state.roll,
                self.state.pitch,
                self.state.yaw,
            ],
            dtype=np.float64,
        )
        raw[np.abs(raw) < self.deadzone] = 0.0
        if self._filtered_axes is None or self.smoothing_alpha >= 1.0:
            self._filtered_axes = raw
        else:
            alpha = self.smoothing_alpha
            self._filtered_axes = alpha * raw + (1.0 - alpha) * self._filtered_axes
        axes = self._filtered_axes * self.axis_sign * self.axis_enable
        self._last_axes = axes
        return axes

    def _button_rising_edges(self):
        buttons = list(self.state.buttons)
        if self._prev_buttons is None or len(self._prev_buttons) != len(buttons):
            # First read: nothing can have been "pressed" yet, otherwise a
            # button held while connecting would toggle immediately.
            self._prev_buttons = buttons
            return set()
        edges = {
            idx
            for idx, (prev, curr) in enumerate(zip(self._prev_buttons, buttons))
            if curr > 0 and prev <= 0
        }
        self._prev_buttons = buttons
        return edges

    def set_command_data(self):
        axes = self._get_axes()
        edges = self._button_rising_edges()
        if self.precision_button is not None and self.precision_button in edges:
            self.precision_mode = not self.precision_mode
            print(
                f"[{self.__class__.__name__}] Precision mode "
                f"{'ON' if self.precision_mode else 'OFF'}."
            )

        now = time.time()
        if self._last_command_time is None:
            dt = 0.0
        else:
            dt = min(now - self._last_command_time, self.max_dt)
        self._last_command_time = now

        if self.max_lin_vel is None:
            pos_gain = self.pos_scale
        else:
            pos_gain = self.max_lin_vel * dt
        if self.max_ang_vel is None:
            rpy_gain = self.rpy_scale
        else:
            rpy_gain = self.max_ang_vel * dt
        if self.precision_mode:
            pos_gain *= self.precision_scale
            rpy_gain *= self.precision_scale

        # Set arm command
        delta_pos = pos_gain * np.array(
            [
                -1.0 * axes[1],
                axes[0],
                axes[2],
            ]
        )
        delta_rpy = rpy_gain * np.array(
            [
                -1.0 * axes[3],
                -1.0 * axes[4],
                -2.0 * axes[5],
            ]
        )

        target_se3 = self.arm_manager.target_se3.copy()
        target_se3.translation += delta_pos
        if self.rotation_frame == "world":
            target_se3.rotation = pin.rpy.rpyToMatrix(*delta_rpy) @ target_se3.rotation
        else:
            target_se3.rotation = target_se3.rotation @ pin.rpy.rpyToMatrix(*delta_rpy)

        self.arm_manager.set_command_eef_pose(target_se3)

        # Set gripper command
        gripper_joint_pos = self.arm_manager.get_command_gripper_joint_pos().copy()

        if self.gripper_mode == "toggle":
            if self.gripper_button in edges:
                # Decide from the current command rather than an internal
                # flag, so the toggle stays right after something else (a
                # script, a policy) moved the gripper.
                midpoint = 0.5 * (self.gripper_open_value + self.gripper_closed_value)
                is_closed = np.mean(gripper_joint_pos) >= midpoint
                gripper_joint_pos[:] = (
                    self.gripper_open_value if is_closed else self.gripper_closed_value
                )
        elif self.state.buttons[0] > 0 and self.state.buttons[-1] <= 0:
            gripper_joint_pos += self.gripper_scale
        elif self.state.buttons[-1] > 0 and self.state.buttons[0] <= 0:
            gripper_joint_pos -= self.gripper_scale

        self.arm_manager.set_command_gripper_joint_pos(gripper_joint_pos)
