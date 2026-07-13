import threading

import numpy as np
import pinocchio as pin

from .InputDeviceBase import InputDeviceBase


class KeyboardInputDevice(InputDeviceBase):
    """Keyboard for teleoperation input device."""

    # Default key bindings. Override via the key_bindings constructor arg (e.g. to
    # run two KeyboardInputDevice instances off one physical keyboard for dual-arm
    # teleop, with non-overlapping key sets).
    DEFAULT_KEY_BINDINGS = {
        "x_pos": "w",
        "x_neg": "s",
        "y_pos": "a",
        "y_neg": "d",
        "z_pos": "q",
        "z_neg": "e",
        "roll_neg": "j",
        "roll_pos": "l",
        "pitch_pos": "i",
        "pitch_neg": "k",
        "yaw_pos": "u",
        "yaw_neg": "o",
        "gripper_close": "z",
        "gripper_open": "x",
    }

    def __init__(
        self,
        arm_manager,
        pos_scale=1e-2,
        rpy_scale=5e-2,
        gripper_scale=50.0,
        key_bindings=None,
    ):
        super().__init__()

        self.arm_manager = arm_manager
        self.pos_scale = pos_scale
        self.rpy_scale = rpy_scale
        self.gripper_scale = gripper_scale
        self.key_bindings = key_bindings or self.DEFAULT_KEY_BINDINGS

        self.state = {key: False for key in self.key_bindings.values()}

        self.listener = None
        self.listener_thread = None

    def connect(self):
        if self.connected:
            return

        from pynput import keyboard

        self.listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )

        # keyboard listener another thread
        self.listener_thread = threading.Thread(target=self._start_listener)
        self.listener_thread.daemon = True
        self.listener_thread.start()

        self.connected = True
        print(f"[{self.__class__.__name__}] Connected.")
        kb = self.key_bindings
        print(f"""[{self.__class__.__name__}] Key Bindings:
  - {kb['x_pos'].upper()}{kb['x_neg'].upper()}{kb['y_pos'].upper()}{kb['y_neg'].upper()} : XY movement
  - {kb['z_pos'].upper()}{kb['z_neg'].upper()}   : Z-axis movement
  - {kb['roll_neg'].upper()}{kb['pitch_pos'].upper()}{kb['pitch_neg'].upper()}{kb['roll_pos'].upper()} : Roll and Pitch rotation
  - {kb['yaw_pos'].upper()}{kb['yaw_neg'].upper()}  : Yaw rotation
  - {kb['gripper_close'].upper()}/{kb['gripper_open'].upper()}  : Gripper open/close""")

    def _start_listener(self):
        self.listener.start()
        self.listener.join()

    def _on_press(self, key):
        try:
            k = key.char.lower()
            if k in self.state:
                self.state[k] = True
        except AttributeError:
            pass

    def _on_release(self, key):
        try:
            k = key.char.lower()
            if k in self.state:
                self.state[k] = False
        except AttributeError:
            pass

    def read(self):
        if not self.connected:
            raise RuntimeError(f"[{self.__class__.__name__}] Device is not connected.")

    def set_command_data(self):
        kb = self.key_bindings
        delta_pos = np.zeros(3)

        # X-axis
        if self.state[kb["x_pos"]]:
            delta_pos[0] += self.pos_scale
        if self.state[kb["x_neg"]]:
            delta_pos[0] -= self.pos_scale

        # Y-axis
        if self.state[kb["y_pos"]]:
            delta_pos[1] += self.pos_scale
        if self.state[kb["y_neg"]]:
            delta_pos[1] -= self.pos_scale

        # Z-axis
        if self.state[kb["z_pos"]]:
            delta_pos[2] += self.pos_scale
        if self.state[kb["z_neg"]]:
            delta_pos[2] -= self.pos_scale

        delta_rpy = np.zeros(3)

        # Roll
        if self.state[kb["roll_neg"]]:
            delta_rpy[0] -= self.rpy_scale
        if self.state[kb["roll_pos"]]:
            delta_rpy[0] += self.rpy_scale

        # Pitch
        if self.state[kb["pitch_pos"]]:
            delta_rpy[1] += self.rpy_scale
        if self.state[kb["pitch_neg"]]:
            delta_rpy[1] -= self.rpy_scale

        # Yaw
        if self.state[kb["yaw_pos"]]:
            delta_rpy[2] += self.rpy_scale * 2.0
        if self.state[kb["yaw_neg"]]:
            delta_rpy[2] -= self.rpy_scale * 2.0

        target_se3 = self.arm_manager.target_se3.copy()
        target_se3.translation += delta_pos
        target_se3.rotation = pin.rpy.rpyToMatrix(*delta_rpy) @ target_se3.rotation

        self.arm_manager.set_command_eef_pose(target_se3)

        # Set gripper command
        gripper_joint_pos = self.arm_manager.get_command_gripper_joint_pos().copy()

        if self.state[kb["gripper_close"]] and not self.state[kb["gripper_open"]]:
            gripper_joint_pos += self.gripper_scale
        elif self.state[kb["gripper_open"]] and not self.state[kb["gripper_close"]]:
            gripper_joint_pos -= self.gripper_scale

        self.arm_manager.set_command_gripper_joint_pos(gripper_joint_pos)

    def disconnect(self):
        if self.connected:
            if self.listener:
                self.listener.stop()
            self.connected = False
            print(f"[{self.__class__.__name__}] Disconnected.")
