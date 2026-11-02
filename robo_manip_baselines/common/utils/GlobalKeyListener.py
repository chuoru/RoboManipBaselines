"""System-wide key listener for foot pedals (and keyboard fallbacks).

cv2.waitKey() only sees keys while the OpenCV window has focus, which an
operator with both hands on a SpaceMouse cannot guarantee. Foot pedals that
enumerate as a keyboard send ordinary key events, so a pynput listener (the
same mechanism as ViveInputDevice's gripper key bindings) catches them no
matter which window is focused.

Physical keys are mapped to LOGICAL action names ("fail", "intervene",
"success") by a YAML file (common/configs/pedal_default.yaml); what each
action means is up to the caller (TeleopBase maps them to its 'f'/'n'/'s'
keys, the segment runner to fail/intervene/success).
"""

import os
import threading
import time

import yaml

DEFAULT_PEDAL_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "pedal_default.yaml"
)


def get_pynput_key_name(key):
    """Stable, lowercase name of a pynput key: the character for printable
    keys ("a", "1"), the Key enum name otherwise ("f13", "page_down"). None if
    the key has neither (some vendor keys only carry a virtual-key code)."""
    char = getattr(key, "char", None)
    if char is not None:
        return char.lower()
    name = getattr(key, "name", None)
    if name is not None:
        return name.lower()
    vk = getattr(key, "vk", None)
    if vk is not None:
        return f"vk{vk}"
    return None


def load_pedal_config(path=None):
    if path is None:
        path = DEFAULT_PEDAL_CONFIG_PATH
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


class GlobalKeyListener:
    """Collects logical actions from system-wide key presses.

    poll() returns (and clears) the set of actions triggered since the last
    call. Each physical press fires once: OS key auto-repeat while a pedal is
    held down is ignored until the key is released, and presses of the same
    action closer together than debounce_sec are merged (a pedal's contact
    can bounce).
    """

    def __init__(self, action_keys, debounce_sec=0.3, enabled_actions=None):
        # action name -> list of key names
        self.key_to_action = {}
        for action, keys in action_keys.items():
            if isinstance(keys, str):
                keys = [keys]
            for key in keys or []:
                self.key_to_action[str(key).lower()] = action
        self.debounce_sec = debounce_sec
        self.enabled_actions = None if enabled_actions is None else set(enabled_actions)

        self._lock = threading.Lock()
        self._pending = set()
        self._pressed_keys = set()
        self._last_fire_time = {}
        self._listener = None
        self.available = False

    @classmethod
    def from_config(cls, path=None, enabled_actions=None):
        config = load_pedal_config(path)
        return cls(
            config.get("pedal_keys", {}),
            debounce_sec=config.get("debounce_sec", 0.3),
            enabled_actions=enabled_actions,
        )

    def start(self):
        try:
            from pynput import keyboard
        except Exception as e:
            # Pedals are a convenience; the cv2 keys keep working without
            # them, so a missing/unusable pynput must not stop teleop.
            print(
                f"[{self.__class__.__name__}] Global key listener disabled "
                f"(pynput unavailable: {e}). Use the OpenCV window keys instead."
            )
            self.available = False
            return False

        try:
            self._listener = keyboard.Listener(
                on_press=self._on_press, on_release=self._on_release
            )
            self._listener.daemon = True
            self._listener.start()
        except Exception as e:
            # e.g. no X display on Linux
            print(
                f"[{self.__class__.__name__}] Global key listener disabled "
                f"(failed to start: {e}). Use the OpenCV window keys instead."
            )
            self._listener = None
            self.available = False
            return False

        self.available = True
        bindings = ", ".join(
            f"{key}->{action}" for key, action in sorted(self.key_to_action.items())
        )
        print(f"[{self.__class__.__name__}] Listening for pedal keys: {bindings}")
        return True

    def stop(self):
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None
        self.available = False

    def set_enabled_actions(self, enabled_actions):
        """Restrict which actions are reported (None = all). Presses of
        disabled actions are dropped, not queued."""
        with self._lock:
            self.enabled_actions = (
                None if enabled_actions is None else set(enabled_actions)
            )
            if self.enabled_actions is not None:
                self._pending &= self.enabled_actions

    def _on_press(self, key):
        self.handle_key_press(get_pynput_key_name(key))

    def _on_release(self, key):
        self.handle_key_release(get_pynput_key_name(key))

    def handle_key_press(self, key_name, now=None):
        """Feed one key press (also used directly by tests)."""
        if key_name is None:
            return
        if now is None:
            now = time.time()
        with self._lock:
            if key_name in self._pressed_keys:
                return  # auto-repeat while held
            self._pressed_keys.add(key_name)
            action = self.key_to_action.get(key_name)
            if action is None:
                return
            if self.enabled_actions is not None and action not in self.enabled_actions:
                return
            last = self._last_fire_time.get(action)
            if last is not None and now - last < self.debounce_sec:
                return
            self._last_fire_time[action] = now
            self._pending.add(action)

    def handle_key_release(self, key_name):
        if key_name is None:
            return
        with self._lock:
            self._pressed_keys.discard(key_name)

    def poll(self):
        with self._lock:
            actions = self._pending
            self._pending = set()
        return actions
