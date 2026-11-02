"""Print the name of every key pressed anywhere on the system, and the
logical action it is bound to in a pedal config.

Use it to find what each foot pedal sends before editing
common/configs/pedal_default.yaml. The names printed here are exactly the
ones that file expects.

Usage:
    python ./misc/CheckPedalKeys.py [--pedal_config ./common/configs/pedal_default.yaml]
Press Esc (or Ctrl+C) to quit.
"""

import argparse

from robo_manip_baselines.common.utils.GlobalKeyListener import (
    GlobalKeyListener,
    get_pynput_key_name,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pedal_config",
        type=str,
        default=None,
        help="pedal config to show the bound action from (default: common/configs/pedal_default.yaml)",
    )
    args = parser.parse_args()

    from pynput import keyboard

    key_to_action = GlobalKeyListener.from_config(args.pedal_config).key_to_action
    print("Press keys / pedals (Esc to quit).")

    def on_press(key):
        name = get_pynput_key_name(key)
        action = key_to_action.get(name)
        print(
            f"pressed: {name!r:16} raw: {key!r:24} -> "
            f"{action if action is not None else '(not bound)'}",
            flush=True,
        )
        if key == keyboard.Key.esc:
            return False

    with keyboard.Listener(on_press=on_press) as listener:
        try:
            listener.join()
        except KeyboardInterrupt:
            pass
