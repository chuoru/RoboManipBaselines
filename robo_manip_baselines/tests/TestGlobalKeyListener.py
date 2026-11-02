import unittest

from robo_manip_baselines.common.utils.GlobalKeyListener import (
    GlobalKeyListener,
    load_pedal_config,
)


class TestGlobalKeyListener(unittest.TestCase):
    def make_listener(self, **kwargs):
        return GlobalKeyListener(
            {"fail": ["f13", "1"], "intervene": "f14", "success": ["F15"]}, **kwargs
        )

    def test_press_maps_to_action_once(self):
        listener = self.make_listener(debounce_sec=0.3)
        listener.handle_key_press("f13", now=0.0)
        listener.handle_key_press("f13", now=0.5)  # auto-repeat while held
        self.assertEqual(listener.poll(), {"fail"})
        self.assertEqual(listener.poll(), set())
        listener.handle_key_release("f13")
        listener.handle_key_press("f13", now=1.0)
        self.assertEqual(listener.poll(), {"fail"})

    def test_debounce_and_case(self):
        listener = self.make_listener(debounce_sec=0.3)
        listener.handle_key_press("f15", now=0.0)
        listener.handle_key_release("f15")
        listener.handle_key_press("f15", now=0.1)  # contact bounce
        self.assertEqual(listener.poll(), {"success"})
        listener.handle_key_release("f15")
        listener.handle_key_press("f15", now=0.5)
        self.assertEqual(listener.poll(), {"success"})

    def test_unbound_and_disabled_actions(self):
        listener = self.make_listener(enabled_actions={"success"})
        listener.handle_key_press("x", now=0.0)
        listener.handle_key_press("1", now=0.0)
        listener.handle_key_press("f14", now=0.0)
        self.assertEqual(listener.poll(), set())
        listener.set_enabled_actions(None)
        listener.handle_key_release("f14")
        listener.handle_key_press("f14", now=1.0)
        self.assertEqual(listener.poll(), {"intervene"})

    def test_default_config_has_three_actions(self):
        config = load_pedal_config()
        self.assertEqual(set(config["pedal_keys"]), {"fail", "intervene", "success"})


if __name__ == "__main__":
    unittest.main()
