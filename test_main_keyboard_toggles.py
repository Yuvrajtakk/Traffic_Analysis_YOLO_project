import unittest
from unittest.mock import patch

import numpy as np

from main import (
    KEY_TOGGLE_MAP,
    build_status_text,
    resize_frame_for_display,
    should_quit,
    toggle_module_state,
)


class KeyboardToggleTests(unittest.TestCase):
    def test_toggle_module_state_flips_expected_flags(self):
        module_state = {
            "stationary": True,
            "wrong_way": True,
            "hazards": True,
            "congestion": True,
        }

        toggle_module_state(module_state, ord("s"))
        self.assertFalse(module_state["stationary"])
        self.assertTrue(module_state["wrong_way"])
        self.assertTrue(module_state["hazards"])
        self.assertTrue(module_state["congestion"])

        toggle_module_state(module_state, ord("w"))
        self.assertFalse(module_state["wrong_way"])

    def test_status_text_reports_on_and_off_state(self):
        module_state = {
            "stationary": True,
            "wrong_way": False,
            "hazards": True,
            "congestion": False,
        }

        status_text = build_status_text(module_state)

        self.assertIn("S:ON", status_text)
        self.assertIn("W:OFF", status_text)
        self.assertIn("H:ON", status_text)
        self.assertIn("C:OFF", status_text)
        self.assertTrue(status_text.startswith("S:ON"))

    def test_toggle_map_contains_expected_keys(self):
        self.assertEqual(KEY_TOGGLE_MAP[ord("s")], "stationary")
        self.assertEqual(KEY_TOGGLE_MAP[ord("w")], "wrong_way")
        self.assertEqual(KEY_TOGGLE_MAP[ord("h")], "hazards")
        self.assertEqual(KEY_TOGGLE_MAP[ord("c")], "congestion")

    def test_should_quit_on_q_or_closed_window(self):
        self.assertTrue(should_quit(ord("q"), "Traffic Dashboard"))

        with patch("main.cv2.getWindowProperty", return_value=0):
            self.assertTrue(should_quit(0, "Traffic Dashboard"))

        with patch("main.cv2.getWindowProperty", return_value=1):
            self.assertFalse(should_quit(0, "Traffic Dashboard"))

    def test_resize_frame_for_display_scales_large_frame(self):
        frame = np.zeros((1000, 1500, 3), dtype=np.uint8)

        resized = resize_frame_for_display(frame, max_width=1280, max_height=720)

        self.assertLessEqual(resized.shape[1], 1280)
        self.assertLessEqual(resized.shape[0], 720)
        self.assertGreater(resized.shape[1], 0)
        self.assertGreater(resized.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
