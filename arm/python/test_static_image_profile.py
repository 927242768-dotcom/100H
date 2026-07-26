import argparse
import unittest

from pipeline import build_static_image_tiles, configure_static_image_profile


def make_args(**overrides):
    values = {
        "input_image": "",
        "disable_sd_image_accuracy": False,
        "detector_accuracy_priority": False,
        "detector_interval": 3,
        "detector_interval_hit": 4,
        "detector_submit_max_gap_seconds": 0.5,
        "detector": "rknn",
        "rknn_conf_threshold": 0.15,
        "detector_fast_pass_width": 0,
        "rknn_input_size": 640,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class StaticImageProfileTest(unittest.TestCase):
    def test_camera_configuration_is_unchanged(self):
        args = make_args()
        before = vars(args).copy()

        self.assertFalse(configure_static_image_profile(args))
        self.assertEqual(vars(args), before)

    def test_static_image_enables_accuracy_profile(self):
        args = make_args(input_image="/mnt/sdcard/test.jpg")

        self.assertTrue(configure_static_image_profile(args))
        self.assertTrue(args.detector_accuracy_priority)
        self.assertEqual(args.detector_interval, 1)
        self.assertEqual(args.detector_interval_hit, 1)
        self.assertEqual(args.detector_submit_max_gap_seconds, 0.0)
        self.assertEqual(args.rknn_conf_threshold, 0.10)
        self.assertEqual(args.detector_fast_pass_width, 640)

    def test_static_profile_can_be_disabled(self):
        args = make_args(
            input_image="/mnt/sdcard/test.jpg",
            disable_sd_image_accuracy=True,
        )
        before = vars(args).copy()

        self.assertFalse(configure_static_image_profile(args))
        self.assertEqual(vars(args), before)

    def test_static_tiles_cover_image_at_two_scales(self):
        tiles = build_static_image_tiles(4000, 3000)

        self.assertEqual(len(tiles), 13)
        self.assertIn((0, 0, 2400, 1800), tiles)
        self.assertIn((1600, 1200, 2400, 1800), tiles)
        self.assertIn((0, 0, 1600, 1200), tiles)
        self.assertIn((2400, 1800, 1600, 1200), tiles)


if __name__ == "__main__":
    unittest.main()
