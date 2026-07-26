import argparse
import unittest

from detector import Detection
from pipeline import (
    build_static_image_tiles,
    configure_static_image_profile,
    deduplicate_static_plate_detections,
)


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

    def test_static_duplicate_prefers_complete_plate_text(self):
        detections = [
            Detection(
                label="沪AF710",
                raw_label="沪AF710",
                type_name="白色警用",
                full_text="白色警用, 沪AF710",
                score=0.92,
                box=(920, 820, 280, 120),
            ),
            Detection(
                label="沪AF71017",
                raw_label="沪AF71017",
                type_name="新能源车牌",
                full_text="新能源车牌, 沪AF71017",
                score=0.78,
                box=(900, 800, 430, 160),
            ),
            Detection(
                label="单层车牌",
                score=0.95,
                box=(1180, 805, 155, 155),
            ),
        ]

        result = deduplicate_static_plate_detections(detections)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].raw_label, "沪AF71017")
        self.assertEqual(result[0].box, (900, 800, 430, 160))

    def test_static_dedup_keeps_adjacent_real_plates(self):
        detections = [
            Detection(
                label="苏ED51712",
                raw_label="苏ED51712",
                score=0.80,
                box=(100, 200, 220, 70),
            ),
            Detection(
                label="粤SP8888",
                raw_label="粤SP8888",
                score=0.82,
                box=(340, 200, 220, 70),
            ),
        ]

        result = deduplicate_static_plate_detections(detections)

        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
