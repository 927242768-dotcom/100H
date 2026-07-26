import argparse
import tempfile
import threading
import time
import unittest
from pathlib import Path

import cv2
import numpy as np
from detector import Detection
from pipeline import (
    AsyncFrameDetectorRunner,
    StaticImageSource,
    build_static_image_tiles,
    configure_static_image_profile,
    deduplicate_static_plate_detections,
    image_navigation_step,
    run_detector_once,
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

    def test_image_directory_natural_order_and_wraparound(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            nested = root / "nested"
            nested.mkdir()
            image = np.zeros((24, 32, 3), dtype=np.uint8)
            cv2.imwrite(str(root / "plate10.jpg"), image)
            cv2.imwrite(str(root / "plate2.jpg"), image)
            cv2.imwrite(str(nested / "plate1.png"), image)
            (root / "ignore.txt").write_text("not an image", encoding="utf-8")

            source = StaticImageSource(str(root), refresh_fps=1000.0)
            try:
                self.assertEqual(source.image_count, 3)
                self.assertEqual(source.current_source_name, "plate1.png")
                self.assertTrue(source.move(1))
                self.assertEqual(source.current_source_name, "plate2.jpg")
                self.assertTrue(source.move(1))
                self.assertEqual(source.current_source_name, "plate10.jpg")
                self.assertTrue(source.move(1))
                self.assertEqual(source.current_source_name, "plate1.png")
                self.assertTrue(source.move(-1))
                self.assertEqual(source.current_source_name, "plate10.jpg")
            finally:
                source.release()

    def test_navigation_keys(self):
        for key in (10, 13, 32, ord("d"), 65363):
            self.assertEqual(image_navigation_step(key), 1)
        for key in (8, 127, ord("a"), 65361):
            self.assertEqual(image_navigation_step(key), -1)
        self.assertEqual(image_navigation_step(ord("q")), 0)

    def test_async_frame_reset_discards_previous_image(self):
        class BlockingDetector:
            def __init__(self):
                self.started = threading.Event()
                self.release_first = threading.Event()
                self.calls = 0
                self.reset_calls = 0

            def detect(self, frame):
                self.calls += 1
                value = int(frame[0, 0, 0])
                if self.calls == 1:
                    self.started.set()
                    self.release_first.wait(timeout=2.0)
                return [Detection(label=str(value), score=1.0, box=(0, 0, 1, 1))]

            def reset(self):
                self.reset_calls += 1

        detector = BlockingDetector()
        runner = AsyncFrameDetectorRunner(detector)
        try:
            runner.submit(np.full((2, 2, 3), 1, dtype=np.uint8))
            self.assertTrue(detector.started.wait(timeout=1.0))
            runner.reset()
            runner.submit(np.full((2, 2, 3), 2, dtype=np.uint8))
            detector.release_first.set()

            deadline = time.monotonic() + 2.0
            latest = []
            while time.monotonic() < deadline:
                latest, _, _, completed, _, _ = runner.snapshot()
                if completed > 0 and latest:
                    break
                time.sleep(0.01)

            self.assertEqual([item.label for item in latest], ["2"])
            self.assertGreaterEqual(detector.reset_calls, 1)
        finally:
            detector.release_first.set()
            runner.close()

    def test_static_multiscale_search_can_be_cancelled_between_inferences(self):
        class CountingDetector:
            def __init__(self):
                self.calls = 0

            def detect(self, _frame):
                self.calls += 1
                return []

        detector = CountingDetector()
        args = argparse.Namespace(
            detector_source="full",
            detector_accuracy_priority=True,
            detector_min_score=0.05,
            detector="rknn",
            detector_max_rois=20,
            hyperlpr_max_num=20,
            detector_search_input_width=0,
            detector_track_input_width=0,
            detector_input_width=0,
            input_image="/mnt/sdcard",
            detector_merge_iou=0.55,
        )
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        detections, mode = run_detector_once(
            detector,
            frame,
            [],
            args,
            cancel_check=lambda: detector.calls >= 1,
        )

        self.assertEqual(detections, [])
        self.assertEqual(mode, "cancelled")
        self.assertEqual(detector.calls, 1)


if __name__ == "__main__":
    unittest.main()
