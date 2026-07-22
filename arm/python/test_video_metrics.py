from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from detector import Detection
from evaluation_metrics import MetricsRecorder
from pipeline import VideoFileSource


class VideoFileSourceTest(unittest.TestCase):
    def test_reads_all_frames_and_reports_eof(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video_path = Path(directory) / "sample.avi"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10.0,
                (64, 48),
            )
            self.assertTrue(writer.isOpened())
            for value in (20, 80, 140):
                writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
            writer.release()

            source = VideoFileSource(str(video_path), loop=False, playback="fast")
            frames = []
            while True:
                ok, frame = source.read()
                if not ok:
                    break
                frames.append((source.current_frame_index, frame))
            source.release()

            self.assertEqual([item[0] for item in frames], [1, 2, 3])
            self.assertTrue(source.ended)
            self.assertEqual(frames[0][1].shape[:2], (48, 64))


class MetricsRecorderTest(unittest.TestCase):
    def test_generates_csv_and_accuracy_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ground_truth_path = root / "ground_truth.json"
            ground_truth_path.write_text(
                json.dumps(
                    {
                        "frames": [
                            {
                                "frame": 1,
                                "plates": [
                                    {"box": [10, 10, 40, 16], "text": "苏ED51712", "type": "绿牌新能源"}
                                ],
                            },
                            {
                                "frame": 2,
                                "plates": [
                                    {"box": [12, 10, 40, 16], "text": "苏ED51712", "type": "绿牌新能源"}
                                ],
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            output_dir = root / "metrics"
            recorder = MetricsRecorder(
                str(output_dir),
                source={"kind": "video", "path": "sample.mp4"},
                arguments={"detector": "rknn"},
                ground_truth_path=str(ground_truth_path),
                iou_threshold=0.5,
            )
            detection = Detection(
                label="苏ED51712",
                raw_label="苏ED51712",
                type_name="绿牌新能源",
                full_text="绿牌新能源, 苏ED51712",
                score=0.9,
                box=(10, 10, 40, 16),
            )
            common = {
                "source_time_s": 0.0,
                "source_name": "sample.mp4",
                "real_fps": 10.0,
                "threshold": 128,
                "candidate_boxes": 1,
                "people": [],
                "detector_mode": "full",
                "detector_busy": False,
                "fpga_busy": False,
                "fpga_ms": 2.0,
                "detector_ms": 20.0,
                "person_ms": 0.0,
                "fpga_updated": True,
                "person_updated": False,
                "active_pixels": 100,
            }
            recorder.record_frame(
                processed_frame=1,
                source_frame=1,
                plates=[detection],
                detector_generation=1,
                detector_updated=True,
                detector_result=[detection],
                detector_source_frame=1,
                detector_source_time_s=0.0,
                detector_source_name="sample.mp4",
                **common,
            )
            recorder.record_frame(
                processed_frame=2,
                source_frame=2,
                plates=[],
                detector_generation=2,
                detector_updated=True,
                detector_result=[],
                detector_source_frame=2,
                detector_source_time_s=0.1,
                detector_source_name="sample.mp4",
                **common,
            )
            summary = recorder.close()

            self.assertEqual(summary["accuracy"]["true_positive"], 1)
            self.assertEqual(summary["accuracy"]["false_negative"], 1)
            self.assertAlmostEqual(summary["accuracy"]["recall"], 0.5)
            self.assertAlmostEqual(summary["accuracy"]["ocr_exact_accuracy"], 1.0)
            self.assertTrue((output_dir / "frames.csv").is_file())
            self.assertTrue((output_dir / "detections.csv").is_file())
            self.assertTrue((output_dir / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
