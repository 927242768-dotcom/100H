from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple


Box = Tuple[int, int, int, int]


@dataclass(frozen=True)
class GroundTruthObject:
    box: Box
    text: str = ""
    type_name: str = ""


class GroundTruthDataset:
    def __init__(self) -> None:
        self._by_frame: Dict[int, List[GroundTruthObject]] = {}
        self._by_source: Dict[str, List[GroundTruthObject]] = {}
        self._annotated_frames: set[int] = set()
        self._annotated_sources: set[str] = set()

    def add(
        self,
        *,
        frame_index: int | None,
        source_name: str,
        objects: Sequence[GroundTruthObject],
    ) -> None:
        if frame_index is not None:
            self._by_frame[frame_index] = list(objects)
            self._annotated_frames.add(frame_index)
        if source_name:
            key = Path(source_name).name
            self._by_source[key] = list(objects)
            self._annotated_sources.add(key)

    def lookup(self, frame_index: int, source_name: str) -> Tuple[bool, List[GroundTruthObject]]:
        source_key = Path(source_name).name if source_name else ""
        if source_key and source_key in self._annotated_sources:
            return True, list(self._by_source.get(source_key, []))
        if frame_index in self._annotated_frames:
            return True, list(self._by_frame.get(frame_index, []))
        return False, []

    @property
    def annotation_count(self) -> int:
        return len(self._annotated_sources) if self._annotated_sources else len(self._annotated_frames)


def _parse_box(value: Any) -> Box:
    if isinstance(value, Mapping):
        x = value.get("x", value.get("left", 0))
        y = value.get("y", value.get("top", 0))
        w = value.get("w", value.get("width", 0))
        h = value.get("h", value.get("height", 0))
        values = (x, y, w, h)
    else:
        values = value
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != 4:
        raise ValueError(f"标注框必须是 [x, y, w, h]，实际为: {value!r}")
    return tuple(int(round(float(item))) for item in values)  # type: ignore[return-value]


def _parse_ground_truth_object(value: Mapping[str, Any]) -> GroundTruthObject:
    box_value = value.get("box", value.get("bbox"))
    if box_value is None:
        raise ValueError(f"标注对象缺少 box/bbox: {value!r}")
    return GroundTruthObject(
        box=_parse_box(box_value),
        text=str(value.get("text", value.get("plate", value.get("label", "")))).strip(),
        type_name=str(value.get("type", value.get("type_name", ""))).strip(),
    )


def load_ground_truth(path: str) -> GroundTruthDataset:
    source_path = Path(path).expanduser().resolve()
    with source_path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)

    if isinstance(payload, Mapping):
        frame_values = payload.get("frames", payload.get("items", payload))
    else:
        frame_values = payload

    if isinstance(frame_values, Mapping):
        normalized_frames: List[Mapping[str, Any]] = []
        for key, value in frame_values.items():
            if isinstance(value, Mapping):
                record = dict(value)
                record.setdefault("frame", key)
            else:
                record = {"frame": key, "plates": value}
            normalized_frames.append(record)
        frame_values = normalized_frames

    if not isinstance(frame_values, Sequence) or isinstance(frame_values, (str, bytes)):
        raise ValueError("标注JSON必须包含 frames 数组或以帧号为键的对象")

    dataset = GroundTruthDataset()
    for record in frame_values:
        if not isinstance(record, Mapping):
            raise ValueError(f"frames 中的元素必须是对象: {record!r}")
        raw_index = record.get("frame_index", record.get("frame", record.get("index")))
        frame_index = int(raw_index) if raw_index not in (None, "") else None
        source_name = str(record.get("source", record.get("file", record.get("name", "")))).strip()
        object_values = record.get("plates", record.get("detections", record.get("objects", [])))
        if not isinstance(object_values, Sequence) or isinstance(object_values, (str, bytes)):
            raise ValueError(f"plates/detections 必须是数组: {object_values!r}")
        objects = [_parse_ground_truth_object(item) for item in object_values]
        dataset.add(frame_index=frame_index, source_name=source_name, objects=objects)
    return dataset


def box_iou(box_a: Box, box_b: Box) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = max(aw, 0) * max(ah, 0) + max(bw, 0) * max(bh, 0) - intersection
    return intersection / union if union > 0 else 0.0


def _normalize_plate_text(value: str) -> str:
    return "".join(value.upper().replace("O", "0").split())


def _normalize_plate_type(value: str) -> str:
    normalized = "".join(value.split())
    for suffix in ("车牌", "牌"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def _is_recognized_text(value: str) -> bool:
    normalized = value.strip()
    return bool(normalized) and normalized not in {"单层车牌", "双层车牌", "车牌"}


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _latency_summary(values: Sequence[float]) -> Dict[str, float | int | None]:
    return {
        "samples": len(values),
        "mean_ms": (sum(values) / len(values)) if values else None,
        "p50_ms": _percentile(values, 50.0),
        "p95_ms": _percentile(values, 95.0),
        "max_ms": max(values) if values else None,
    }


class MetricsRecorder:
    FRAME_FIELDS = (
        "processed_frame",
        "source_frame",
        "source_time_s",
        "source_name",
        "elapsed_s",
        "real_fps",
        "threshold",
        "candidate_boxes",
        "plate_count",
        "person_count",
        "violation_count",
        "detector_mode",
        "detector_busy",
        "fpga_busy",
        "fpga_ms",
        "detector_ms",
        "person_ms",
        "detector_generation",
        "detector_updated",
        "detector_source_frame",
        "detector_source_time_s",
        "active_pixels",
        "ground_truth_count",
        "true_positive",
        "false_positive",
        "false_negative",
    )
    DETECTION_FIELDS = (
        "processed_frame",
        "source_frame",
        "source_time_s",
        "source_name",
        "kind",
        "detection_index",
        "label",
        "raw_label",
        "type_name",
        "full_text",
        "score",
        "x",
        "y",
        "w",
        "h",
        "matched_ground_truth",
        "iou",
        "text_match",
        "type_match",
    )

    def __init__(
        self,
        output_dir: str,
        *,
        source: Mapping[str, Any],
        arguments: Mapping[str, Any],
        ground_truth_path: str = "",
        iou_threshold: float = 0.50,
        warmup_frames: int = 0,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.source = dict(source)
        self.arguments = dict(arguments)
        self.iou_threshold = float(iou_threshold)
        self.warmup_frames = max(0, int(warmup_frames))
        self.ground_truth = load_ground_truth(ground_truth_path) if ground_truth_path else None
        self.ground_truth_path = str(Path(ground_truth_path).expanduser().resolve()) if ground_truth_path else ""

        self._started = time.perf_counter()
        self._measurement_started = 0.0
        self._closed = False
        self._frame_count = 0
        self._detector_events = 0
        self._plate_output_frames = 0
        self._max_plate_count = 0
        self._violation_output_frames = 0
        self._violation_rows = 0
        self._plate_rows = 0
        self._recognized_rows = 0
        self._unique_plate_texts: set[str] = set()
        self._fpga_latencies: List[float] = []
        self._detector_latencies: List[float] = []
        self._person_latencies: List[float] = []
        self._frame_intervals: List[float] = []
        self._last_frame_time = 0.0
        self._annotated_frames = 0
        self._true_positive = 0
        self._false_positive = 0
        self._false_negative = 0
        self._matched_ious: List[float] = []
        self._text_match_count = 0
        self._text_reference_count = 0
        self._type_match_count = 0
        self._type_reference_count = 0

        self._frames_handle = (self.output_dir / "frames.csv").open("w", encoding="utf-8-sig", newline="")
        self._detections_handle = (self.output_dir / "detections.csv").open("w", encoding="utf-8-sig", newline="")
        self._frames_writer = csv.DictWriter(self._frames_handle, fieldnames=self.FRAME_FIELDS)
        self._detections_writer = csv.DictWriter(self._detections_handle, fieldnames=self.DETECTION_FIELDS)
        self._frames_writer.writeheader()
        self._detections_writer.writeheader()

    @staticmethod
    def _match(
        detections: Sequence[Any],
        references: Sequence[GroundTruthObject],
        threshold: float,
    ) -> Tuple[Dict[int, Tuple[int, float]], int, int, int]:
        candidates: List[Tuple[float, int, int]] = []
        for detection_index, detection in enumerate(detections):
            for reference_index, reference in enumerate(references):
                overlap = box_iou(tuple(detection.box), reference.box)
                if overlap >= threshold:
                    candidates.append((overlap, detection_index, reference_index))
        candidates.sort(reverse=True)

        matched_detections: set[int] = set()
        matched_references: set[int] = set()
        matches: Dict[int, Tuple[int, float]] = {}
        for overlap, detection_index, reference_index in candidates:
            if detection_index in matched_detections or reference_index in matched_references:
                continue
            matched_detections.add(detection_index)
            matched_references.add(reference_index)
            matches[detection_index] = (reference_index, overlap)
        true_positive = len(matches)
        return matches, true_positive, len(detections) - true_positive, len(references) - true_positive

    def record_frame(
        self,
        *,
        processed_frame: int,
        source_frame: int,
        source_time_s: float,
        source_name: str,
        real_fps: float,
        threshold: int,
        candidate_boxes: int,
        plates: Sequence[Any],
        people: Sequence[Any],
        violations: Sequence[Any],
        detector_mode: str,
        detector_busy: bool,
        fpga_busy: bool,
        fpga_ms: float,
        detector_ms: float,
        person_ms: float,
        detector_generation: int,
        detector_updated: bool,
        detector_result: Sequence[Any],
        detector_source_frame: int,
        detector_source_time_s: float,
        detector_source_name: str,
        fpga_updated: bool,
        person_updated: bool,
        active_pixels: int,
    ) -> None:
        if self._closed:
            return
        self._frame_count += 1
        frame_time = time.perf_counter()
        elapsed = frame_time - self._started
        measured = processed_frame > self.warmup_frames
        if measured and self._measurement_started <= 0.0:
            self._measurement_started = frame_time

        if measured and self._last_frame_time > 0.0:
            self._frame_intervals.append((frame_time - self._last_frame_time) * 1000.0)
        self._last_frame_time = frame_time
        if measured and detector_updated:
            self._detector_events += 1

        if measured and fpga_updated and fpga_ms > 0:
            self._fpga_latencies.append(float(fpga_ms))
        if measured and detector_updated and detector_ms > 0:
            self._detector_latencies.append(float(detector_ms))
        if measured and person_updated and person_ms > 0:
            self._person_latencies.append(float(person_ms))

        if plates:
            self._plate_output_frames += 1
        if violations:
            self._violation_output_frames += 1
            self._violation_rows += len(violations)
        self._max_plate_count = max(self._max_plate_count, len(plates))

        evaluation_plates = list(detector_result) if detector_updated else []
        evaluation_frame = detector_source_frame if detector_updated else source_frame
        evaluation_time = detector_source_time_s if detector_updated else source_time_s
        evaluation_name = detector_source_name if detector_updated else source_name
        has_annotation = False
        references: List[GroundTruthObject] = []
        matches: Dict[int, Tuple[int, float]] = {}
        true_positive = false_positive = false_negative = 0
        if measured and detector_updated and self.ground_truth is not None:
            has_annotation, references = self.ground_truth.lookup(evaluation_frame, evaluation_name)
            if has_annotation:
                matches, true_positive, false_positive, false_negative = self._match(
                    evaluation_plates, references, self.iou_threshold
                )
                self._annotated_frames += 1
                self._true_positive += true_positive
                self._false_positive += false_positive
                self._false_negative += false_negative

        self._frames_writer.writerow(
            {
                "processed_frame": processed_frame,
                "source_frame": source_frame,
                "source_time_s": f"{source_time_s:.6f}",
                "source_name": source_name,
                "elapsed_s": f"{elapsed:.6f}",
                "real_fps": f"{real_fps:.4f}",
                "threshold": threshold,
                "candidate_boxes": candidate_boxes,
                "plate_count": len(plates),
                "person_count": len(people),
                "violation_count": len(violations),
                "detector_mode": detector_mode,
                "detector_busy": int(detector_busy),
                "fpga_busy": int(fpga_busy),
                "fpga_ms": f"{fpga_ms:.4f}",
                "detector_ms": f"{detector_ms:.4f}",
                "person_ms": f"{person_ms:.4f}",
                "detector_generation": detector_generation,
                "detector_updated": int(detector_updated),
                "detector_source_frame": detector_source_frame if detector_updated else "",
                "detector_source_time_s": f"{detector_source_time_s:.6f}" if detector_updated else "",
                "active_pixels": active_pixels,
                "ground_truth_count": len(references) if has_annotation else "",
                "true_positive": true_positive if has_annotation else "",
                "false_positive": false_positive if has_annotation else "",
                "false_negative": false_negative if has_annotation else "",
            }
        )

        detection_groups = (
            ("plate", evaluation_plates),
            ("person", people if person_updated else []),
            ("violation", violations),
        )
        for kind, detections in detection_groups:
            for detection_index, detection in enumerate(detections):
                matched_index = ""
                overlap: float | str = ""
                text_match: int | str = ""
                type_match: int | str = ""
                if kind == "plate" and detection_index in matches:
                    reference_index, overlap_value = matches[detection_index]
                    reference = references[reference_index]
                    matched_index = reference_index
                    overlap = overlap_value
                    self._matched_ious.append(overlap_value)
                    if reference.text:
                        detected_text = getattr(detection, "raw_label", "") or getattr(detection, "label", "")
                        text_match = int(_normalize_plate_text(detected_text) == _normalize_plate_text(reference.text))
                        self._text_reference_count += 1
                        self._text_match_count += int(text_match)
                    if reference.type_name:
                        detected_type = getattr(detection, "type_name", "")
                        type_match = int(_normalize_plate_type(detected_type) == _normalize_plate_type(reference.type_name))
                        self._type_reference_count += 1
                        self._type_match_count += int(type_match)

                if kind == "plate" and measured:
                    self._plate_rows += 1
                    raw_text = getattr(detection, "raw_label", "")
                    if _is_recognized_text(raw_text):
                        self._recognized_rows += 1
                        self._unique_plate_texts.add(raw_text)

                x, y, width, height = detection.box
                row_source_frame = evaluation_frame if kind == "plate" else source_frame
                row_source_time = evaluation_time if kind == "plate" else source_time_s
                row_source_name = evaluation_name if kind == "plate" else source_name
                self._detections_writer.writerow(
                    {
                        "processed_frame": processed_frame,
                        "source_frame": row_source_frame,
                        "source_time_s": f"{row_source_time:.6f}",
                        "source_name": row_source_name,
                        "kind": kind,
                        "detection_index": detection_index,
                        "label": getattr(detection, "label", ""),
                        "raw_label": getattr(detection, "raw_label", ""),
                        "type_name": getattr(detection, "type_name", ""),
                        "full_text": getattr(detection, "full_text", ""),
                        "score": f"{float(getattr(detection, 'score', 0.0)):.6f}",
                        "x": x,
                        "y": y,
                        "w": width,
                        "h": height,
                        "matched_ground_truth": matched_index,
                        "iou": f"{overlap:.6f}" if isinstance(overlap, float) else "",
                        "text_match": text_match,
                        "type_match": type_match,
                    }
                )

    def close(self) -> Dict[str, Any]:
        if self._closed:
            summary_path = self.output_dir / "summary.json"
            if summary_path.is_file():
                with summary_path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)
            return {}

        elapsed = time.perf_counter() - self._started
        measured_frames = max(self._frame_count - self.warmup_frames, 0)
        measured_elapsed = (
            time.perf_counter() - self._measurement_started
            if self._measurement_started > 0.0
            else 0.0
        )
        precision = self._true_positive / max(self._true_positive + self._false_positive, 1)
        recall = self._true_positive / max(self._true_positive + self._false_negative, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        summary: Dict[str, Any] = {
            "source": self.source,
            "runtime": {
                "processed_frames": self._frame_count,
                "warmup_frames": self.warmup_frames,
                "elapsed_seconds": elapsed,
                "throughput_fps": self._frame_count / max(elapsed, 1e-9),
                "measured_frames": measured_frames,
                "measured_elapsed_seconds": measured_elapsed,
                "measured_throughput_fps": max(measured_frames - 1, 0) / max(measured_elapsed, 1e-9),
                "detector_result_events": self._detector_events,
                "detector_frame_coverage": self._detector_events / max(
                    measured_frames, 1
                ),
            },
            "outputs": {
                "plate_output_frames": self._plate_output_frames,
                "plate_output_frame_rate": self._plate_output_frames / max(self._frame_count, 1),
                "max_simultaneous_plates": self._max_plate_count,
                "plate_detection_rows": self._plate_rows,
                "ocr_text_rows": self._recognized_rows,
                "ocr_output_rate": self._recognized_rows / max(self._plate_rows, 1),
                "unique_plate_texts": sorted(self._unique_plate_texts),
                "pedestrian_violation_output_frames": self._violation_output_frames,
                "pedestrian_violation_rows": self._violation_rows,
            },
            "latency": {
                "pipeline_frame_interval": _latency_summary(self._frame_intervals),
                "fpga": _latency_summary(self._fpga_latencies),
                "plate_detector": _latency_summary(self._detector_latencies),
                "person_detector": _latency_summary(self._person_latencies),
            },
            "ground_truth": {
                "available": self.ground_truth is not None,
                "path": self.ground_truth_path,
                "iou_threshold": self.iou_threshold,
                "annotated_frames_evaluated": self._annotated_frames,
                "annotated_frames_total": (
                    self.ground_truth.annotation_count if self.ground_truth is not None else 0
                ),
            },
            "arguments": self.arguments,
        }
        if self.ground_truth is not None:
            summary["accuracy"] = {
                "true_positive": self._true_positive,
                "false_positive": self._false_positive,
                "false_negative": self._false_negative,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_matched_iou": (
                    sum(self._matched_ious) / len(self._matched_ious)
                    if self._matched_ious
                    else None
                ),
                "ocr_exact_accuracy": (
                    self._text_match_count / self._text_reference_count
                    if self._text_reference_count
                    else None
                ),
                "ocr_reference_count": self._text_reference_count,
                "type_exact_accuracy": (
                    self._type_match_count / self._type_reference_count
                    if self._type_reference_count
                    else None
                ),
                "type_reference_count": self._type_reference_count,
            }

        with (self.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        self._frames_handle.close()
        self._detections_handle.close()
        self._closed = True
        return summary
