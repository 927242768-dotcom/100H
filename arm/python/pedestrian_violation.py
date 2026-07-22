from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from detector import Detection


Point = Tuple[float, float]
PixelPoint = Tuple[int, int]
Box = Tuple[int, int, int, int]


DEFAULT_RESTRICTED_ZONE = "0.05,0.55;0.95,0.55;1.0,1.0;0.0,1.0"


def parse_zone_points(value: str) -> List[Point]:
    text = value.strip() or DEFAULT_RESTRICTED_ZONE
    points: List[Point] = []
    for item in text.split(";"):
        parts = [part.strip() for part in item.split(",")]
        if len(parts) != 2:
            raise ValueError(f"违法区域顶点格式错误: {item!r}")
        points.append((float(parts[0]), float(parts[1])))
    if len(points) < 3:
        raise ValueError("行人违法区域至少需要3个顶点")
    return points


def scale_zone_points(
    points: Sequence[Point],
    width: int,
    height: int,
    *,
    normalized: bool,
) -> List[PixelPoint]:
    max_x = max(width - 1, 0)
    max_y = max(height - 1, 0)
    scaled: List[PixelPoint] = []
    for x, y in points:
        px = x * max_x if normalized else x
        py = y * max_y if normalized else y
        scaled.append(
            (
                max(0, min(max_x, int(round(px)))),
                max(0, min(max_y, int(round(py)))),
            )
        )
    return scaled


def point_in_polygon(point: Point, polygon: Sequence[PixelPoint]) -> bool:
    if len(polygon) < 3:
        return False
    px, py = point
    inside = False
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        cross = (current_y > py) != (previous_y > py)
        if cross:
            edge_x = (previous_x - current_x) * (py - current_y) / (
                previous_y - current_y
            ) + current_x
            if px <= edge_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def box_iou(box_a: Box, box_b: Box) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, bx + bh)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = max(aw, 0) * max(ah, 0) + max(bw, 0) * max(bh, 0) - intersection
    return intersection / union if union > 0 else 0.0


def _center_distance(box_a: Box, box_b: Box) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    return math.hypot(
        ax + aw * 0.5 - bx - bw * 0.5,
        ay + ah * 0.5 - by - bh * 0.5,
    )


def _boxes_match(box_a: Box, box_b: Box, minimum_iou: float) -> bool:
    if box_iou(box_a, box_b) >= minimum_iou:
        return True
    distance = _center_distance(box_a, box_b)
    scale = max(box_a[2], box_a[3], box_b[2], box_b[3], 1)
    return distance <= max(24.0, scale * 0.75)


@dataclass
class _ViolationTrack:
    track_id: int
    detection: Detection
    inside_hits: int
    outside_hits: int
    misses: int
    violation: bool
    last_seen_time: float


class PedestrianViolationMonitor:
    """通过行人脚点进入禁入区并连续确认，标记进入机动车道行为。"""

    def __init__(
        self,
        zone_points: Sequence[Point],
        *,
        normalized: bool = True,
        confirmation_hits: int = 2,
        clear_hits: int = 2,
        match_iou: float = 0.15,
        hold_seconds: float = 0.75,
        foot_y_ratio: float = 0.95,
    ) -> None:
        if len(zone_points) < 3:
            raise ValueError("行人违法区域至少需要3个顶点")
        self._zone_spec = list(zone_points)
        self._normalized = bool(normalized)
        self._confirmation_hits = max(1, int(confirmation_hits))
        self._clear_hits = max(1, int(clear_hits))
        self._match_iou = max(0.0, min(1.0, float(match_iou)))
        self._hold_seconds = max(0.0, float(hold_seconds))
        self._foot_y_ratio = max(0.5, min(1.0, float(foot_y_ratio)))
        self._tracks: Dict[int, _ViolationTrack] = {}
        self._next_track_id = 1
        self._last_generation = -1
        self._zone_pixels: List[PixelPoint] = []
        self._frame_size = (0, 0)

    @property
    def zone_pixels(self) -> List[PixelPoint]:
        return list(self._zone_pixels)

    def _update_zone(self, frame_width: int, frame_height: int) -> None:
        frame_size = (frame_width, frame_height)
        if frame_size == self._frame_size:
            return
        self._frame_size = frame_size
        self._zone_pixels = scale_zone_points(
            self._zone_spec,
            frame_width,
            frame_height,
            normalized=self._normalized,
        )

    def _foot_inside(self, detection: Detection) -> bool:
        x, y, width, height = detection.box
        foot = (x + width * 0.5, y + height * self._foot_y_ratio)
        return point_in_polygon(foot, self._zone_pixels)

    def _find_track(self, detection: Detection, used: set[int]) -> int | None:
        candidates = []
        for track_id, track in self._tracks.items():
            if track_id in used or not _boxes_match(
                detection.box, track.detection.box, self._match_iou
            ):
                continue
            overlap = box_iou(detection.box, track.detection.box)
            distance = _center_distance(detection.box, track.detection.box)
            candidates.append((overlap, -distance, track_id))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][2]

    def update(
        self,
        detections: Sequence[Detection],
        generation: int,
        frame_width: int,
        frame_height: int,
        *,
        now: float | None = None,
    ) -> None:
        self._update_zone(frame_width, frame_height)
        if generation <= 0 or generation == self._last_generation:
            return
        self._last_generation = generation
        current_time = time.monotonic() if now is None else now
        used_tracks: set[int] = set()

        for detection in detections:
            track_id = self._find_track(detection, used_tracks)
            inside = self._foot_inside(detection)
            if track_id is None:
                track_id = self._next_track_id
                self._next_track_id += 1
                inside_hits = 1 if inside else 0
                self._tracks[track_id] = _ViolationTrack(
                    track_id=track_id,
                    detection=detection,
                    inside_hits=inside_hits,
                    outside_hits=0 if inside else 1,
                    misses=0,
                    violation=inside_hits >= self._confirmation_hits,
                    last_seen_time=current_time,
                )
            else:
                track = self._tracks[track_id]
                if inside:
                    track.inside_hits += 1
                    track.outside_hits = 0
                    if track.inside_hits >= self._confirmation_hits:
                        track.violation = True
                else:
                    track.inside_hits = 0
                    track.outside_hits += 1
                    if track.outside_hits >= self._clear_hits:
                        track.violation = False
                track.detection = detection
                track.misses = 0
                track.last_seen_time = current_time
            used_tracks.add(track_id)

        expired: List[int] = []
        for track_id, track in self._tracks.items():
            if track_id in used_tracks:
                continue
            track.misses += 1
            if current_time - track.last_seen_time > self._hold_seconds:
                expired.append(track_id)
        for track_id in expired:
            del self._tracks[track_id]

    def annotate(
        self,
        detections: Sequence[Detection],
        *,
        now: float | None = None,
    ) -> Tuple[List[Detection], List[Detection]]:
        current_time = time.monotonic() if now is None else now
        annotated: List[Detection] = []
        violations: List[Detection] = []
        used_tracks: set[int] = set()
        for detection in detections:
            track_id = self._find_track(detection, used_tracks)
            track = self._tracks.get(track_id) if track_id is not None else None
            active = (
                track is not None
                and track.violation
                and current_time - track.last_seen_time <= self._hold_seconds
            )
            if active:
                marked = Detection(
                    label=detection.label,
                    score=detection.score,
                    box=detection.box,
                    raw_label=detection.raw_label,
                    type_name="行人违法",
                    full_text="违法行为：行人进入机动车道",
                )
                annotated.append(marked)
                violations.append(marked)
                used_tracks.add(track.track_id)
            else:
                annotated.append(detection)
        return annotated, violations
