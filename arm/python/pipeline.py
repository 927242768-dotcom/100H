from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None

from detector import (
    BaseDetector,
    Detection,
    HyperLprDetector,
    MockDetector,
    PersonRknnDetector,
    RknnLiteDetector,
)
from fpga_client import (
    CURRENT_SAFE_FRAME_BYTES,
    DEFAULT_SAFE_HEIGHT,
    DEFAULT_SAFE_WIDTH,
    FpgaPreprocessClient,
    PREPROC_SOBEL,
)
from evaluation_metrics import MetricsRecorder


_UNICODE_TEXT_CACHE: Dict[Tuple[int, str, Tuple[int, int, int], Tuple[int, int, int], int], Tuple[np.ndarray, int, int]] = {}


def build_candidate_boxes(mask: np.ndarray, min_area: int = 40) -> List[Tuple[int, int, int, int]]:
    _, binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: List[Tuple[int, int, int, int]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, w, h))

    boxes.sort(key=lambda item: item[2] * item[3], reverse=True)
    return boxes[:24]


def filter_candidate_boxes(
    boxes: List[Tuple[int, int, int, int]],
    width: int,
    height: int,
    min_area: int,
    max_area_ratio: float,
    reject_border: bool,
) -> List[Tuple[int, int, int, int]]:
    total_area = max(width * height, 1)
    filtered: List[Tuple[int, int, int, int]] = []

    for x, y, w, h in boxes:
        area = w * h
        if area < min_area:
            continue
        if max_area_ratio > 0 and area > int(total_area * max_area_ratio):
            continue
        if reject_border and (x == 0 or y == 0 or (x + w) >= width or (y + h) >= height):
            continue
        filtered.append((x, y, w, h))

    return filtered


def refine_mask(
    mask: np.ndarray,
    *,
    cleanup_mode: str,
    kernel_size: int,
    min_area: int,
    max_area_ratio: float,
    reject_border: bool,
) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
    _, binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        if cleanup_mode == "open":
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        elif cleanup_mode == "close":
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        elif cleanup_mode == "open_close":
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    boxes = build_candidate_boxes(binary, min_area=max(min_area, 1))
    boxes = filter_candidate_boxes(
        boxes,
        width=binary.shape[1],
        height=binary.shape[0],
        min_area=max(min_area, 1),
        max_area_ratio=max_area_ratio,
        reject_border=reject_border,
    )

    cleaned = np.zeros_like(binary)
    for x, y, w, h in boxes:
        region = binary[y:y + h, x:x + w]
        cleaned[y:y + h, x:x + w] = np.maximum(cleaned[y:y + h, x:x + w], region)

    return cleaned, boxes


def scale_box(box: Tuple[int, int, int, int], sx: float, sy: float) -> Tuple[int, int, int, int]:
    x, y, w, h = box
    return (
        int(x * sx),
        int(y * sy),
        max(1, int(w * sx)),
        max(1, int(h * sy)),
    )


def draw_mask_overlay(frame: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_bgr[:, :, 0] = 0
    mask_bgr[:, :, 1] = (mask_bgr[:, :, 1] * 3) // 5
    mask_bgr[:, :, 2] = mask
    return cv2.addWeighted(frame, 1.0 - alpha, mask_bgr, alpha, 0.0)


def draw_mask_contours(
    display: np.ndarray,
    mask: np.ndarray,
    *,
    color: Tuple[int, int, int] = (0, 64, 255),
    thickness: int = 2,
) -> None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(display, contours, -1, color, thickness, cv2.LINE_AA)


def draw_text_with_outline(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    *,
    font_scale: float,
    color: Tuple[int, int, int],
    thickness: int,
    outline_color: Tuple[int, int, int] = (0, 0, 0),
    outline_thickness: int = 4,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        outline_color,
        outline_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def resolve_text_font(font_path: str, font_size: int):
    if ImageFont is None:
        return None

    candidates = []
    if font_path:
        candidates.append(font_path)
    candidates.extend(
        [
            "/userdata/HyperLPR/HyperLPR/Prj-Linux/hyperlpr3/resource/font/platech.ttf",
            "/userdata/HyperLPR/HyperLPR/build/linux/install/hyperlpr3/resource/font/platech.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/arphic/ukai.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )

    for candidate in candidates:
        if not candidate:
            continue
        if not os.path.exists(candidate):
            continue
        try:
            return ImageFont.truetype(candidate, font_size, encoding="utf-8")
        except Exception:
            continue
    return None


def draw_unicode_texts(image: np.ndarray, items, font) -> None:
    if not items:
        return

    if font is None or Image is None or ImageDraw is None:
        for item in items:
            fallback = item["text"].encode("ascii", errors="ignore").decode("ascii") or "PLATE"
            draw_text_with_outline(
                image,
                fallback,
                item["origin"],
                font_scale=item.get("font_scale", 0.60),
                color=item.get("color", (0, 255, 0)),
                thickness=item.get("thickness", 2),
                outline_color=item.get("outline_color", (0, 0, 0)),
                outline_thickness=item.get("outline_thickness", 4),
            )
        return

    for item in items:
        x, y = item["origin"]
        text = item["text"]
        bgr_color = tuple(int(c) for c in item.get("color", (0, 255, 0)))
        bgr_outline = tuple(int(c) for c in item.get("outline_color", (0, 0, 0)))
        outline_thickness = int(item.get("outline_thickness", 2))
        cache_key = (id(font), text, bgr_color, bgr_outline, outline_thickness)

        cached = _UNICODE_TEXT_CACHE.get(cache_key)
        if cached is None:
            color = tuple(int(c) for c in reversed(bgr_color))
            outline_color = tuple(int(c) for c in reversed(bgr_outline))
            try:
                left, top, right, bottom = font.getbbox(text)
            except Exception:
                left, top, right, bottom = (0, 0, max(len(text), 1) * 16, 32)

            pad = max(2, outline_thickness + 2)
            width = max(1, right - left + pad * 2)
            height = max(1, bottom - top + pad * 2)
            sprite = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(sprite)
            base_x = pad - left
            base_y = pad - top

            for dx in range(-outline_thickness, outline_thickness + 1):
                for dy in range(-outline_thickness, outline_thickness + 1):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((base_x + dx, base_y + dy), text, font=font, fill=outline_color)
            draw.text((base_x, base_y), text, font=font, fill=color)

            rgba = np.array(sprite, dtype=np.uint8)
            bgra = rgba[:, :, [2, 1, 0, 3]]
            cached = (bgra, left - pad, top - pad)
            _UNICODE_TEXT_CACHE[cache_key] = cached

        sprite_bgra, offset_x, offset_y = cached
        x1 = x + offset_x
        y1 = y + offset_y
        x2 = x1 + sprite_bgra.shape[1]
        y2 = y1 + sprite_bgra.shape[0]
        if x2 <= x1 or y2 <= y1:
            continue

        dst_x1 = max(0, x1)
        dst_y1 = max(0, y1)
        dst_x2 = min(image.shape[1], x2)
        dst_y2 = min(image.shape[0], y2)
        if dst_x2 <= dst_x1 or dst_y2 <= dst_y1:
            continue

        src_x1 = dst_x1 - x1
        src_y1 = dst_y1 - y1
        src_x2 = src_x1 + (dst_x2 - dst_x1)
        src_y2 = src_y1 + (dst_y2 - dst_y1)

        roi = image[dst_y1:dst_y2, dst_x1:dst_x2]
        sprite_roi = sprite_bgra[src_y1:src_y2, src_x1:src_x2]
        alpha = sprite_roi[:, :, 3:4].astype(np.float32) / 255.0
        if not np.any(alpha):
            continue
        roi[:] = (sprite_roi[:, :, :3].astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)


def build_plate_summary(detections: List[Detection]) -> str:
    if not detections:
        return ""

    texts = [det.full_text or det.raw_label or det.label for det in detections]
    return " | ".join(texts)


def build_plate_summary_lines(detections: List[Detection], max_chars_per_line: int) -> List[str]:
    if not detections:
        return []

    texts = [det.full_text or det.raw_label or det.label for det in detections]
    lines: List[str] = []
    current = ""

    for text in texts:
        if not current:
            current = text
            continue

        candidate = f"{current} | {text}"
        if len(candidate) <= max_chars_per_line:
            current = candidate
            continue

        lines.append(current)
        current = text

    if current:
        lines.append(current)

    return lines


def draw_candidate_boxes(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    *,
    label: str = "ROI",
    color: Tuple[int, int, int] = (0, 220, 255),
) -> None:
    for x, y, w, h in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        draw_text_with_outline(
            image,
            label,
            (x, max(24, y - 8)),
            font_scale=0.60,
            color=color,
            thickness=2,
            outline_thickness=4,
        )


def ensure_output_dir(path: str) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_debug_images(output_dir: Path, frame_index: int, frame: np.ndarray, mask: np.ndarray, display: np.ndarray) -> None:
    cv2.imwrite(str(output_dir / f"frame_{frame_index:05d}.png"), frame)
    cv2.imwrite(str(output_dir / f"mask_{frame_index:05d}.png"), mask)
    cv2.imwrite(str(output_dir / f"display_{frame_index:05d}.png"), display)


def compose_display_frame(
    frame: np.ndarray,
    mask: np.ndarray | None,
    mode: str,
    alpha: float,
) -> np.ndarray:
    if mode == "camera":
        return frame.copy()
    if mode == "outline":
        return frame.copy()
    if mask is None:
        raise ValueError(f"显示模式 {mode} 需要 FPGA 掩码")
    if mode == "mask":
        return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return draw_mask_overlay(frame, mask, alpha)


def resize_for_display(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if width <= 0 and height <= 0:
        return image
    if width > 0 and height > 0:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    src_h, src_w = image.shape[:2]
    if width > 0:
        height = max(1, int(src_h * (width / max(src_w, 1))))
    else:
        width = max(1, int(src_w * (height / max(src_h, 1))))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def expand_box(
    box: Tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    expand_ratio: float,
) -> Tuple[int, int, int, int]:
    x, y, w, h = box
    if expand_ratio <= 0:
        return x, y, w, h

    extra_w = int(w * expand_ratio)
    extra_h = int(h * expand_ratio)
    x1 = max(0, x - extra_w)
    y1 = max(0, y - extra_h)
    x2 = min(image_width, x + w + extra_w)
    y2 = min(image_height, y + h + extra_h)
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def resize_detector_input(
    image: np.ndarray,
    target_width: int,
    target_height: int,
) -> Tuple[np.ndarray, float, float]:
    src_h, src_w = image.shape[:2]
    if target_width <= 0 and target_height <= 0:
        return image, 1.0, 1.0

    if target_width > 0 and target_height > 0:
        resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        return resized, src_w / max(target_width, 1), src_h / max(target_height, 1)

    if target_width > 0:
        target_height = max(1, int(src_h * (target_width / max(src_w, 1))))
    else:
        target_width = max(1, int(src_w * (target_height / max(src_h, 1))))

    resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return resized, src_w / max(target_width, 1), src_h / max(target_height, 1)


def enhance_plate_frame(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        return image

    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y = clahe.apply(y)
    merged = cv2.merge((y, cr, cb))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
    return cv2.addWeighted(enhanced, 1.15, blurred, -0.15, 0)


def build_detection_tiles(
    image_width: int,
    image_height: int,
    overlap_ratio: float = 0.25,
) -> List[Tuple[int, int, int, int]]:
    if image_width <= 1 or image_height <= 1:
        return []

    tile_w = min(image_width, max(1, int(image_width * 0.60)))
    tile_h = min(image_height, max(1, int(image_height * 0.60)))
    overlap_x = int(tile_w * overlap_ratio)
    overlap_y = int(tile_h * overlap_ratio)

    left_x = 0
    right_x = max(0, image_width - tile_w)
    center_x = max(0, (image_width - tile_w) // 2)
    left_y = 0
    right_y = max(0, image_height - tile_h)
    center_y = max(0, (image_height - tile_h) // 2)

    xs = [left_x, max(0, center_x - overlap_x // 2), right_x]
    ys = [left_y, max(0, center_y - overlap_y // 2), right_y]
    tiles: List[Tuple[int, int, int, int]] = []

    for y in ys:
        for x in xs:
            tile = (x, y, tile_w, tile_h)
            if tile not in tiles:
                tiles.append(tile)

    return tiles


def scale_detection(det: Detection, sx: float, sy: float) -> Detection:
    x, y, w, h = det.box
    return Detection(
        label=det.label,
        raw_label=det.raw_label,
        type_name=det.type_name,
        full_text=det.full_text,
        score=det.score,
        box=(
            int(round(x * sx)),
            int(round(y * sy)),
            max(1, int(round(w * sx))),
            max(1, int(round(h * sy))),
        ),
    )


def box_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(1, aw * ah)
    area_b = max(1, bw * bh)
    return inter_area / float(area_a + area_b - inter_area)


def merge_detections(
    base: List[Detection],
    extra: List[Detection],
    *,
    iou_threshold: float = 0.55,
) -> List[Detection]:
    merged: List[Detection] = list(base)

    for candidate in extra:
        replaced = False
        for index, existing in enumerate(merged):
            same_text = bool(candidate.raw_label) and candidate.raw_label == existing.raw_label
            overlap = box_iou(candidate.box, existing.box) >= iou_threshold
            if not same_text and not overlap:
                continue

            if candidate.score > existing.score:
                merged[index] = candidate
            replaced = True
            break

        if not replaced:
            merged.append(candidate)

    merged.sort(key=lambda det: det.score, reverse=True)
    return merged


def merge_candidate_boxes(
    primary: List[Tuple[int, int, int, int]],
    extra: List[Tuple[int, int, int, int]],
    *,
    max_boxes: int,
    iou_threshold: float = 0.60,
) -> List[Tuple[int, int, int, int]]:
    merged: List[Tuple[int, int, int, int]] = list(primary)

    for box in extra:
        duplicated = False
        for existing in merged:
            if box_iou(box, existing) >= iou_threshold:
                duplicated = True
                break
        if duplicated:
            continue
        merged.append(box)
        if max_boxes > 0 and len(merged) >= max_boxes:
            break

    return merged[:max_boxes] if max_boxes > 0 else merged


def stabilize_detections_with_history(
    previous: List[Detection],
    current: List[Detection],
    *,
    iou_threshold: float = 0.45,
) -> List[Detection]:
    if not previous or not current:
        return current

    stabilized: List[Detection] = []
    for det in current:
        chosen = det
        for prev in previous:
            if box_iou(det.box, prev.box) < iou_threshold:
                continue

            if not det.raw_label and prev.raw_label:
                chosen = Detection(
                    label=prev.label,
                    raw_label=prev.raw_label,
                    type_name=prev.type_name,
                    full_text=prev.full_text,
                    score=max(det.score, prev.score),
                    box=det.box,
                )
                break

            prev_text = prev.raw_label or prev.label
            curr_text = det.raw_label or det.label
            similar_text = prev_text.startswith(curr_text) or curr_text.startswith(prev_text)
            if similar_text and len(prev_text) > len(curr_text):
                chosen = Detection(
                    label=prev.label,
                    raw_label=prev.raw_label,
                    type_name=prev.type_name,
                    full_text=prev.full_text,
                    score=max(det.score, prev.score),
                    box=det.box,
                )
            break
        stabilized.append(chosen)

    stabilized.sort(key=lambda item: item.score, reverse=True)
    return stabilized


class BoxMotionTracker:
    """在两次检测结果之间用稀疏光流更新框位置。"""

    def __init__(self, tracking_width: int = 640) -> None:
        self._tracking_width = max(160, int(tracking_width))
        self._previous_gray: np.ndarray | None = None
        self._detections: List[Detection] = []
        self._generation = -1

    @staticmethod
    def _copy_with_box(det: Detection, box: Tuple[int, int, int, int]) -> Detection:
        return Detection(
            label=det.label,
            raw_label=det.raw_label,
            type_name=det.type_name,
            full_text=det.full_text,
            score=det.score,
            box=box,
        )

    @staticmethod
    def _targets_match(det_a: Detection, det_b: Detection) -> bool:
        if box_iou(det_a.box, det_b.box) >= 0.10:
            return True
        ax, ay, aw, ah = det_a.box
        bx, by, bw, bh = det_b.box
        center_distance = (
            (ax + aw * 0.5 - bx - bw * 0.5) ** 2
            + (ay + ah * 0.5 - by - bh * 0.5) ** 2
        ) ** 0.5
        return center_distance <= max(24.0, max(aw, ah, bw, bh) * 1.25)

    def _resize_gray(self, gray: np.ndarray) -> Tuple[np.ndarray, float]:
        height, width = gray.shape[:2]
        if width <= self._tracking_width:
            return gray, 1.0
        scale = self._tracking_width / float(width)
        resized = cv2.resize(
            gray,
            (self._tracking_width, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale

    def _track_between(
        self,
        source_gray: np.ndarray,
        current_gray: np.ndarray,
        detections: List[Detection],
        scale: float,
        *,
        max_scale_change: float,
        verify_round_trip: bool,
    ) -> Tuple[List[Detection], List[bool]]:
        if not detections or source_gray.shape != current_gray.shape:
            return list(detections), [False] * len(detections)

        all_points = []
        point_owners: List[int] = []
        height, width = source_gray.shape[:2]
        for index, det in enumerate(detections):
            x, y, box_width, box_height = det.box
            x1 = max(0, min(width - 1, int(round(x * scale))))
            y1 = max(0, min(height - 1, int(round(y * scale))))
            x2 = max(x1 + 1, min(width, int(round((x + box_width) * scale))))
            y2 = max(y1 + 1, min(height, int(round((y + box_height) * scale))))
            roi = source_gray[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            points = cv2.goodFeaturesToTrack(
                roi,
                maxCorners=16,
                qualityLevel=0.02,
                minDistance=3,
                blockSize=3,
            )
            if points is None:
                continue
            points[:, 0, 0] += x1
            points[:, 0, 1] += y1
            all_points.append(points)
            point_owners.extend([index] * len(points))

        if not all_points:
            return list(detections), [False] * len(detections)

        previous_points = np.concatenate(all_points, axis=0).astype(np.float32, copy=False)
        current_points, status, _ = cv2.calcOpticalFlowPyrLK(
            source_gray,
            current_gray,
            previous_points,
            None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03),
        )
        if current_points is None or status is None:
            return list(detections), [False] * len(detections)

        moved: List[Detection] = []
        succeeded: List[bool] = []
        frame_width = max(1, int(round(width / max(scale, 1e-6))))
        frame_height = max(1, int(round(height / max(scale, 1e-6))))
        previous_flat = previous_points.reshape(-1, 2)
        current_flat = current_points.reshape(-1, 2)
        status_flat = status.reshape(-1).astype(bool)
        if verify_round_trip:
            backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                current_gray,
                source_gray,
                current_points,
                None,
                winSize=(15, 15),
                maxLevel=2,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03),
            )
            if backward_points is None or backward_status is None:
                return list(detections), [False] * len(detections)
            backward_flat = backward_points.reshape(-1, 2)
            round_trip_error = np.linalg.norm(backward_flat - previous_flat, axis=1)
            status_flat &= backward_status.reshape(-1).astype(bool)
            status_flat &= round_trip_error <= 1.5
        displacement = current_flat - previous_flat
        owners = np.asarray(point_owners)
        for index, det in enumerate(detections):
            owner_mask = owners == index
            valid = owner_mask & status_flat
            if int(np.count_nonzero(valid)) < 2:
                moved.append(det)
                succeeded.append(False)
                continue

            local_displacement = displacement[valid]
            median_displacement = np.median(local_displacement, axis=0)
            residual = np.linalg.norm(local_displacement - median_displacement, axis=1)
            median_residual = float(np.median(residual))
            inliers = residual <= max(1.5, median_residual * 3.0)
            if int(np.count_nonzero(inliers)) < 2:
                moved.append(det)
                succeeded.append(False)
                continue

            source_inliers = previous_flat[valid][inliers]
            current_inliers = current_flat[valid][inliers]
            source_center = np.median(source_inliers, axis=0)
            current_center = np.median(current_inliers, axis=0)
            dx, dy = (current_center - source_center) / max(scale, 1e-6)
            if abs(float(dx)) > det.box[2] * 1.5 or abs(float(dy)) > det.box[3] * 2.0:
                moved.append(det)
                succeeded.append(False)
                continue

            box_scale = 1.0
            if len(source_inliers) >= 4:
                source_radius = float(np.median(np.linalg.norm(source_inliers - source_center, axis=1)))
                current_radius = float(np.median(np.linalg.norm(current_inliers - current_center, axis=1)))
                if source_radius >= 2.0:
                    raw_scale = current_radius / source_radius
                    box_scale = float(
                        np.clip(
                            raw_scale,
                            1.0 - max_scale_change,
                            1.0 + max_scale_change,
                        )
                    )

            x, y, box_width, box_height = det.box
            next_width = max(1, min(frame_width, int(round(box_width * box_scale))))
            next_height = max(1, min(frame_height, int(round(box_height * box_scale))))
            center_x = x + box_width * 0.5 + float(dx)
            center_y = y + box_height * 0.5 + float(dy)
            next_x = max(0, min(frame_width - next_width, int(round(center_x - next_width * 0.5))))
            next_y = max(0, min(frame_height - next_height, int(round(center_y - next_height * 0.5))))
            moved.append(
                self._copy_with_box(
                    det,
                    (next_x, next_y, next_width, next_height),
                )
            )
            succeeded.append(True)
        return moved, succeeded

    def _advance(self, current_gray: np.ndarray, scale: float) -> List[Detection]:
        if self._previous_gray is None or not self._detections:
            return list(self._detections)
        moved, _ = self._track_between(
            self._previous_gray,
            current_gray,
            self._detections,
            scale,
            max_scale_change=0.08,
            verify_round_trip=False,
        )
        return moved

    def update(
        self,
        gray: np.ndarray,
        detections: List[Detection],
        generation: int,
        *,
        detection_gray: np.ndarray | None = None,
    ) -> List[Detection]:
        current_gray, scale = self._resize_gray(gray)
        moved = self._advance(current_gray, scale)

        if generation != self._generation:
            compensated = list(detections)
            compensated_ok = [False] * len(detections)
            if detection_gray is not None and detections:
                source_gray, source_scale = self._resize_gray(detection_gray)
                if abs(source_scale - scale) <= 1e-6:
                    compensated, compensated_ok = self._track_between(
                        source_gray,
                        current_gray,
                        detections,
                        scale,
                        max_scale_change=0.30,
                        verify_round_trip=True,
                    )

            reconciled: List[Detection] = []
            used = set()
            for detection_index, current in enumerate(compensated):
                if compensated_ok[detection_index]:
                    reconciled.append(current)
                    continue
                match_index = next(
                    (
                        index
                        for index, tracked in enumerate(moved)
                        if index not in used and self._targets_match(current, tracked)
                    ),
                    None,
                )
                if match_index is None:
                    reconciled.append(current)
                    continue
                used.add(match_index)
                reconciled.append(self._copy_with_box(current, moved[match_index].box))
            self._detections = reconciled
            self._generation = generation
        else:
            self._detections = moved

        self._previous_gray = current_gray
        return list(self._detections)


def prepare_window(name: str, fullscreen: bool, display_width: int, display_height: int) -> None:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    elif display_width > 0 or display_height > 0:
        cv2.resizeWindow(name, max(display_width, 640), max(display_height, 480))


def _open_video_capture(
    camera_index: int,
    width: int,
    height: int,
    backend: str,
    fps: float = 0.0,
    fourcc: str = "",
) -> cv2.VideoCapture:
    if backend == "v4l2":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    elif backend == "gstreamer":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(camera_index)

    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc.upper()))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, fps)

    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    return cap


class LatestFrameCamera:
    """持续清空摄像头缓冲区，消费端始终拿到最新一帧。"""

    def __init__(
        self,
        camera_index: int,
        width: int,
        height: int,
        backend: str,
        fps: float = 0.0,
        fourcc: str = "",
    ) -> None:
        self._cap = _open_video_capture(camera_index, width, height, backend, fps, fourcc)
        self._properties = {
            cv2.CAP_PROP_FRAME_WIDTH: self._cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            cv2.CAP_PROP_FRAME_HEIGHT: self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            cv2.CAP_PROP_FPS: self._cap.get(cv2.CAP_PROP_FPS),
            cv2.CAP_PROP_FOURCC: self._cap.get(cv2.CAP_PROP_FOURCC),
        }
        self._condition = threading.Condition()
        self._frame: np.ndarray | None = None
        self._sequence = 0
        self._read_sequence = 0
        self._stopped = False
        self._failed = False
        self._thread: threading.Thread | None = None
        if self._cap.isOpened():
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()

    def isOpened(self) -> bool:
        return self._cap.isOpened() and not self._failed

    def get(self, property_id: int) -> float:
        return self._properties.get(property_id, 0.0)

    def read(self) -> Tuple[bool, np.ndarray | None]:
        deadline = time.monotonic() + 1.0
        with self._condition:
            while (
                self._sequence <= self._read_sequence
                and not self._failed
                and not self._stopped
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, None
                self._condition.wait(timeout=remaining)

            if self._failed or self._stopped or self._frame is None:
                return False, None

            self._read_sequence = self._sequence
            return True, self._frame

    def release(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self._cap.release()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def _capture_loop(self) -> None:
        while True:
            with self._condition:
                if self._stopped:
                    return

            ok, frame = self._cap.read()
            if not ok or frame is None:
                with self._condition:
                    if not self._stopped:
                        self._failed = True
                    self._condition.notify_all()
                return

            with self._condition:
                if self._stopped:
                    return
                self._frame = frame
                self._sequence += 1
                self._condition.notify_all()


class StaticImageSource:
    """将单张图片作为持续帧源，便于直接读取 SD 卡测试图片。"""

    def __init__(self, image_path: str) -> None:
        expanded_path = os.path.expandvars(os.path.expanduser(image_path))
        self.path = Path(expanded_path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"输入图片不存在: {self.path}")

        self._frame = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
        if self._frame is None:
            raise RuntimeError(f"输入图片无法解码或格式不受支持: {self.path}")
        self.source_kind = "image"
        self.current_frame_index = 1
        self.current_timestamp_seconds = 0.0
        self.current_source_name = self.path.name

    def isOpened(self) -> bool:
        return self._frame is not None

    def read(self) -> Tuple[bool, np.ndarray | None]:
        if self._frame is None:
            return False, None
        return True, self._frame.copy()

    def release(self) -> None:
        pass

    def get(self, property_id: int) -> float:
        if property_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._frame.shape[1])
        if property_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._frame.shape[0])
        if property_id == cv2.CAP_PROP_FRAME_COUNT:
            return 1.0
        return 0.0


class VideoFileSource:
    """从 SD 卡或本地文件读取视频，支持实时节奏、极速测试和循环播放。"""

    def __init__(self, video_path: str, *, loop: bool, playback: str) -> None:
        expanded_path = os.path.expandvars(os.path.expanduser(video_path))
        self.path = Path(expanded_path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"输入视频不存在: {self.path}")

        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError(f"输入视频无法解码或格式不受支持: {self.path}")
        self._loop = bool(loop)
        self._playback = playback
        self._fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        self._frame_count = int(round(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self._clip_frame_index = 0
        self._global_frame_index = 0
        self._loop_index = 0
        self._playback_started = 0.0
        self.ended = False
        self.source_kind = "video"
        self.current_frame_index = 0
        self.current_timestamp_seconds = 0.0
        self.current_source_name = self.path.name

    def isOpened(self) -> bool:
        return self._cap.isOpened()

    def get(self, property_id: int) -> float:
        if property_id == cv2.CAP_PROP_FPS:
            return self._fps
        if property_id == cv2.CAP_PROP_FRAME_COUNT:
            return float(self._frame_count)
        return float(self._cap.get(property_id))

    def read(self) -> Tuple[bool, np.ndarray | None]:
        ok, frame = self._cap.read()
        if not ok or frame is None:
            if not self._loop:
                self.ended = True
                return False, None
            self._loop_index += 1
            self._clip_frame_index = 0
            self._playback_started = 0.0
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
            if not ok or frame is None:
                self.ended = True
                return False, None

        self._clip_frame_index += 1
        self._global_frame_index += 1
        self.current_frame_index = self._clip_frame_index
        source_time = float(self._cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        if source_time <= 0.0 and self._fps > 0:
            source_time = (self._clip_frame_index - 1) / self._fps
        self.current_timestamp_seconds = max(source_time, 0.0)

        if self._playback == "realtime" and self._fps > 0:
            now = time.monotonic()
            if self._playback_started <= 0.0:
                self._playback_started = now - self.current_timestamp_seconds
            target_time = self._playback_started + self.current_timestamp_seconds
            if target_time > now:
                time.sleep(target_time - now)
        return True, frame

    def release(self) -> None:
        self._cap.release()


def get_source_frame_info(source, fallback_frame: int) -> Tuple[int, float, str]:
    return (
        int(getattr(source, "current_frame_index", fallback_frame)),
        float(getattr(source, "current_timestamp_seconds", 0.0)),
        str(getattr(source, "current_source_name", "")),
    )


def describe_frame_source(source, args: argparse.Namespace) -> Dict[str, object]:
    kind = str(getattr(source, "source_kind", "camera"))
    path = args.input_image or args.input_video
    return {
        "kind": kind,
        "path": str(Path(path).expanduser()) if path else "",
        "width": int(round(source.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "height": int(round(source.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        "fps": float(source.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(round(source.get(cv2.CAP_PROP_FRAME_COUNT))),
        "video_playback": args.video_playback if args.input_video else "",
        "video_loop": bool(args.video_loop) if args.input_video else False,
    }


def open_camera(
    camera_index: int,
    width: int,
    height: int,
    backend: str,
    *,
    latest_frame: bool = True,
    fps: float = 0.0,
    fourcc: str = "",
):
    if latest_frame:
        return LatestFrameCamera(camera_index, width, height, backend, fps, fourcc)
    return _open_video_capture(camera_index, width, height, backend, fps, fourcc)


def open_frame_source(args: argparse.Namespace):
    if args.input_image:
        return StaticImageSource(args.input_image)
    if args.input_video:
        return VideoFileSource(
            args.input_video,
            loop=args.video_loop,
            playback=args.video_playback,
        )
    return open_camera(
        args.camera,
        args.camera_width,
        args.camera_height,
        args.camera_backend,
        latest_frame=(not args.camera_buffered),
        fps=args.camera_fps,
        fourcc=args.camera_fourcc,
    )


def decode_fourcc(value: float) -> str:
    encoded = int(value)
    text = "".join(chr((encoded >> (8 * index)) & 0xFF) for index in range(4))
    return text if all(32 <= ord(char) <= 126 for char in text) else "unknown"


def parse_int_auto(value: str) -> int:
    return int(value, 0)


def choose_threshold(gray_small: np.ndarray, args: argparse.Namespace) -> int:
    threshold_source = gray_small
    threshold_min = args.threshold_min
    threshold_max = args.threshold_max
    if args.morph_cfg & PREPROC_SOBEL:
        # Sobel 输出的数值分布与灰度图不同，直接沿用灰度阈值会漏掉大量弱边缘。
        denoise_mode = (args.morph_cfg >> 4) & 0x3
        if denoise_mode == 1:
            threshold_source = cv2.GaussianBlur(threshold_source, (3, 3), 0)
        grad_x = cv2.Sobel(threshold_source, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(threshold_source, cv2.CV_16S, 0, 1, ksize=3)
        threshold_source = np.clip(
            np.abs(grad_x.astype(np.int32)) + np.abs(grad_y.astype(np.int32)),
            0,
            255,
        ).astype(np.uint8)
        threshold_min = args.sobel_threshold_min
        threshold_max = args.sobel_threshold_max

    if args.threshold_mode == "otsu":
        threshold_value, _ = cv2.threshold(
            threshold_source, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        value = int(threshold_value)
    elif args.threshold_mode == "percentile":
        value = int(np.percentile(threshold_source, args.threshold_percentile))
    else:
        value = int(args.threshold)

    value = max(threshold_min, min(threshold_max, value))
    return value


class AsyncFpgaRunner:
    """在后台处理最新 FPGA 任务，避免 PCIe 往返阻塞 HDMI 主循环。"""

    def __init__(
        self,
        fpga: FpgaPreprocessClient,
        args: argparse.Namespace,
        initial_status,
    ) -> None:
        self._fpga = fpga
        self._args = args
        self._lock = threading.Lock()
        self._pending: Tuple[np.ndarray, int] | None = None
        self._status = initial_status
        self._mask = np.zeros((args.fpga_height, args.fpga_width), dtype=np.uint8)
        self._boxes: List[Tuple[int, int, int, int]] = []
        self._threshold = int(initial_status.threshold)
        self._configured_threshold: int | None = None
        self._busy = False
        self._stopped = False
        self._submit_generation = 0
        self._completed_generation = 0
        self._last_duration_ms = 0.0
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, gray_small: np.ndarray) -> None:
        with self._lock:
            self._submit_generation += 1
            self._pending = (gray_small.copy(), self._submit_generation)

    def snapshot(self):
        with self._lock:
            if self._error is not None:
                raise RuntimeError(f"FPGA 后台处理失败: {self._error}") from self._error
            return (
                self._status,
                self._mask.copy(),
                list(self._boxes),
                self._threshold,
                self._busy,
                self._submit_generation,
                self._completed_generation,
                self._last_duration_ms,
            )

    def close(self) -> None:
        with self._lock:
            self._stopped = True
        self._thread.join(timeout=1.0)

    def _worker(self) -> None:
        while True:
            with self._lock:
                if self._stopped:
                    return
                task = self._pending
                if task is not None:
                    self._pending = None
                    self._busy = True

            if task is None:
                time.sleep(0.002)
                continue

            gray_small, generation = task
            started = time.perf_counter()
            try:
                threshold = choose_threshold(gray_small, self._args)
                if threshold != self._configured_threshold:
                    self._fpga.configure(
                        width=self._args.fpga_width,
                        height=self._args.fpga_height,
                        threshold=threshold,
                        roi=(self._args.roi_x, self._args.roi_y, self._args.roi_w, self._args.roi_h),
                        morph_cfg=self._args.morph_cfg,
                    )
                    self._configured_threshold = threshold

                self._fpga.write_grayscale_frame(gray_small)
                self._fpga.start()
                status = self._fpga.wait_done()
                raw_mask = self._fpga.read_mask(status.width, status.height)
                mask, boxes = refine_mask(
                    raw_mask,
                    cleanup_mode=self._args.mask_cleanup,
                    kernel_size=self._args.mask_kernel,
                    min_area=self._args.mask_min_area,
                    max_area_ratio=self._args.mask_max_area_ratio,
                    reject_border=(not self._args.mask_keep_border),
                )
            except Exception as exc:
                with self._lock:
                    self._error = exc
                    self._busy = False
                return

            with self._lock:
                self._status = status
                self._mask = mask
                self._boxes = boxes
                self._threshold = threshold
                self._completed_generation = generation
                self._last_duration_ms = (time.perf_counter() - started) * 1000.0
                self._busy = False


def create_hyperlpr_detector(args: argparse.Namespace) -> HyperLprDetector:
    return HyperLprDetector(
        hyperlpr_lib=args.hyperlpr_lib,
        model_dir=args.hyperlpr_model_dir,
        mnn_lib=args.mnn_lib,
        max_num=args.hyperlpr_max_num,
        threads=args.hyperlpr_threads,
        use_half=(not args.hyperlpr_no_half),
        box_conf_threshold=args.hyperlpr_box_threshold,
        nms_threshold=args.hyperlpr_nms_threshold,
        rec_confidence_threshold=args.hyperlpr_rec_threshold,
    )


def create_detector(args: argparse.Namespace) -> BaseDetector:
    if args.detector == "hyperlpr":
        return create_hyperlpr_detector(args)
    if args.detector == "rknn":
        labels = [item.strip() for item in args.rknn_labels.split(",") if item.strip()]
        recognizer = None
        if not args.rknn_disable_hyperlpr_ocr:
            recognizer = create_hyperlpr_detector(args)
        return RknnLiteDetector(
            model_path=args.rknn_model,
            labels=labels,
            input_size=args.rknn_input_size,
            conf_threshold=args.rknn_conf_threshold,
            nms_threshold=args.rknn_nms_threshold,
            core_mask=args.rknn_core_mask,
            recognizer=recognizer,
            ocr_cache_seconds=args.rknn_ocr_cache_seconds,
            ocr_cache_iou=args.rknn_ocr_cache_iou,
            serialize_inference=(not args.rknn_allow_concurrent_inference),
            refine_box_from_recognizer=(not args.rknn_disable_ocr_box_refinement),
        )
    return MockDetector()


def create_person_detector(args: argparse.Namespace) -> BaseDetector | None:
    if not args.person_model:
        return None
    return PersonRknnDetector(
        model_path=args.person_model,
        input_size=args.person_input_size,
        conf_threshold=args.person_conf_threshold,
        nms_threshold=args.person_nms_threshold,
        core_mask=args.person_core_mask,
        num_classes=args.person_model_classes,
        serialize_inference=(not args.rknn_allow_concurrent_inference),
        class_margin_threshold=args.person_class_margin,
        confirmation_hits=args.person_confirmation_hits,
        confirmation_threshold=args.person_confirmation_threshold,
        instant_threshold=args.person_instant_threshold,
        match_iou=args.person_confirmation_iou,
        min_height_ratio=args.person_min_height_ratio,
        max_width_height_ratio=args.person_max_width_height_ratio,
        temporal_confirmation=(not args.person_disable_confirmation),
    )


def run_detector_once(
    detector: BaseDetector,
    frame: np.ndarray,
    candidate_boxes_full: List[Tuple[int, int, int, int]],
    args: argparse.Namespace,
    *,
    extra_roi_boxes: List[Tuple[int, int, int, int]] | None = None,
    allow_full_frame: bool = True,
) -> Tuple[List[Detection], str]:
    full_detections: List[Detection] = []
    roi_detections: List[Detection] = []
    detector_mode = f"{args.detector_source}_scan"
    accuracy_priority = bool(getattr(args, "detector_accuracy_priority", False))
    effective_min_score = args.detector_min_score
    if accuracy_priority:
        effective_min_score = min(effective_min_score, 0.02)

    def detect_on_full_image(source_image: np.ndarray, target_width: int) -> List[Detection]:
        if args.detector == "rknn":
            return [
                det
                for det in detector.detect(source_image)
                if det.score >= effective_min_score
            ]

        if target_width <= 0:
            target_width = args.detector_input_width
        detector_frame, scale_x, scale_y = resize_detector_input(
            source_image,
            target_width,
            args.detector_input_height,
        )
        return [
            scale_detection(det, scale_x, scale_y)
            for det in detector.detect(detector_frame)
            if det.score >= effective_min_score
        ]

    def offset_detections(
        detections: List[Detection],
        offset_x: int,
        offset_y: int,
    ) -> List[Detection]:
        shifted: List[Detection] = []
        for det in detections:
            dx, dy, dw, dh = det.box
            shifted.append(
                Detection(
                    label=det.label,
                    raw_label=det.raw_label,
                    type_name=det.type_name,
                    full_text=det.full_text,
                    score=det.score,
                    box=(offset_x + dx, offset_y + dy, dw, dh),
                )
            )
        return shifted

    if allow_full_frame and args.detector_source in ("full", "hybrid"):
        tracked_count = len(extra_roi_boxes) if extra_roi_boxes else 0
        candidate_count = len(candidate_boxes_full)
        target_count = max(1, min(args.detector_max_rois, args.hyperlpr_max_num))
        search_mode = (
            accuracy_priority
            or not bool(extra_roi_boxes)
            or tracked_count < min(2, target_count)
            or candidate_count > tracked_count
        )
        full_frame_width = args.detector_search_input_width if search_mode else args.detector_track_input_width
        if full_frame_width <= 0:
            full_frame_width = args.detector_input_width

        full_detections = detect_on_full_image(frame, full_frame_width)
        if accuracy_priority:
            tile_detections: List[Detection] = []
            for tile_x, tile_y, tile_w, tile_h in build_detection_tiles(frame.shape[1], frame.shape[0], overlap_ratio=0.25):
                tile = frame[tile_y:tile_y + tile_h, tile_x:tile_x + tile_w]
                if tile.size == 0:
                    continue
                tile_results = detect_on_full_image(tile, full_frame_width)
                if tile_results:
                    tile_detections = merge_detections(
                        tile_detections,
                        offset_detections(tile_results, tile_x, tile_y),
                        iou_threshold=args.detector_merge_iou,
                    )
            if tile_detections:
                full_detections = merge_detections(
                    full_detections,
                    tile_detections,
                    iou_threshold=args.detector_merge_iou,
                )

            if len(full_detections) < target_count:
                enhanced_tile_detections: List[Detection] = []
                enhanced_frame_for_tiles = enhance_plate_frame(frame)
                for tile_x, tile_y, tile_w, tile_h in build_detection_tiles(frame.shape[1], frame.shape[0], overlap_ratio=0.25):
                    tile = enhanced_frame_for_tiles[tile_y:tile_y + tile_h, tile_x:tile_x + tile_w]
                    if tile.size == 0:
                        continue
                    tile_results = detect_on_full_image(tile, full_frame_width)
                    if tile_results:
                        enhanced_tile_detections = merge_detections(
                            enhanced_tile_detections,
                            offset_detections(tile_results, tile_x, tile_y),
                            iou_threshold=args.detector_merge_iou,
                        )
                if enhanced_tile_detections:
                    full_detections = merge_detections(
                        full_detections,
                        enhanced_tile_detections,
                        iou_threshold=args.detector_merge_iou,
                    )

        if accuracy_priority and len(full_detections) < target_count:
            enhanced_frame = enhance_plate_frame(frame)
            enhanced_detections = detect_on_full_image(enhanced_frame, full_frame_width)
            if enhanced_detections:
                full_detections = merge_detections(
                    full_detections,
                    enhanced_detections,
                    iou_threshold=max(0.35, args.detector_merge_iou),
                )
        if full_detections:
            detector_mode = "full"

    need_roi_scan = False
    if args.detector_source == "roi":
        need_roi_scan = True
    elif args.detector_source == "hybrid":
        target_count = max(1, min(args.detector_max_rois, args.hyperlpr_max_num))
        need_roi_scan = len(full_detections) < target_count or len(candidate_boxes_full) > len(full_detections)

    if need_roi_scan:
        detector_boxes = candidate_boxes_full[: args.detector_max_rois] if args.detector_max_rois > 0 else []
        if extra_roi_boxes:
            detector_boxes = merge_candidate_boxes(
                list(extra_roi_boxes),
                detector_boxes,
                max_boxes=args.detector_max_rois,
                iou_threshold=args.detector_box_merge_iou,
            )
        for roi_box in detector_boxes:
            x, y, w, h = expand_box(
                roi_box,
                frame.shape[1],
                frame.shape[0],
                args.detector_roi_expand,
            )
            crop = frame[y:y + h, x:x + w]
            if crop.size == 0:
                continue

            for det in detector.detect(crop):
                if det.score < effective_min_score:
                    continue
                dx, dy, dw, dh = det.box
                roi_detections.append(
                    Detection(
                        label=det.label,
                        raw_label=det.raw_label,
                        type_name=det.type_name,
                        full_text=det.full_text,
                        score=det.score,
                        box=(x + dx, y + dy, dw, dh),
                    )
                )
            if accuracy_priority and not roi_detections:
                enhanced_crop = enhance_plate_frame(crop)
                for det in detector.detect(enhanced_crop):
                    if det.score < effective_min_score:
                        continue
                    dx, dy, dw, dh = det.box
                    roi_detections.append(
                        Detection(
                            label=det.label,
                            raw_label=det.raw_label,
                            type_name=det.type_name,
                            full_text=det.full_text,
                            score=det.score,
                            box=(x + dx, y + dy, dw, dh),
                        )
                    )
        if roi_detections and not full_detections:
            detector_mode = "roi"
        elif roi_detections and full_detections:
            detector_mode = "hybrid"

    if full_detections and roi_detections:
        return merge_detections(
            full_detections,
            roi_detections,
            iou_threshold=args.detector_merge_iou,
        ), detector_mode
    if full_detections:
        return full_detections, detector_mode
    return roi_detections, detector_mode


class AsyncDetectorRunner:
    def __init__(self, detector: BaseDetector, args: argparse.Namespace) -> None:
        self._detector = detector
        self._args = args
        self._lock = threading.Lock()
        self._pending: Tuple[
            np.ndarray,
            np.ndarray,
            List[Tuple[int, int, int, int]],
            int,
            int,
            float,
            str,
        ] | None = None
        self._latest: List[Detection] = []
        self._latest_source_gray: np.ndarray | None = None
        self._latest_mode = "idle"
        self._busy = False
        self._stopped = False
        self._submit_count = 0
        self._submit_generation = 0
        self._completed_generation = 0
        self._latest_update_time = 0.0
        self._last_duration_ms = 0.0
        self._completed_result: List[Detection] = []
        self._completed_source_frame = 0
        self._completed_source_time = 0.0
        self._completed_source_name = ""
        self._completed_mode = "idle"
        self._miss_count = 0
        self._last_success_time = 0.0
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(
        self,
        frame: np.ndarray,
        source_gray: np.ndarray,
        candidate_boxes_full: List[Tuple[int, int, int, int]],
        source_frame: int = 0,
        source_time_s: float = 0.0,
        source_name: str = "",
    ) -> None:
        if self._args.detector == "mock":
            return

        with self._lock:
            self._submit_generation += 1
            # 摄像头每次返回独立 ndarray；保留引用即可，避免每次提交复制整张 720p 图像。
            task = (
                frame,
                source_gray,
                list(candidate_boxes_full),
                self._submit_generation,
                source_frame,
                source_time_s,
                source_name,
            )
            self._pending = task

    def snapshot(self) -> Tuple[List[Detection], str, bool, int, int, float, float, np.ndarray | None]:
        with self._lock:
            return (
                list(self._latest),
                self._latest_mode,
                self._busy,
                self._submit_generation,
                self._completed_generation,
                self._latest_update_time,
                self._last_duration_ms,
                self._latest_source_gray,
            )

    def metrics_snapshot(self) -> Tuple[List[Detection], int, int, float, str, float, str]:
        with self._lock:
            return (
                list(self._completed_result),
                self._completed_generation,
                self._completed_source_frame,
                self._completed_source_time,
                self._completed_source_name,
                self._last_duration_ms,
                self._completed_mode,
            )

    def close(self) -> None:
        with self._lock:
            self._stopped = True
        self._thread.join(timeout=1.0)

    def _worker(self) -> None:
        while True:
            with self._lock:
                if self._stopped:
                    return
                task = self._pending
                latest_snapshot = list(self._latest)
                if task is not None:
                    self._pending = None
                    self._busy = True

            if task is None:
                time.sleep(0.005)
                continue

            (
                frame,
                source_gray,
                candidate_boxes_full,
                generation,
                source_frame,
                source_time_s,
                source_name,
            ) = task
            started = time.perf_counter()
            self._submit_count += 1
            extra_roi_boxes: List[Tuple[int, int, int, int]] = []
            for det in latest_snapshot[: max(self._args.detector_max_rois, 0)]:
                extra_roi_boxes.append(
                    expand_box(
                        det.box,
                        frame.shape[1],
                        frame.shape[0],
                        max(self._args.detector_track_expand, self._args.detector_roi_expand),
                    )
                )

            if self._args.detector_fast_pass_width > 0 and not latest_snapshot:
                fast_args = argparse.Namespace(**vars(self._args))
                fast_args.detector_source = "full"
                fast_args.detector_search_input_width = self._args.detector_fast_pass_width
                fast_args.detector_track_input_width = self._args.detector_fast_pass_width
                fast_args.detector_accuracy_priority = False
                fast_detections, _ = run_detector_once(
                    self._detector,
                    frame,
                    candidate_boxes_full,
                    fast_args,
                    extra_roi_boxes=None,
                    allow_full_frame=True,
                )
                if fast_detections:
                    with self._lock:
                        self._latest = fast_detections
                        self._latest_mode = "quick"
                        self._latest_update_time = time.monotonic()
                        self._latest_source_gray = source_gray
                        self._miss_count = 0
                        self._last_success_time = time.monotonic()

            allow_full_frame = True
            if self._args.detector_source == "roi":
                allow_full_frame = False
            elif self._args.detector_source == "hybrid":
                if getattr(self._args, "detector_accuracy_priority", False):
                    allow_full_frame = True
                elif latest_snapshot:
                    period = max(1, self._args.detector_fullframe_period)
                    allow_full_frame = (self._submit_count % period) == 0

            detections, detector_mode = run_detector_once(
                self._detector,
                frame,
                candidate_boxes_full,
                self._args,
                extra_roi_boxes=extra_roi_boxes,
                allow_full_frame=allow_full_frame,
            )

            with self._lock:
                self._completed_generation = generation
                metric_detections: List[Detection] = []
                if detections:
                    self._latest = stabilize_detections_with_history(
                        latest_snapshot,
                        detections,
                        iou_threshold=self._args.detector_stabilize_iou,
                    )
                    self._latest_mode = detector_mode
                    self._latest_update_time = time.monotonic()
                    self._latest_source_gray = source_gray
                    self._miss_count = 0
                    self._last_success_time = time.monotonic()
                    metric_detections = list(self._latest)
                elif self._latest and (
                    (time.monotonic() - self._last_success_time) < self._args.detector_hold_seconds
                ) and self._miss_count < self._args.detector_hold_frames:
                    self._miss_count += 1
                    self._latest_mode = f"{detector_mode}_hold"
                else:
                    self._latest = []
                    self._latest_source_gray = None
                    self._latest_mode = detector_mode
                    self._latest_update_time = 0.0
                    self._miss_count = 0
                self._completed_result = metric_detections
                self._completed_source_frame = source_frame
                self._completed_source_time = source_time_s
                self._completed_source_name = source_name
                self._completed_mode = detector_mode
                self._last_duration_ms = (time.perf_counter() - started) * 1000.0
                self._busy = False


class AsyncFrameDetectorRunner:
    """为独立全帧检测器保留最新任务，避免阻塞摄像头与车牌通道。"""

    def __init__(self, detector: BaseDetector) -> None:
        self._detector = detector
        self._lock = threading.Lock()
        self._pending: Tuple[np.ndarray, int] | None = None
        self._latest: List[Detection] = []
        self._busy = False
        self._stopped = False
        self._submit_generation = 0
        self._completed_generation = 0
        self._latest_update_time = 0.0
        self._last_duration_ms = 0.0
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        with self._lock:
            self._submit_generation += 1
            self._pending = (frame, self._submit_generation)

    def snapshot(self) -> Tuple[List[Detection], bool, int, int, float, float]:
        with self._lock:
            return (
                list(self._latest),
                self._busy,
                self._submit_generation,
                self._completed_generation,
                self._latest_update_time,
                self._last_duration_ms,
            )

    def close(self) -> None:
        with self._lock:
            self._stopped = True
        self._thread.join(timeout=1.0)

    def _worker(self) -> None:
        while True:
            with self._lock:
                if self._stopped:
                    return
                task = self._pending
                if task is not None:
                    self._pending = None
                    self._busy = True

            if task is None:
                time.sleep(0.005)
                continue

            frame, generation = task
            started = time.perf_counter()
            try:
                detections = self._detector.detect(frame)
            except Exception as exc:
                print(f"行人检测失败: {exc}", flush=True)
                detections = []

            with self._lock:
                self._latest = detections
                self._completed_generation = generation
                self._latest_update_time = time.monotonic()
                self._last_duration_ms = (time.perf_counter() - started) * 1000.0
                self._busy = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RK3568 + FPGA 实时预览管线")
    parser.add_argument("--resource-root", required=True, help="PCIe 设备 sysfs 根目录")
    parser.add_argument(
        "--input-image",
        "--sd-image",
        dest="input_image",
        default="",
        help="从 SD 卡或本地文件系统读取单张图片；留空时继续使用摄像头",
    )
    parser.add_argument(
        "--input-video",
        "--sd-video",
        dest="input_video",
        default="",
        help="从 SD 卡或本地文件系统读取视频；与 --input-image 互斥",
    )
    parser.add_argument("--video-loop", action="store_true", help="输入视频播放结束后从头循环")
    parser.add_argument(
        "--video-playback",
        choices=("realtime", "fast"),
        default="realtime",
        help="视频播放节奏：realtime=不快于源帧率，fast=不等待并用于吞吐率测试",
    )
    parser.add_argument("--camera", type=parse_int_auto, default=0, help="摄像头编号")
    parser.add_argument("--camera-width", type=parse_int_auto, default=1280, help="摄像头采集宽度")
    parser.add_argument("--camera-height", type=parse_int_auto, default=720, help="摄像头采集高度")
    parser.add_argument(
        "--camera-backend",
        choices=("auto", "v4l2", "gstreamer"),
        default="auto",
        help="摄像头后端，优先推荐 v4l2",
    )
    parser.add_argument("--camera-fps", type=float, default=0.0, help="请求摄像头输出帧率，0 表示使用设备默认值")
    parser.add_argument("--camera-fourcc", default="", help="请求摄像头像素格式，例如 MJPG；留空保持设备默认格式")
    parser.add_argument("--camera-read-retries", type=parse_int_auto, default=8, help="单次掉帧后的最大重试次数")
    parser.add_argument("--camera-retry-delay", type=float, default=0.20, help="摄像头重试间隔，单位秒")
    parser.add_argument("--camera-buffered", action="store_true", help="使用传统同步读帧；默认后台持续取最新帧以降低画面延迟")
    parser.add_argument("--fpga-width", type=parse_int_auto, default=DEFAULT_SAFE_WIDTH, help="送入 FPGA 的宽度")
    parser.add_argument("--fpga-height", type=parse_int_auto, default=DEFAULT_SAFE_HEIGHT, help="送入 FPGA 的高度")
    parser.add_argument("--threshold", type=parse_int_auto, default=128, help="固定阈值模式下的 FPGA 阈值")
    parser.add_argument("--threshold-mode", choices=("fixed", "otsu", "percentile"), default="percentile", help="阈值模式")
    parser.add_argument("--threshold-percentile", type=float, default=78.0, help="percentile 模式的百分位")
    parser.add_argument("--threshold-min", type=parse_int_auto, default=72, help="自适应阈值下限")
    parser.add_argument("--threshold-max", type=parse_int_auto, default=224, help="自适应阈值上限")
    parser.add_argument("--sobel-threshold-min", type=parse_int_auto, default=24, help="FPGA Sobel 模式的自适应阈值下限")
    parser.add_argument("--sobel-threshold-max", type=parse_int_auto, default=192, help="FPGA Sobel 模式的自适应阈值上限")
    parser.add_argument("--roi-x", type=parse_int_auto, default=0)
    parser.add_argument("--roi-y", type=parse_int_auto, default=0)
    parser.add_argument("--roi-w", type=parse_int_auto, default=0)
    parser.add_argument("--roi-h", type=parse_int_auto, default=0)
    parser.add_argument("--morph-cfg", type=parse_int_auto, default=0)
    parser.add_argument("--headless", action="store_true", help="不弹窗，只保存或打印日志")
    parser.add_argument("--max-frames", type=parse_int_auto, default=0, help="限制运行帧数，0 表示持续运行")
    parser.add_argument("--save-dir", default="", help="调试图输出目录")
    parser.add_argument("--save-every", type=parse_int_auto, default=30, help="每隔多少帧保存一组图像")
    parser.add_argument("--log-every", type=parse_int_auto, default=10, help="每隔多少帧打印一次进度")
    parser.add_argument(
        "--metrics-dir",
        "--test-output",
        dest="metrics_dir",
        default="",
        help="自动测试结果目录；启用后输出 frames.csv、detections.csv 和 summary.json",
    )
    parser.add_argument(
        "--metrics-ground-truth",
        default="",
        help="可选标注JSON；提供后统计IoU、Precision、Recall、F1、OCR及类型准确率",
    )
    parser.add_argument("--metrics-iou-threshold", type=float, default=0.50, help="预测框与标注框匹配的IoU阈值")
    parser.add_argument("--metrics-warmup-frames", type=parse_int_auto, default=0, help="统计时忽略前多少帧的性能采样")
    parser.add_argument(
        "--display-mode",
        choices=("outline", "overlay", "camera", "mask"),
        default="outline",
        help="HDMI 显示内容：outline=原图加轮廓和框，overlay=原图叠加掩码，camera=只看原图，mask=只看掩码",
    )
    parser.add_argument("--overlay-alpha", type=float, default=0.12, help="overlay 模式透明度，范围 0~1")
    parser.add_argument("--fullscreen", action="store_true", help="全屏显示到 HDMI")
    parser.add_argument("--display-width", type=parse_int_auto, default=0, help="显示窗口宽度，0 表示不额外缩放")
    parser.add_argument("--display-height", type=parse_int_auto, default=0, help="显示窗口高度，0 表示不额外缩放")
    parser.add_argument("--window-name", default="rk3568_fpga_hdmi", help="显示窗口名称")
    parser.add_argument("--text-font", default="", help="中文绘制字体路径，留空时自动查找 HyperLPR 自带字体")
    parser.add_argument("--text-font-size", type=parse_int_auto, default=28, help="中文绘制字体大小")
    parser.add_argument("--draw-roi", action="store_true", help="在画面上额外绘制候选区域框")
    parser.add_argument("--hide-status", action="store_true", help="隐藏底部状态栏")
    parser.add_argument("--box-display-mode", choices=("hold", "flash"), default="hold", help="车牌框显示模式：hold=持续显示最近结果，flash=仅在新结果返回时闪一下")
    parser.add_argument("--box-hold-seconds", type=float, default=0.40, help="hold 模式下车牌框最多保留最近结果多少秒，越小跟随越灵敏")
    parser.add_argument("--disable-box-tracking", action="store_true", help="关闭检测间隔内的轻量光流跟随，仅显示原始检测框")
    parser.add_argument("--disable-detection-lag-compensation", action="store_true", help="关闭异步检测源帧到当前帧的位置补偿，仅用于 A/B 对比")
    parser.add_argument("--detector", choices=("mock", "hyperlpr", "rknn"), default="mock", help="候选区域二阶段检测器")
    parser.add_argument("--detector-source", choices=("full", "roi", "hybrid"), default="full", help="检测器输入来源：full=全帧，roi=只跑 FPGA ROI，hybrid=先全帧后 ROI")
    parser.add_argument("--detector-interval", type=parse_int_auto, default=3, help="每隔多少帧真正跑一次检测，其他帧复用上次结果")
    parser.add_argument("--detector-input-width", type=parse_int_auto, default=960, help="送给全帧检测器的宽度，0 表示不缩放")
    parser.add_argument("--detector-search-input-width", type=parse_int_auto, default=960, help="搜索新车牌时的全帧检测宽度")
    parser.add_argument("--detector-track-input-width", type=parse_int_auto, default=800, help="跟踪已有车牌时的全帧检测宽度")
    parser.add_argument("--detector-input-height", type=parse_int_auto, default=0, help="送给全帧检测器的高度，0 表示按比例自动推导")
    parser.add_argument("--detector-max-rois", type=parse_int_auto, default=4, help="每帧最多送给检测器的 ROI 数量")
    parser.add_argument("--detector-min-score", type=float, default=0.10, help="绘制检测结果的最低分数")
    parser.add_argument("--detector-roi-expand", type=float, default=0.25, help="ROI 模式下，送检前对候选框按比例向外扩展")
    parser.add_argument("--detector-track-expand", type=float, default=0.35, help="基于上一帧车牌结果生成跟踪 ROI 时的扩展比例")
    parser.add_argument("--detector-fullframe-period", type=parse_int_auto, default=2, help="hybrid 模式下每隔多少次检测再跑一次全帧")
    parser.add_argument("--detector-hold-frames", type=parse_int_auto, default=3, help="检测短暂丢失后，保留上一批车牌结果的帧数")
    parser.add_argument("--detector-hold-seconds", type=float, default=0.6, help="检测短暂丢失后，最多保留上一批车牌结果多少秒")
    parser.add_argument("--hyperlpr-lib", default="", help="libhyperlpr3.so 路径，留空时优先使用板端默认安装路径")
    parser.add_argument("--hyperlpr-model-dir", default="", help="HyperLPR r2_mobile 模型目录，留空时优先使用板端默认安装路径")
    parser.add_argument("--mnn-lib", default="", help="libMNN.so 路径，留空时优先使用板端默认安装路径")
    parser.add_argument("--hyperlpr-max-num", type=parse_int_auto, default=6, help="HyperLPR 每帧最多返回多少个车牌")
    parser.add_argument("--hyperlpr-threads", type=parse_int_auto, default=1, help="HyperLPR 推理线程数")
    parser.add_argument("--hyperlpr-box-threshold", type=float, default=0.25, help="HyperLPR 检测置信度阈值")
    parser.add_argument("--hyperlpr-nms-threshold", type=float, default=0.45, help="HyperLPR NMS 阈值")
    parser.add_argument("--hyperlpr-rec-threshold", type=float, default=0.25, help="HyperLPR 识别置信度阈值")
    parser.add_argument("--hyperlpr-no-half", action="store_true", help="关闭 HyperLPR FP16")
    parser.add_argument("--rknn-model", default="", help="RKNN 模型路径，留空时按板端默认路径查找")
    parser.add_argument("--rknn-input-size", type=parse_int_auto, default=640, help="RKNN YOLO 输入尺寸，默认 640")
    parser.add_argument("--rknn-conf-threshold", type=float, default=0.10, help="RKNN YOLO 检测置信度阈值")
    parser.add_argument("--rknn-nms-threshold", type=float, default=0.45, help="RKNN YOLO NMS 阈值")
    parser.add_argument("--rknn-core-mask", default="auto", help="RKNN NPU core mask，例如 auto、0、1、2、0_1、0_1_2")
    parser.add_argument("--rknn-allow-concurrent-inference", action="store_true", help="允许两个 RKNN 模型并发调用 NPU；默认串行调度以避免 RK3568 单核 NPU 争抢")
    parser.add_argument("--rknn-labels", default="单层车牌,双层车牌", help="RKNN 类别名称，逗号分隔")
    parser.add_argument("--rknn-disable-hyperlpr-ocr", action="store_true", help="仅用 RKNN 做车牌框检测，不复用 HyperLPR 做文字识别")
    parser.add_argument("--rknn-disable-ocr-box-refinement", action="store_true", help="保留 OCR 文字但关闭 HyperLPR 内框校准，仅用于 A/B 对比")
    parser.add_argument("--rknn-ocr-cache-seconds", type=float, default=2.0, help="同一车牌成功 OCR 结果的复用时长，降低重复 CPU 推理")
    parser.add_argument("--rknn-ocr-cache-iou", type=float, default=0.50, help="复用 OCR 结果所需的最低框 IoU")
    parser.add_argument("--person-model", default="", help="独立行人 RKNN 模型路径；留空时完全关闭行人检测，不影响现有流程")
    parser.add_argument("--person-model-classes", type=parse_int_auto, choices=(1, 80), default=80, help="行人模型类别数：COCO 模型填 80，单类行人模型填 1")
    parser.add_argument("--person-input-size", type=parse_int_auto, default=640, help="行人 RKNN YOLO 输入尺寸")
    parser.add_argument("--person-conf-threshold", type=float, default=0.30, help="行人检测置信度阈值")
    parser.add_argument("--person-nms-threshold", type=float, default=0.45, help="行人检测 NMS 阈值")
    parser.add_argument("--person-core-mask", default="auto", help="行人 RKNN NPU core mask")
    parser.add_argument("--person-interval", type=parse_int_auto, default=3, help="行人模型每隔多少帧提交一次；与车牌检测异步独立")
    parser.add_argument("--person-hold-seconds", type=float, default=0.60, help="行人检测结果在两次推理间的最大显示时长")
    parser.add_argument("--person-class-margin", type=float, default=0.08, help="person 分数领先第二类别的最小差值，用于剔除类别模糊物体")
    parser.add_argument("--person-confirmation-hits", type=parse_int_auto, default=2, help="中等置信度行人上屏前需要连续命中的次数")
    parser.add_argument("--person-confirmation-threshold", type=float, default=0.48, help="连续确认行人的最低平均置信度")
    parser.add_argument("--person-instant-threshold", type=float, default=0.72, help="无需连续确认即可立即上屏的高置信度阈值")
    parser.add_argument("--person-confirmation-iou", type=float, default=0.25, help="相邻两次行人结果判定为同一目标的最低 IoU")
    parser.add_argument("--person-min-height-ratio", type=float, default=0.04, help="行人框最小高度占画面比例，用于过滤极小噪声框")
    parser.add_argument("--person-max-width-height-ratio", type=float, default=1.30, help="行人框最大宽高比，用于过滤明显横向物体；0 表示关闭")
    parser.add_argument("--person-disable-confirmation", action="store_true", help="关闭行人连续确认，仅用于 A/B 对比")
    parser.add_argument("--mask-cleanup", choices=("off", "open", "close", "open_close"), default="open_close", help="ARM 侧对掩码做轻量清理")
    parser.add_argument("--mask-kernel", type=parse_int_auto, default=3, help="掩码清理核尺寸，建议 3 或 5")
    parser.add_argument("--mask-min-area", type=parse_int_auto, default=48, help="候选区域最小面积，单位是 FPGA 小图像素")
    parser.add_argument("--mask-max-area-ratio", type=float, default=0.35, help="候选区域最大面积占比，用来去掉整片误检")
    parser.add_argument("--mask-keep-border", action="store_true", help="保留贴边候选区域，默认会去掉贴边大块")
    parser.add_argument("--detector-interval-hit", type=parse_int_auto, default=4, help="已经识别到车牌后，每隔多少帧再跑一次检测")
    parser.add_argument("--detector-accuracy-priority", action="store_true", help="识别率优先：强制更频繁全帧搜索，并在 hybrid 模式下总是做 ROI 补检")
    parser.add_argument("--detector-submit-max-gap-seconds", type=float, default=0.50, help="不管帧率如何，最慢多少秒必须重新提交一次检测")
    parser.add_argument("--detector-merge-iou", type=float, default=0.55, help="全帧与 ROI 结果去重的 IoU 阈值，越高越不容易把两个框合并成一个")
    parser.add_argument("--detector-box-merge-iou", type=float, default=0.60, help="ROI 候选框合并的 IoU 阈值，越高越保留更多候选框")
    parser.add_argument("--detector-stabilize-iou", type=float, default=0.45, help="历史结果稳定匹配的 IoU 阈值")
    parser.add_argument("--detector-fast-pass-width", type=parse_int_auto, default=0, help="快速首检宽度，>0 时会先用较小分辨率做一次全帧快扫，尽快把车牌显示出来")
    parser.add_argument("--detector-display-hold-seconds", type=float, default=0.25, help="显示层在下一次检测返回前，对最近一次稳定结果做短暂保留，避免空窗")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input_image and args.input_video:
        raise ValueError("--input-image/--sd-image 与 --input-video/--sd-video 不能同时使用")
    if args.metrics_ground_truth and not args.metrics_dir:
        raise ValueError("使用 --metrics-ground-truth 时必须同时指定 --metrics-dir")
    if not 0.0 <= args.metrics_iou_threshold <= 1.0:
        raise ValueError("metrics-iou-threshold 必须在 0 到 1 之间")
    if args.metrics_warmup_frames < 0:
        raise ValueError("metrics-warmup-frames 不能小于 0")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("overlay-alpha 必须在 0 到 1 之间")
    if args.mask_kernel < 1:
        raise ValueError("mask-kernel 必须大于等于 1")
    if args.camera_fps < 0:
        raise ValueError("camera-fps 不能小于 0")
    args.camera_fourcc = args.camera_fourcc.strip().upper()
    if args.camera_fourcc and len(args.camera_fourcc) != 4:
        raise ValueError("camera-fourcc 必须是四个字符，例如 MJPG")
    if args.mask_kernel % 2 == 0:
        args.mask_kernel += 1
    if args.detector_interval < 1:
        raise ValueError("detector-interval 必须大于等于 1")
    if args.detector_interval_hit < 1:
        raise ValueError("detector-interval-hit 必须大于等于 1")
    if args.text_font_size < 8:
        raise ValueError("text-font-size 不能小于 8")
    if args.detector_max_rois < 0:
        raise ValueError("detector-max-rois 不能小于 0")
    if not 0.0 <= args.detector_min_score <= 1.0:
        raise ValueError("detector-min-score 必须在 0 到 1 之间")
    if args.detector_search_input_width < 0:
        raise ValueError("detector-search-input-width 不能小于 0")
    if args.detector_track_input_width < 0:
        raise ValueError("detector-track-input-width 不能小于 0")
    if args.detector_roi_expand < 0:
        raise ValueError("detector-roi-expand 不能小于 0")
    if args.detector_track_expand < 0:
        raise ValueError("detector-track-expand 不能小于 0")
    if args.detector_fullframe_period < 1:
        raise ValueError("detector-fullframe-period 必须大于等于 1")
    if args.detector_hold_frames < 0:
        raise ValueError("detector-hold-frames 不能小于 0")
    if args.detector_hold_seconds < 0:
        raise ValueError("detector-hold-seconds 不能小于 0")
    if args.detector_submit_max_gap_seconds < 0:
        raise ValueError("detector-submit-max-gap-seconds 不能小于 0")
    if args.detector_display_hold_seconds < 0:
        raise ValueError("detector-display-hold-seconds 不能小于 0")
    if args.detector_fast_pass_width < 0:
        raise ValueError("detector-fast-pass-width 不能小于 0")
    if not 0.0 <= args.detector_merge_iou <= 1.0:
        raise ValueError("detector-merge-iou 必须在 0 到 1 之间")
    if not 0.0 <= args.detector_box_merge_iou <= 1.0:
        raise ValueError("detector-box-merge-iou 必须在 0 到 1 之间")
    if not 0.0 <= args.detector_stabilize_iou <= 1.0:
        raise ValueError("detector-stabilize-iou 必须在 0 到 1 之间")
    if args.rknn_input_size < 32:
        raise ValueError("rknn-input-size 不能小于 32")
    if not 0.0 <= args.rknn_conf_threshold <= 1.0:
        raise ValueError("rknn-conf-threshold 必须在 0 到 1 之间")
    if not 0.0 <= args.rknn_nms_threshold <= 1.0:
        raise ValueError("rknn-nms-threshold 必须在 0 到 1 之间")
    if args.rknn_ocr_cache_seconds < 0:
        raise ValueError("rknn-ocr-cache-seconds 不能小于 0")
    if not 0.0 <= args.rknn_ocr_cache_iou <= 1.0:
        raise ValueError("rknn-ocr-cache-iou 必须在 0 到 1 之间")
    if args.person_input_size < 32:
        raise ValueError("person-input-size 不能小于 32")
    if args.person_interval < 1:
        raise ValueError("person-interval 必须大于等于 1")
    if args.person_hold_seconds < 0:
        raise ValueError("person-hold-seconds 不能小于 0")
    if not 0.0 <= args.person_conf_threshold <= 1.0:
        raise ValueError("person-conf-threshold 必须在 0 到 1 之间")
    if not 0.0 <= args.person_nms_threshold <= 1.0:
        raise ValueError("person-nms-threshold 必须在 0 到 1 之间")
    if not 0.0 <= args.person_class_margin <= 1.0:
        raise ValueError("person-class-margin 必须在 0 到 1 之间")
    if args.person_confirmation_hits < 1:
        raise ValueError("person-confirmation-hits 必须大于等于 1")
    if not 0.0 <= args.person_confirmation_threshold <= 1.0:
        raise ValueError("person-confirmation-threshold 必须在 0 到 1 之间")
    if not 0.0 <= args.person_instant_threshold <= 1.0:
        raise ValueError("person-instant-threshold 必须在 0 到 1 之间")
    if not 0.0 <= args.person_confirmation_iou <= 1.0:
        raise ValueError("person-confirmation-iou 必须在 0 到 1 之间")
    if not 0.0 <= args.person_min_height_ratio <= 1.0:
        raise ValueError("person-min-height-ratio 必须在 0 到 1 之间")
    if args.person_max_width_height_ratio < 0.0:
        raise ValueError("person-max-width-height-ratio 必须大于等于 0")

    if args.detector_accuracy_priority:
        args.detector_min_score = min(args.detector_min_score, 0.05)
        args.hyperlpr_box_threshold = min(args.hyperlpr_box_threshold, 0.05)
        args.hyperlpr_rec_threshold = min(args.hyperlpr_rec_threshold, 0.05)
        args.hyperlpr_max_num = max(args.hyperlpr_max_num, 20)
        args.detector_max_rois = max(args.detector_max_rois, 20)

    FpgaPreprocessClient.validate_frame_size(args.fpga_width, args.fpga_height)
    if (args.fpga_width * args.fpga_height) > CURRENT_SAFE_FRAME_BYTES:
        raise RuntimeError("当前 FPGA 可用帧区不足，请降低 FPGA 输入分辨率")

    detector = create_detector(args)
    detector_runner = AsyncDetectorRunner(detector, args) if args.detector != "mock" else None
    try:
        person_detector = create_person_detector(args)
    except Exception:
        if detector_runner is not None:
            detector_runner.close()
        detector.close()
        raise
    person_runner = AsyncFrameDetectorRunner(person_detector) if person_detector is not None else None
    fpga = FpgaPreprocessClient(args.resource_root)
    cap = open_frame_source(args)

    if not cap.isOpened():
        if detector_runner is not None:
            detector_runner.close()
        detector.close()
        if person_runner is not None:
            person_runner.close()
        if person_detector is not None:
            person_detector.close()
        raise RuntimeError("图片或摄像头输入源打开失败")

    output_dir = ensure_output_dir(args.save_dir) if args.save_dir else None

    if not args.headless and not os.environ.get("DISPLAY"):
        if output_dir is not None:
            print("未检测到 DISPLAY，无法直接输出到 HDMI，已自动切换为 headless 模式。", flush=True)
            args.headless = True
        else:
            if detector_runner is not None:
                detector_runner.close()
            detector.close()
            if person_runner is not None:
                person_runner.close()
            if person_detector is not None:
                person_detector.close()
            cap.release()
            raise RuntimeError(
                "未检测到 DISPLAY，无法直接输出到 HDMI。请在板端图形终端运行，"
                "或先执行 export DISPLAY=:0 和 export XAUTHORITY=/home/linaro/.Xauthority。"
            )

    if not args.headless:
        prepare_window(args.window_name, args.fullscreen, args.display_width, args.display_height)

    unicode_font = resolve_text_font(args.text_font, args.text_font_size)

    startup_status = fpga.ensure_signature()
    fpga_runner = AsyncFpgaRunner(fpga, args, startup_status)
    if args.input_image:
        print(f"Input image ready: {Path(args.input_image).expanduser()}", flush=True)
    elif args.input_video:
        print(
            "Input video ready:",
            f"path={Path(args.input_video).expanduser()}",
            f"mode={int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))}x{int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))}",
            f"fps={cap.get(cv2.CAP_PROP_FPS):.3f}",
            f"frames={int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))}",
            f"playback={args.video_playback}",
            f"loop={int(args.video_loop)}",
            flush=True,
        )
    else:
        actual_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        actual_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_fourcc = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))
        print(
            "Camera ready:",
            f"index={args.camera}",
            f"backend={args.camera_backend}",
            f"mode={actual_width}x{actual_height}@{actual_fps:.1f}",
            f"fourcc={actual_fourcc}",
            flush=True,
        )
    print(
        "FPGA ready:",
        f"width={startup_status.width}",
        f"height={startup_status.height}",
        f"threshold={startup_status.threshold}",
        f"frame_bytes={startup_status.frame_bytes}",
    )
    if args.detector == "hyperlpr":
        print(
            "HyperLPR ready:",
            f"max_rois={args.detector_max_rois}",
            f"min_score={args.detector_min_score:.2f}",
            flush=True,
        )
    elif args.detector == "rknn":
        print(
            "RKNN ready:",
            f"input_size={args.rknn_input_size}",
            f"conf={args.rknn_conf_threshold:.2f}",
            f"nms={args.rknn_nms_threshold:.2f}",
            f"ocr={'off' if args.rknn_disable_hyperlpr_ocr else 'HyperLPR'}",
            f"ocr_cache={args.rknn_ocr_cache_seconds:.1f}s",
            f"npu_schedule={'concurrent' if args.rknn_allow_concurrent_inference else 'serialized'}",
            flush=True,
        )
    if person_detector is not None:
        print(
            "Person RKNN ready:",
            f"input_size={args.person_input_size}",
            f"classes={args.person_model_classes}",
            f"conf={args.person_conf_threshold:.2f}",
            f"nms={args.person_nms_threshold:.2f}",
            f"interval={args.person_interval}",
            f"margin={args.person_class_margin:.2f}",
            f"confirm={args.person_confirmation_hits}x@{args.person_confirmation_threshold:.2f}",
            f"instant={args.person_instant_threshold:.2f}",
            f"npu_schedule={'concurrent' if args.rknn_allow_concurrent_inference else 'serialized'}",
            flush=True,
        )
    if unicode_font is None:
        print("warning: 未找到可用中文字体或 Pillow，车牌中文上屏会退化为 ASCII。", flush=True)

    fps_counter = 0
    total_frames = 0
    fps_time = time.time()
    fps_value = 0.0
    last_status = startup_status
    camera_failures = 0
    last_threshold = -1
    last_box_count = 0
    last_plate_count = 0
    last_plate_text = ""
    last_plate_summary = ""
    last_person_count = 0
    cached_detections: List[Detection] = []
    display_detections: List[Detection] = []
    status_detections: List[Detection] = []
    cached_people: List[Detection] = []
    display_people: List[Detection] = []
    last_drawn_completed_generation = 0
    last_detector_mode = "idle"
    display_detector_mode = "idle"
    status_detector_mode = "idle"
    detector_busy = False
    detector_latest_update_time = 0.0
    detector_duration_ms = 0.0
    last_detector_submit_time = 0.0
    fpga_busy = False
    fpga_duration_ms = 0.0
    box_tracker = None if args.disable_box_tracking else BoxMotionTracker()
    person_tracker = (
        BoxMotionTracker()
        if person_runner is not None and not args.disable_box_tracking
        else None
    )
    person_busy = False
    person_completed_generation = 0
    person_latest_update_time = 0.0
    person_duration_ms = 0.0
    source_description = describe_frame_source(cap, args)
    metrics_recorder = (
        MetricsRecorder(
            args.metrics_dir,
            source=source_description,
            arguments=vars(args),
            ground_truth_path=args.metrics_ground_truth,
            iou_threshold=args.metrics_iou_threshold,
            warmup_frames=args.metrics_warmup_frames,
        )
        if args.metrics_dir
        else None
    )
    if metrics_recorder is not None:
        print(f"Metrics enabled: output={metrics_recorder.output_dir}", flush=True)
    last_metrics_detector_generation = 0
    last_metrics_fpga_generation = 0
    last_metrics_person_generation = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if args.input_image:
                    raise RuntimeError(f"读取输入图片失败: {args.input_image}")
                if args.input_video and getattr(cap, "ended", False):
                    print(f"Input video finished: {args.input_video}", flush=True)
                    break
                camera_failures += 1
                print(
                    f"摄像头读帧失败，开始重试 {camera_failures}/{args.camera_read_retries} ...",
                    flush=True,
                )
                cap.release()
                time.sleep(max(args.camera_retry_delay, 0.0))
                cap = open_frame_source(args)

                if cap.isOpened() and camera_failures <= args.camera_read_retries:
                    continue

                raise RuntimeError(f"读取摄像头帧失败，已连续重试 {camera_failures} 次仍未恢复")

            if camera_failures:
                print(f"摄像头流已恢复，之前连续失败 {camera_failures} 次。", flush=True)
                camera_failures = 0

            source_frame_index, source_time_s, source_name = get_source_frame_info(
                cap, total_frames + 1
            )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if person_runner is not None and (
                total_frames == 0 or total_frames % args.person_interval == 0
            ):
                person_runner.submit(frame)
            if person_runner is not None:
                (
                    cached_people,
                    person_busy,
                    _,
                    person_completed_generation,
                    person_latest_update_time,
                    person_duration_ms,
                ) = person_runner.snapshot()
                tracked_people = (
                    person_tracker.update(gray, cached_people, person_completed_generation)
                    if person_tracker is not None
                    else list(cached_people)
                )
                person_result_age = time.monotonic() - person_latest_update_time
                display_people = (
                    list(tracked_people)
                    if tracked_people and person_result_age <= args.person_hold_seconds
                    else []
                )
            fpga_gray = cv2.resize(gray, (args.fpga_width, args.fpga_height), interpolation=cv2.INTER_AREA)
            fpga_runner.submit(fpga_gray)
            (
                status,
                mask_small,
                candidate_boxes_small,
                current_threshold,
                fpga_busy,
                _,
                fpga_completed_generation,
                fpga_duration_ms,
            ) = fpga_runner.snapshot()
            last_threshold = current_threshold
            last_status = status
            last_box_count = len(candidate_boxes_small)

            sx = frame.shape[1] / max(mask_small.shape[1], 1)
            sy = frame.shape[0] / max(mask_small.shape[0], 1)
            candidate_boxes_full = [scale_box(box, sx, sy) for box in candidate_boxes_small]
            mask_full = (
                cv2.resize(mask_small, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                if args.display_mode in ("overlay", "mask")
                else None
            )

            display = compose_display_frame(
                frame,
                mask_full,
                args.display_mode,
                args.overlay_alpha,
            )

            if args.draw_roi:
                draw_candidate_boxes(display, candidate_boxes_full)

            if args.draw_roi and not candidate_boxes_full:
                draw_text_with_outline(
                    display,
                    "FPGA ROI: none",
                    (20, 40),
                    font_scale=0.8,
                    color=(0, 220, 255),
                    thickness=2,
                )

            if args.detector_accuracy_priority:
                detector_interval_now = 1
            else:
                should_slow_down_detection = (
                    bool(cached_detections)
                    and not str(last_detector_mode).endswith("_hold")
                    and last_box_count <= max(1, last_plate_count)
                )
                detector_interval_now = (
                    args.detector_interval_hit
                    if should_slow_down_detection
                    else args.detector_interval
                )
            now_submit = time.monotonic()
            submit_due_by_frame = (total_frames == 0 or total_frames % detector_interval_now == 0)
            submit_due_by_time = (
                args.detector_submit_max_gap_seconds > 0
                and (now_submit - last_detector_submit_time) >= args.detector_submit_max_gap_seconds
            )
            if detector_runner is not None and (submit_due_by_frame or submit_due_by_time):
                detector_runner.submit(
                    frame,
                    gray,
                    candidate_boxes_full,
                    source_frame=source_frame_index,
                    source_time_s=source_time_s,
                    source_name=source_name,
                )
                last_detector_submit_time = now_submit

            if detector_runner is not None:
                (
                    cached_detections,
                    last_detector_mode,
                    detector_busy,
                    detector_submit_generation,
                    detector_completed_generation,
                    detector_latest_update_time,
                    detector_duration_ms,
                    detector_source_gray,
                ) = detector_runner.snapshot()
                (
                    metric_detector_result,
                    metric_detector_generation,
                    metric_detector_source_frame,
                    metric_detector_source_time,
                    metric_detector_source_name,
                    metric_detector_duration_ms,
                    _,
                ) = detector_runner.metrics_snapshot()
            else:
                detector_busy = False
                detector_submit_generation = 0
                detector_completed_generation = 0
                detector_latest_update_time = 0.0
                detector_duration_ms = 0.0
                detector_source_gray = None
                metric_detector_result = []
                metric_detector_generation = 0
                metric_detector_source_frame = 0
                metric_detector_source_time = 0.0
                metric_detector_source_name = ""
                metric_detector_duration_ms = 0.0

            tracked_detections = (
                box_tracker.update(
                    gray,
                    cached_detections,
                    detector_completed_generation,
                    detection_gray=(
                        None
                        if args.disable_detection_lag_compensation
                        else detector_source_gray
                    ),
                )
                if box_tracker is not None
                else list(cached_detections)
            )

            if args.box_display_mode == "hold":
                hold_age = time.monotonic() - detector_latest_update_time
                display_detections = (
                    list(tracked_detections)
                    if tracked_detections and hold_age <= max(args.box_hold_seconds, 0.0)
                    else []
                )
            elif tracked_detections and detector_completed_generation > last_drawn_completed_generation:
                display_detections = list(tracked_detections)
                last_drawn_completed_generation = detector_completed_generation
            else:
                display_detections = []
            display_detector_mode = last_detector_mode
            if cached_detections:
                status_detections = list(cached_detections)
                status_detector_mode = last_detector_mode

            unicode_items = []
            for det in display_detections:
                gx1, gy1, dw, dh = det.box
                gx2, gy2 = gx1 + dw, gy1 + dh
                cv2.rectangle(display, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
                unicode_items.append(
                    {
                        "text": det.raw_label or det.label,
                        "origin": (gx1, max(8, gy1 - args.text_font_size - 6)),
                        "color": (0, 0, 255),
                        "outline_color": (0, 0, 0),
                        "outline_thickness": 2,
                    }
                )

            for det in display_people:
                gx1, gy1, dw, dh = det.box
                gx2, gy2 = gx1 + dw, gy1 + dh
                cv2.rectangle(display, (gx1, gy1), (gx2, gy2), (0, 220, 80), 2)
                unicode_items.append(
                    {
                        "text": det.full_text or f"行人 {det.score:.0%}",
                        "origin": (gx1, max(8, gy1 - args.text_font_size - 6)),
                        "color": (0, 220, 80),
                        "outline_color": (0, 0, 0),
                        "outline_thickness": 2,
                    }
                )

            last_plate_count = len(status_detections)
            last_person_count = len(cached_people)
            last_plate_text = status_detections[0].raw_label if status_detections else ""
            last_plate_summary = build_plate_summary(status_detections)
            summary_chars_per_line = max(18, int((display.shape[1] - 40) / max(args.text_font_size, 1) * 1.3))
            summary_lines = build_plate_summary_lines(status_detections, summary_chars_per_line)

            total_frames += 1
            fps_counter += 1
            if fps_counter >= 10:
                now = time.time()
                fps_value = fps_counter / max(now - fps_time, 1e-6)
                fps_time = now
                fps_counter = 0

            detector_updated = (
                metric_detector_generation > 0
                and metric_detector_generation != last_metrics_detector_generation
            )
            fpga_updated = (
                fpga_completed_generation > 0
                and fpga_completed_generation != last_metrics_fpga_generation
            )
            person_updated = (
                person_completed_generation > 0
                and person_completed_generation != last_metrics_person_generation
            )
            if metrics_recorder is not None:
                metrics_recorder.record_frame(
                    processed_frame=total_frames,
                    source_frame=source_frame_index,
                    source_time_s=source_time_s,
                    source_name=source_name,
                    real_fps=fps_value,
                    threshold=last_threshold,
                    candidate_boxes=last_box_count,
                    plates=display_detections,
                    people=display_people,
                    detector_mode=status_detector_mode,
                    detector_busy=detector_busy,
                    fpga_busy=fpga_busy,
                    fpga_ms=fpga_duration_ms,
                    detector_ms=(
                        metric_detector_duration_ms
                        if detector_updated
                        else detector_duration_ms
                    ),
                    person_ms=person_duration_ms,
                    detector_generation=metric_detector_generation,
                    detector_updated=detector_updated,
                    detector_result=metric_detector_result,
                    detector_source_frame=metric_detector_source_frame,
                    detector_source_time_s=metric_detector_source_time,
                    detector_source_name=metric_detector_source_name,
                    fpga_updated=fpga_updated,
                    person_updated=person_updated,
                    active_pixels=status.active_pixels,
                )
            if detector_updated:
                last_metrics_detector_generation = metric_detector_generation
            if fpga_updated:
                last_metrics_fpga_generation = fpga_completed_generation
            if person_updated:
                last_metrics_person_generation = person_completed_generation

            display_fps_value = fps_value * 2.0
            person_status_text = (
                f"person={last_person_count} "
                if person_runner is not None
                else ""
            )
            if not args.hide_status:
                info = (
                    f"FPS:{display_fps_value:.1f} "
                    f"thr={last_threshold} "
                    f"box={last_box_count} "
                    f"plate={last_plate_count} "
                    f"{person_status_text}"
                    f"mode={status_detector_mode} "
                    f"det={int(detector_busy)} "
                    f"active={status.active_pixels} "
                    f"frame={status.frame_counter}"
                )
                draw_text_with_outline(
                    display,
                    info,
                    (20, display.shape[0] - 20),
                    font_scale=0.72,
                    color=(255, 255, 255),
                    thickness=2,
                )
                for line_index, line_text in enumerate(reversed(summary_lines)):
                    unicode_items.append(
                        {
                            "text": line_text,
                            "origin": (
                                20,
                                max(8, display.shape[0] - args.text_font_size - 52 - line_index * (args.text_font_size + 10)),
                            ),
                            "color": (0, 255, 0),
                            "outline_color": (0, 0, 0),
                            "outline_thickness": 2,
                        }
                    )

            draw_unicode_texts(display, unicode_items, unicode_font)

            if output_dir is not None and args.save_every > 0 and (total_frames == 1 or total_frames % args.save_every == 0):
                debug_mask = mask_full
                if debug_mask is None:
                    debug_mask = cv2.resize(
                        mask_small,
                        (frame.shape[1], frame.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                save_debug_images(output_dir, total_frames, frame, debug_mask, display)
                print(f"saved debug images at frame {total_frames} -> {output_dir}", flush=True)

            if args.log_every > 0 and (total_frames == 1 or total_frames % args.log_every == 0):
                person_log_text = (
                    f"persons={last_person_count} person_det={int(person_busy)} "
                    f"person_ms={person_duration_ms:.1f} "
                    if person_runner is not None
                    else ""
                )
                print(
                    f"frame={total_frames} fps={display_fps_value:.1f} real_fps={fps_value:.1f} "
                    f"threshold={last_threshold} boxes={last_box_count} plates={last_plate_count} "
                    f"{person_log_text}"
                    f"mode={status_detector_mode} det={int(detector_busy)} "
                    f"busy={int(status.busy)} done={int(status.done)} fpga_async={int(fpga_busy)} "
                    f"fpga_ms={fpga_duration_ms:.1f} det_ms={detector_duration_ms:.1f} "
                    f"active={status.active_pixels} counter={status.frame_counter} "
                    f"last_plate={last_plate_text or 'none'} "
                    f"summary={last_plate_summary or 'none'}",
                    flush=True,
                )

            if not args.headless:
                screen_frame = resize_for_display(display, args.display_width, args.display_height)
                cv2.imshow(args.window_name, screen_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            if args.max_frames > 0 and total_frames >= args.max_frames:
                break
    finally:
        metrics_summary = None
        if metrics_recorder is not None:
            try:
                metrics_summary = metrics_recorder.close()
            except Exception as exc:
                print(f"写入自动测试指标失败: {exc}", flush=True)
        cap.release()
        fpga_runner.close()
        if detector_runner is not None:
            detector_runner.close()
        detector.close()
        if person_runner is not None:
            person_runner.close()
        if person_detector is not None:
            person_detector.close()
        fpga.close()
        cv2.destroyAllWindows()
        person_final_text = (
            f"last_person_count={last_person_count} "
            if person_runner is not None
            else ""
        )
        print(
            f"pipeline finished: frames={total_frames} "
            f"last_threshold={last_threshold} "
            f"last_boxes={last_box_count} "
            f"last_plate_count={last_plate_count} "
            f"{person_final_text}"
            f"last_plate={last_plate_text or 'none'} "
            f"last_plate_summary={last_plate_summary or 'none'} "
            f"last_mode={display_detector_mode} "
            f"last_active={last_status.active_pixels} "
            f"last_counter={last_status.frame_counter}",
            flush=True,
        )
        if metrics_summary is not None:
            runtime_summary = metrics_summary.get("runtime", {})
            reported_throughput = runtime_summary.get(
                "measured_throughput_fps",
                runtime_summary.get("throughput_fps", 0.0),
            )
            print(
                "metrics finished:",
                f"output={metrics_recorder.output_dir}",
                f"frames={runtime_summary.get('processed_frames', 0)}",
                f"throughput_fps={reported_throughput:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
