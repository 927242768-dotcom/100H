from __future__ import annotations

import ctypes
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2

import numpy as np


_RKNN_INFERENCE_LOCK = threading.Lock()


@dataclass
class Detection:
    label: str
    score: float
    box: Tuple[int, int, int, int]
    raw_label: str = ""
    type_name: str = ""
    full_text: str = ""


class BaseDetector:
    def detect(self, image) -> List[Detection]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class MockDetector(BaseDetector):
    def detect(self, image) -> List[Detection]:
        return []


class _HLPRPlateResult(ctypes.Structure):
    _fields_ = [
        ("x1", ctypes.c_float),
        ("y1", ctypes.c_float),
        ("x2", ctypes.c_float),
        ("y2", ctypes.c_float),
        ("type", ctypes.c_int),
        ("text_confidence", ctypes.c_float),
        ("code", ctypes.c_char * 128),
    ]


class _HLPRPlateResultList(ctypes.Structure):
    _fields_ = [
        ("plate_size", ctypes.c_ulong),
        ("plates", ctypes.POINTER(_HLPRPlateResult)),
    ]


class _HLPRContextConfiguration(ctypes.Structure):
    _fields_ = [
        ("models_path", ctypes.c_char_p),
        ("max_num", ctypes.c_int),
        ("threads", ctypes.c_int),
        ("use_half", ctypes.c_bool),
        ("box_conf_threshold", ctypes.c_float),
        ("nms_threshold", ctypes.c_float),
        ("rec_confidence_threshold", ctypes.c_float),
        ("det_level", ctypes.c_int),
    ]


class HyperLprDetector(BaseDetector):
    _IMAGE_FORMAT_BGR = 1
    _ROTATION_0 = 0
    _DETECT_LEVEL_LOW = 0

    _DEFAULT_HYPERLPR_LIBS = (
        "/userdata/HyperLPR/HyperLPR/Prj-Linux/hyperlpr3/lib/libhyperlpr3.so",
        "/userdata/HyperLPR/HyperLPR/build/linux/install/hyperlpr3/lib/libhyperlpr3.so",
    )
    _DEFAULT_MODEL_DIRS = (
        "/userdata/HyperLPR/HyperLPR/Prj-Linux/hyperlpr3/resource/models/r2_mobile",
        "/userdata/HyperLPR/HyperLPR/build/linux/install/hyperlpr3/resource/models/r2_mobile",
    )
    _DEFAULT_MNN_LIBS = (
        "/userdata/HyperLPR/HyperLPR/3rdparty_hyper_inspire_op/MNN-2.2.0/linux/lib/libMNN.so",
        "/userdata/MNN/build-linux/libMNN.so",
    )

    _PROVINCE_MAP = {
        "京": "BJ",
        "津": "TJ",
        "沪": "SH",
        "渝": "CQ",
        "冀": "JI",
        "晋": "SX",
        "蒙": "NM",
        "辽": "LN",
        "吉": "JL",
        "黑": "HLJ",
        "苏": "SU",
        "浙": "ZJ",
        "皖": "WAN",
        "闽": "MIN",
        "赣": "GAN",
        "鲁": "LU",
        "豫": "YU",
        "鄂": "E",
        "湘": "XIANG",
        "粤": "YUE",
        "桂": "GX",
        "琼": "QIONG",
        "川": "CHUAN",
        "贵": "GUI",
        "云": "YN",
        "藏": "XZ",
        "陕": "SHAAN",
        "甘": "GAN",
        "青": "QH",
        "宁": "NX",
        "新": "XJ",
        "港": "HK",
        "澳": "MO",
    }
    _PLATE_TYPE_MAP = {
        0: "蓝牌",
        1: "绿牌新能源",
        2: "黄牌",
        3: "白色警用",
        4: "港澳车牌",
        5: "教练牌",
        6: "武警牌",
        7: "双层黄牌",
        8: "双层武警",
        9: "双层军牌",
        10: "使馆车牌",
        11: "领馆车牌",
        12: "民航牌",
        13: "新能源大型车",
        14: "新能源小型车",
    }

    def __init__(
        self,
        *,
        hyperlpr_lib: str = "",
        model_dir: str = "",
        mnn_lib: str = "",
        max_num: int = 3,
        threads: int = 1,
        use_half: bool = True,
        box_conf_threshold: float = 0.30,
        nms_threshold: float = 0.45,
        rec_confidence_threshold: float = 0.50,
    ) -> None:
        self._hyperlpr_lib_path = self._resolve_existing_path(
            hyperlpr_lib,
            self._DEFAULT_HYPERLPR_LIBS,
            description="HyperLPR 动态库",
            require_dir=False,
        )
        self._model_dir_path = self._resolve_existing_path(
            model_dir,
            self._DEFAULT_MODEL_DIRS,
            description="HyperLPR 模型目录",
            require_dir=True,
        )
        self._mnn_lib_path = self._resolve_existing_path(
            mnn_lib,
            self._DEFAULT_MNN_LIBS,
            description="MNN 动态库",
            require_dir=False,
        )

        self._mnn_handle = None
        self._lib = None
        self._context = None
        self._buffer = None
        self._model_dir_bytes = str(self._model_dir_path).encode("utf-8")
        self._set_stream_format = None
        self._set_stream_rotation = None

        self._load_libraries()
        self._configure_ffi()

        config = _HLPRContextConfiguration(
            models_path=self._model_dir_bytes,
            max_num=max(1, int(max_num)),
            threads=max(1, int(threads)),
            use_half=bool(use_half),
            box_conf_threshold=float(box_conf_threshold),
            nms_threshold=float(nms_threshold),
            rec_confidence_threshold=float(rec_confidence_threshold),
            det_level=self._DETECT_LEVEL_LOW,
        )

        self._context = self._lib.HLPR_CreateContext(ctypes.byref(config))
        if not self._context:
            raise RuntimeError("创建 HyperLPR 上下文失败")

        status = self._lib.HLPR_ContextQueryStatus(self._context)
        if int(status) != 0:
            raise RuntimeError(f"HyperLPR 上下文初始化失败，状态码={int(status)}")

        self._buffer = self._lib.HLPR_CreateDataBufferEmpty()
        if not self._buffer:
            self.close()
            raise RuntimeError("创建 HyperLPR 数据缓冲区失败")

        if self._set_stream_format is not None:
            self._check_status(
                "设置图像格式",
                self._set_stream_format(self._buffer, self._IMAGE_FORMAT_BGR),
            )
        if self._set_stream_rotation is not None:
            self._check_status(
                "设置图像旋转",
                self._set_stream_rotation(self._buffer, self._ROTATION_0),
            )

    @staticmethod
    def _resolve_existing_path(
        preferred_path: str,
        fallback_paths: Sequence[str],
        *,
        description: str,
        require_dir: bool,
    ) -> Path:
        candidates = [preferred_path] if preferred_path else []
        candidates.extend(fallback_paths)

        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate).expanduser()
            if require_dir and path.is_dir():
                return path
            if not require_dir and path.is_file():
                return path

        expected = "目录" if require_dir else "文件"
        tried = "\n".join(str(Path(item).expanduser()) for item in candidates if item)
        raise FileNotFoundError(f"未找到可用的 {description}{expected}，已尝试：\n{tried}")

    def _load_libraries(self) -> None:
        load_mode = getattr(ctypes, "RTLD_GLOBAL", 0)
        self._mnn_handle = ctypes.CDLL(str(self._mnn_lib_path), mode=load_mode)
        self._lib = ctypes.CDLL(str(self._hyperlpr_lib_path), mode=load_mode)

    def _optional_function(self, name: str, argtypes, restype):
        try:
            func = getattr(self._lib, name)
        except AttributeError:
            return None
        func.argtypes = argtypes
        func.restype = restype
        return func

    def _configure_ffi(self) -> None:
        self._lib.HLPR_CreateContext.argtypes = [ctypes.POINTER(_HLPRContextConfiguration)]
        self._lib.HLPR_CreateContext.restype = ctypes.c_void_p

        self._lib.HLPR_ContextQueryStatus.argtypes = [ctypes.c_void_p]
        self._lib.HLPR_ContextQueryStatus.restype = ctypes.c_int

        self._lib.HLPR_ReleaseContext.argtypes = [ctypes.c_void_p]
        self._lib.HLPR_ReleaseContext.restype = ctypes.c_int

        self._lib.HLPR_CreateDataBufferEmpty.argtypes = []
        self._lib.HLPR_CreateDataBufferEmpty.restype = ctypes.c_void_p

        self._lib.HLPR_DataBufferSetData.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
        ]
        self._lib.HLPR_DataBufferSetData.restype = ctypes.c_int

        self._set_stream_format = self._optional_function(
            "HLPR_DataBufferSetStreamFormat",
            [ctypes.c_void_p, ctypes.c_int],
            ctypes.c_int,
        )
        self._set_stream_rotation = self._optional_function(
            "HLPR_DataBufferSetStreamRotation",
            [ctypes.c_void_p, ctypes.c_int],
            ctypes.c_int,
        )

        self._lib.HLPR_ReleaseDataBuffer.argtypes = [ctypes.c_void_p]
        self._lib.HLPR_ReleaseDataBuffer.restype = ctypes.c_int

        self._lib.HLPR_ContextUpdateStream.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(_HLPRPlateResultList),
        ]
        self._lib.HLPR_ContextUpdateStream.restype = ctypes.c_int

    @staticmethod
    def _check_status(action: str, status: int) -> None:
        if int(status) != 0:
            raise RuntimeError(f"{action}失败，状态码={int(status)}")

    def _format_plate_for_display(self, plate_text: str) -> str:
        if not plate_text:
            return "PLATE"

        first_char = plate_text[0]
        prefix = self._PROVINCE_MAP.get(first_char, first_char if first_char.isascii() else "CN")
        suffix = "".join(ch for ch in plate_text[1:] if ch.isascii()).upper()
        return f"{prefix}-{suffix}" if suffix else prefix

    @staticmethod
    def _normalize_plate_text(plate_text: str) -> str:
        normalized = plate_text.replace("O", "0").replace("o", "0")
        hang_index = normalized.find("航")
        if hang_index > 0:
            normalized = normalized[:hang_index - 1] + "民" + normalized[hang_index:]
        return normalized

    def _normalize_type_name(self, plate_type: int, plate_text: str) -> str:
        type_name = self._PLATE_TYPE_MAP.get(int(plate_type), f"类型{int(plate_type)}")
        if not plate_text:
            return type_name

        normalized = plate_text.strip().upper()
        if "航" in plate_text:
            return "民航车牌"
        if normalized.startswith("WJ") or "武警" in plate_text:
            return "武警车牌"
        if "警" in plate_text:
            return "白色警用车牌"
        if "学" in plate_text:
            return "教练车牌"
        if "使" in plate_text or "领" in plate_text:
            return "使馆车牌"
        if "粤Z" in plate_text or "粤Ｚ" in plate_text or "港" in plate_text or "澳" in plate_text:
            return "港澳粤Z牌"

        # 新能源车牌常见特征：
        # 1. 总长度通常为 8
        # 2. 小型新能源第三位常见 D/F
        # 3. 大型新能源末位常见 D/F
        if len(normalized) >= 8:
            body = normalized[1:]
            if len(body) >= 2 and body[1] in ("D", "F"):
                return "新能源车牌"
            if body and body[-1] in ("D", "F"):
                return "新能源车牌"

        return type_name

    def _format_full_text(self, type_name: str, plate_text: str) -> str:
        if plate_text:
            return f"{type_name}, {plate_text}"
        return type_name

    def detect(self, image) -> List[Detection]:
        if image is None:
            return []

        frame = np.asarray(image)
        if frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3:
            return []

        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        height, width = frame.shape[:2]
        data_ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        self._check_status(
            "写入 HyperLPR 输入图像",
            self._lib.HLPR_DataBufferSetData(self._buffer, data_ptr, width, height),
        )

        result_list = _HLPRPlateResultList()
        self._check_status(
            "执行 HyperLPR 推理",
            self._lib.HLPR_ContextUpdateStream(self._context, self._buffer, ctypes.byref(result_list)),
        )

        if not result_list.plates or int(result_list.plate_size) <= 0:
            return []

        detections: List[Detection] = []
        for index in range(int(result_list.plate_size)):
            plate = result_list.plates[index]
            raw_label = bytes(plate.code).split(b"\0", 1)[0].decode("utf-8", errors="ignore").strip()
            raw_label = self._normalize_plate_text(raw_label)
            if not raw_label:
                continue

            x1 = max(0, min(width - 1, int(round(float(plate.x1)))))
            y1 = max(0, min(height - 1, int(round(float(plate.y1)))))
            x2 = max(x1 + 1, min(width, int(round(float(plate.x2)))))
            y2 = max(y1 + 1, min(height, int(round(float(plate.y2)))))

            type_name = self._normalize_type_name(int(plate.type), raw_label)

            detections.append(
                Detection(
                    label=raw_label,
                    raw_label=raw_label,
                    type_name=type_name,
                    full_text=self._format_full_text(type_name, raw_label),
                    score=float(plate.text_confidence),
                    box=(x1, y1, x2 - x1, y2 - y1),
                )
            )

        return detections

    def close(self) -> None:
        if self._buffer is not None and self._lib is not None:
            try:
                self._lib.HLPR_ReleaseDataBuffer(self._buffer)
            finally:
                self._buffer = None

        if self._context is not None and self._lib is not None:
            try:
                self._lib.HLPR_ReleaseContext(self._context)
            finally:
                self._context = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class RknnLiteDetector(BaseDetector):
    _DEFAULT_MODEL_PATHS = (
        "/userdata/yolov8-plate/yolov8s.rknn",
        "/userdata/yolov8-plate/weights/yolov8s.rknn",
        "/home/linaro/yolov8-plate/yolov8s.rknn",
        "/home/linaro/yolov8-plate/weights/yolov8s.rknn",
        "/userdata/models/yolov8_plate.rknn",
    )

    def __init__(
        self,
        *,
        model_path: str = "",
        labels: Sequence[str] = ("单层车牌", "双层车牌"),
        input_size: int = 640,
        conf_threshold: float = 0.20,
        nms_threshold: float = 0.45,
        core_mask: str = "auto",
        recognizer: BaseDetector | None = None,
        ocr_cache_seconds: float = 2.0,
        ocr_cache_iou: float = 0.50,
        class_filter: Sequence[int] | None = None,
        serialize_inference: bool = True,
        refine_box_from_recognizer: bool = True,
        class_margin_threshold: float = 0.0,
    ) -> None:
        try:
            from rknnlite.api import RKNNLite
        except Exception:
            self._inject_local_rknnlite_paths()
            try:
                from rknnlite.api import RKNNLite
            except Exception as exc:
                raise RuntimeError(
                    "未找到 rknnlite 运行时。请先把 rknn_toolkit_lite2 wheel 解压到工程目录下的 vendor/，"
                    "或在板子上安装 RKNNLite，再使用 --detector rknn。"
                ) from exc

        self._RKNNLite = RKNNLite
        self._model_path = self._resolve_existing_path(
            model_path,
            self._DEFAULT_MODEL_PATHS,
            description="RKNN 模型",
            require_dir=False,
        )
        self._labels = [label.strip() for label in labels if label and label.strip()]
        if not self._labels:
            self._labels = ["单层车牌", "双层车牌"]
        self._num_classes = len(self._labels)
        self._input_size = max(32, int(input_size))
        self._letterbox_canvas = np.empty(
            (self._input_size, self._input_size, 3),
            dtype=np.uint8,
        )
        self._rgb_input = np.empty_like(self._letterbox_canvas)
        self._input_tensor = np.empty(
            (1, self._input_size, self._input_size, 3),
            dtype=np.float32,
        )
        self._conf_threshold = float(conf_threshold)
        self._nms_threshold = float(nms_threshold)
        self._class_filter = (
            np.asarray(sorted(set(int(item) for item in class_filter)), dtype=np.int64)
            if class_filter
            else None
        )
        self._class_margin_threshold = max(0.0, float(class_margin_threshold))
        self._serialize_inference = bool(serialize_inference)
        self._recognizer = recognizer
        self._refine_box_from_recognizer = bool(refine_box_from_recognizer)
        self._ocr_cache_seconds = max(0.0, float(ocr_cache_seconds))
        self._ocr_cache_iou = min(1.0, max(0.0, float(ocr_cache_iou)))
        self._ocr_cache: List[
            Tuple[
                int,
                int,
                Tuple[int, int, int, int],
                float,
                Detection,
                Tuple[float, float, float, float] | None,
            ]
        ] = []
        self._rknn = self._RKNNLite()

        load_status = self._rknn.load_rknn(str(self._model_path))
        if int(load_status) != 0:
            raise RuntimeError(f"加载 RKNN 模型失败，状态码={int(load_status)}，路径={self._model_path}")

        runtime_kwargs = {}
        resolved_core_mask = self._resolve_core_mask(core_mask)
        if resolved_core_mask is not None:
            runtime_kwargs["core_mask"] = resolved_core_mask

        init_status = self._rknn.init_runtime(**runtime_kwargs)
        if int(init_status) != 0:
            raise RuntimeError(f"初始化 RKNN 运行时失败，状态码={int(init_status)}")

    @staticmethod
    def _inject_local_rknnlite_paths() -> None:
        base_dir = Path(__file__).resolve().parent
        candidates = (
            base_dir / "vendor",
            base_dir / "vendor" / "python",
            base_dir / "third_party",
            base_dir / "third_party" / "python",
        )
        for candidate in candidates:
            candidate_str = str(candidate)
            if candidate.is_dir() and candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)

    @staticmethod
    def _resolve_existing_path(
        preferred_path: str,
        fallback_paths: Sequence[str],
        *,
        description: str,
        require_dir: bool,
    ) -> Path:
        candidates = [preferred_path] if preferred_path else []
        candidates.extend(fallback_paths)

        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate).expanduser()
            if require_dir and path.is_dir():
                return path
            if not require_dir and path.is_file():
                return path

        expected = "目录" if require_dir else "文件"
        tried = "\n".join(str(Path(item).expanduser()) for item in candidates if item)
        raise FileNotFoundError(f"未找到可用的 {description}{expected}，已尝试：\n{tried}")

    def _resolve_core_mask(self, core_mask: str):
        if not core_mask:
            return None

        normalized = str(core_mask).strip().lower()
        if normalized in ("", "auto"):
            return getattr(self._RKNNLite, "NPU_CORE_AUTO", None)

        mapping = {
            "0": "NPU_CORE_0",
            "1": "NPU_CORE_1",
            "2": "NPU_CORE_2",
            "0_1": "NPU_CORE_0_1",
            "0_1_2": "NPU_CORE_0_1_2",
        }
        attr_name = mapping.get(normalized)
        if not attr_name:
            return None
        return getattr(self._RKNNLite, attr_name, None)

    def _letterbox(self, image: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        height, width = image.shape[:2]
        scale = min(self._input_size / max(height, 1), self._input_size / max(width, 1))
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))

        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        canvas = self._letterbox_canvas
        canvas.fill(114)
        pad_x = (self._input_size - new_width) // 2
        pad_y = (self._input_size - new_height) // 2
        canvas[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized
        return canvas, scale, pad_x, pad_y

    def _prepare_input(self, image: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        letterboxed, scale, pad_x, pad_y = self._letterbox(image)
        cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB, dst=self._rgb_input)
        np.copyto(self._input_tensor[0], self._rgb_input, casting="unsafe")
        self._input_tensor[0] /= 255.0
        return self._input_tensor, scale, pad_x, pad_y

    def _select_prediction_array(self, outputs) -> np.ndarray | None:
        arrays: List[np.ndarray] = []
        for output in outputs:
            array = np.asarray(output)
            if array.size <= 0:
                continue
            arrays.append(array)

        if not arrays:
            return None

        prediction = max(arrays, key=lambda item: item.size)
        while prediction.ndim >= 3 and prediction.shape[0] == 1:
            prediction = prediction[0]

        if prediction.ndim != 2:
            prediction = prediction.reshape(prediction.shape[-2], prediction.shape[-1])

        attr_min = 4 + self._num_classes
        if prediction.shape[1] < attr_min <= prediction.shape[0]:
            prediction = prediction.T
        elif prediction.shape[0] <= (attr_min + 16) and prediction.shape[1] > prediction.shape[0]:
            prediction = prediction.T

        if prediction.shape[1] < attr_min:
            return None
        return prediction.astype(np.float32, copy=False)

    @staticmethod
    def _xywh_to_xyxy(boxes_xywh: np.ndarray) -> np.ndarray:
        boxes = boxes_xywh.copy()
        boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
        boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
        boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
        boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0
        return boxes

    def _restore_boxes(
        self,
        boxes_xyxy: np.ndarray,
        *,
        scale: float,
        pad_x: int,
        pad_y: int,
        image_width: int,
        image_height: int,
    ) -> np.ndarray:
        boxes = boxes_xyxy.copy()
        boxes[:, [0, 2]] -= pad_x
        boxes[:, [1, 3]] -= pad_y
        boxes /= max(scale, 1e-6)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_width - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_height - 1)
        return boxes

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        if len(boxes) == 0:
            return []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = np.maximum(1.0, (x2 - x1) * (y2 - y1))
        order = scores.argsort()[::-1]
        keep: List[int] = []

        while order.size > 0:
            current = int(order[0])
            keep.append(current)
            if order.size == 1:
                break

            rest = order[1:]
            xx1 = np.maximum(x1[current], x1[rest])
            yy1 = np.maximum(y1[current], y1[rest])
            xx2 = np.minimum(x2[current], x2[rest])
            yy2 = np.minimum(y2[current], y2[rest])
            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            inter = inter_w * inter_h
            union = areas[current] + areas[rest] - inter
            iou = np.divide(inter, np.maximum(union, 1e-6))
            order = rest[iou <= threshold]

        return keep

    def _recognize_crop(self, crop: np.ndarray) -> Detection | None:
        if self._recognizer is None or crop.size == 0:
            return None

        try:
            detections = self._recognizer.detect(crop)
        except Exception:
            return None

        if not detections:
            return None

        detections.sort(
            key=lambda det: (
                det.score,
                len(det.raw_label or det.label or ""),
            ),
            reverse=True,
        )
        return detections[0]

    def _recognize_box(
        self,
        frame: np.ndarray,
        box: Tuple[int, int, int, int],
        image_width: int,
        image_height: int,
    ) -> Tuple[Detection | None, Tuple[int, int, int, int] | None]:
        if self._recognizer is None:
            return None, None

        cached = self._find_cached_recognition(
            box,
            image_width=image_width,
            image_height=image_height,
        )
        if cached is not None:
            return cached

        ox1, oy1, ox2, oy2 = self._expand_box_for_ocr(
            box,
            image_width,
            image_height,
            ratio=0.25,
        )
        crop = frame[oy1:oy2, ox1:ox2]
        recognized = self._recognize_crop(crop)
        refined_box = None
        if recognized is not None:
            if self._refine_box_from_recognizer:
                refined_box = self._map_recognized_box(
                    box,
                    recognized.box,
                    crop_origin=(ox1, oy1),
                    image_width=image_width,
                    image_height=image_height,
                )
            self._store_cached_recognition(
                box,
                recognized,
                refined_box=refined_box,
                image_width=image_width,
                image_height=image_height,
            )
        return recognized, refined_box

    @classmethod
    def _map_recognized_box(
        cls,
        detector_box: Tuple[int, int, int, int],
        recognized_box: Tuple[int, int, int, int],
        *,
        crop_origin: Tuple[int, int],
        image_width: int,
        image_height: int,
    ) -> Tuple[int, int, int, int] | None:
        rx, ry, rw, rh = recognized_box
        if rw < 2 or rh < 2:
            return None

        ox, oy = crop_origin
        raw_x1 = ox + rx
        raw_y1 = oy + ry
        x1 = max(0, min(image_width - 1, raw_x1))
        y1 = max(0, min(image_height - 1, raw_y1))
        x2 = max(x1 + 1, min(image_width, raw_x1 + rw))
        y2 = max(y1 + 1, min(image_height, raw_y1 + rh))

        dx1, dy1, dx2, dy2 = detector_box
        detector_width = max(1, dx2 - dx1)
        detector_height = max(1, dy2 - dy1)
        detector_xyxy = (dx1, dy1, dx2, dy2)
        refined_xyxy = (x1, y1, x2, y2)
        area_ratio = ((x2 - x1) * (y2 - y1)) / float(detector_width * detector_height)
        if not 0.20 <= area_ratio <= 2.50:
            return None
        if cls._box_iou_xyxy(detector_xyxy, refined_xyxy) < 0.20:
            return None
        return x1, y1, x2 - x1, y2 - y1

    @staticmethod
    def _box_iou_xyxy(
        box_a: Tuple[int, int, int, int],
        box_b: Tuple[int, int, int, int],
    ) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        intersection = (ix2 - ix1) * (iy2 - iy1)
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return intersection / float(area_a + area_b - intersection)

    def _find_cached_recognition(
        self,
        box: Tuple[int, int, int, int],
        *,
        image_width: int,
        image_height: int,
    ) -> Tuple[Detection, Tuple[int, int, int, int] | None] | None:
        if self._ocr_cache_seconds <= 0:
            return None

        now = time.monotonic()
        self._ocr_cache = [
            entry
            for entry in self._ocr_cache
            if (now - entry[3]) <= self._ocr_cache_seconds
        ]
        best_iou = self._ocr_cache_iou
        best: Tuple[Detection, Tuple[int, int, int, int] | None] | None = None
        for cached_width, cached_height, cached_box, _, cached_result, box_transform in self._ocr_cache:
            if cached_width != image_width or cached_height != image_height:
                continue
            overlap = self._box_iou_xyxy(box, cached_box)
            if overlap < best_iou:
                continue
            best_iou = overlap
            refined_box = None
            if box_transform is not None:
                x1, y1, x2, y2 = box
                width = max(1, x2 - x1)
                height = max(1, y2 - y1)
                rel_x, rel_y, rel_width, rel_height = box_transform
                refined_x = max(0, min(image_width - 1, int(round(x1 + rel_x * width))))
                refined_y = max(0, min(image_height - 1, int(round(y1 + rel_y * height))))
                refined_width = max(1, min(image_width - refined_x, int(round(rel_width * width))))
                refined_height = max(1, min(image_height - refined_y, int(round(rel_height * height))))
                refined_box = (refined_x, refined_y, refined_width, refined_height)
            best = cached_result, refined_box
        return best

    def _store_cached_recognition(
        self,
        box: Tuple[int, int, int, int],
        recognized: Detection,
        *,
        refined_box: Tuple[int, int, int, int] | None,
        image_width: int,
        image_height: int,
    ) -> None:
        if self._ocr_cache_seconds <= 0 or not (recognized.raw_label or recognized.label):
            return

        now = time.monotonic()
        retained = []
        for entry in self._ocr_cache:
            cached_width, cached_height, cached_box, cached_time, _, _ = entry
            if (now - cached_time) > self._ocr_cache_seconds:
                continue
            if (
                cached_width == image_width
                and cached_height == image_height
                and self._box_iou_xyxy(box, cached_box) >= self._ocr_cache_iou
            ):
                continue
            retained.append(entry)
        box_transform = None
        if refined_box is not None:
            x1, y1, x2, y2 = box
            width = max(1, x2 - x1)
            height = max(1, y2 - y1)
            refined_x, refined_y, refined_width, refined_height = refined_box
            box_transform = (
                (refined_x - x1) / float(width),
                (refined_y - y1) / float(height),
                refined_width / float(width),
                refined_height / float(height),
            )
        retained.append((image_width, image_height, box, now, recognized, box_transform))
        self._ocr_cache = retained

    @staticmethod
    def _expand_box_for_ocr(
        box: Tuple[int, int, int, int],
        image_width: int,
        image_height: int,
        ratio: float = 0.25,
    ) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = box
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        pad_x = int(round(width * ratio))
        pad_y = int(round(height * ratio))
        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(image_width, x2 + pad_x),
            min(image_height, y2 + pad_y),
        )

    def detect(self, image) -> List[Detection]:
        if image is None:
            return []

        frame = np.asarray(image)
        if frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3:
            return []

        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        image_height, image_width = frame.shape[:2]

        input_tensor, scale, pad_x, pad_y = self._prepare_input(frame)
        if self._serialize_inference:
            # RK3568 只有一个 NPU 核，两个模型并发调用只会产生争抢和额外延迟。
            with _RKNN_INFERENCE_LOCK:
                outputs = self._rknn.inference(inputs=[input_tensor])
        else:
            outputs = self._rknn.inference(inputs=[input_tensor])
        prediction = self._select_prediction_array(outputs)
        if prediction is None or prediction.shape[0] == 0:
            return []

        class_scores = prediction[:, 4:4 + self._num_classes]
        scores = class_scores.max(axis=1)
        class_indices = class_scores.argmax(axis=1)
        keep_mask = scores >= self._conf_threshold
        if self._class_filter is not None:
            keep_mask &= np.isin(class_indices, self._class_filter)
        if self._class_margin_threshold > 0.0 and self._num_classes > 1:
            candidate_indices = np.flatnonzero(keep_mask)
            if candidate_indices.size > 0:
                candidate_scores = class_scores[candidate_indices].copy()
                candidate_classes = class_indices[candidate_indices]
                candidate_scores[np.arange(candidate_indices.size), candidate_classes] = -np.inf
                runner_up_scores = candidate_scores.max(axis=1)
                class_margins = scores[candidate_indices] - runner_up_scores
                keep_mask[candidate_indices] = class_margins >= self._class_margin_threshold
        if not np.any(keep_mask):
            return []

        filtered_boxes = self._xywh_to_xyxy(prediction[keep_mask, :4])
        filtered_boxes = self._restore_boxes(
            filtered_boxes,
            scale=scale,
            pad_x=pad_x,
            pad_y=pad_y,
            image_width=image_width,
            image_height=image_height,
        )
        filtered_scores = scores[keep_mask]
        filtered_classes = class_indices[keep_mask]

        keep_indices = self._nms(filtered_boxes, filtered_scores, self._nms_threshold)
        detections: List[Detection] = []
        for index in keep_indices:
            x1, y1, x2, y2 = filtered_boxes[index]
            ix1 = int(round(x1))
            iy1 = int(round(y1))
            ix2 = int(round(x2))
            iy2 = int(round(y2))
            if ix2 <= ix1 or iy2 <= iy1:
                continue

            class_id = int(filtered_classes[index])
            class_name = self._labels[class_id] if 0 <= class_id < len(self._labels) else f"plate_{class_id}"
            recognized, refined_box = self._recognize_box(
                frame,
                (ix1, iy1, ix2, iy2),
                image_width,
                image_height,
            )

            raw_label = ""
            type_name = class_name
            full_text = class_name
            label = class_name
            if recognized is not None:
                raw_label = recognized.raw_label or recognized.label
                type_name = recognized.type_name or type_name
                full_text = recognized.full_text or raw_label or full_text
                label = raw_label or recognized.label or label

            detections.append(
                Detection(
                    label=label,
                    raw_label=raw_label,
                    type_name=type_name,
                    full_text=full_text,
                    score=float(filtered_scores[index]),
                    box=refined_box or (ix1, iy1, max(1, ix2 - ix1), max(1, iy2 - iy1)),
                )
            )

        detections.sort(key=lambda det: det.score, reverse=True)
        return detections

    def close(self) -> None:
        if self._recognizer is not None:
            try:
                self._recognizer.close()
            finally:
                self._recognizer = None

        if self._rknn is not None:
            try:
                self._rknn.release()
            finally:
                self._rknn = None


@dataclass
class _PersonTrack:
    detection: Detection
    hits: int
    score_sum: float
    confirmed: bool


class PersonRknnDetector(BaseDetector):
    """复用通用 YOLOv8 RKNN 后处理，只保留行人类别。"""

    _COCO_LABELS = (
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
        "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
    )

    def __init__(
        self,
        *,
        model_path: str,
        input_size: int = 640,
        conf_threshold: float = 0.30,
        nms_threshold: float = 0.45,
        core_mask: str = "auto",
        num_classes: int = 80,
        serialize_inference: bool = True,
        class_margin_threshold: float = 0.08,
        confirmation_hits: int = 2,
        confirmation_threshold: float = 0.48,
        instant_threshold: float = 0.72,
        match_iou: float = 0.25,
        min_height_ratio: float = 0.04,
        max_width_height_ratio: float = 1.30,
        temporal_confirmation: bool = True,
    ) -> None:
        if int(num_classes) == 1:
            labels = ("person",)
        elif int(num_classes) == len(self._COCO_LABELS):
            labels = self._COCO_LABELS
        else:
            raise ValueError("行人 RKNN 模型类别数目前只支持 1 或 80")

        self._confirmation_hits = max(1, int(confirmation_hits))
        self._confirmation_threshold = min(1.0, max(0.0, float(confirmation_threshold)))
        self._instant_threshold = min(1.0, max(0.0, float(instant_threshold)))
        self._match_iou = min(1.0, max(0.0, float(match_iou)))
        self._min_height_ratio = min(1.0, max(0.0, float(min_height_ratio)))
        self._max_width_height_ratio = max(0.0, float(max_width_height_ratio))
        self._temporal_confirmation = bool(temporal_confirmation)
        self._tracks: List[_PersonTrack] = []

        self._detector = RknnLiteDetector(
            model_path=model_path,
            labels=labels,
            input_size=input_size,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            core_mask=core_mask,
            recognizer=None,
            class_filter=(0,),
            serialize_inference=serialize_inference,
            class_margin_threshold=(class_margin_threshold if int(num_classes) > 1 else 0.0),
        )

    @staticmethod
    def _box_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = box_a
        bx, by, bw, bh = box_b
        ix1 = max(ax, bx)
        iy1 = max(ay, by)
        ix2 = min(ax + aw, bx + bw)
        iy2 = min(ay + ah, by + bh)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        intersection = (ix2 - ix1) * (iy2 - iy1)
        union = max(1, aw * ah + bw * bh - intersection)
        return intersection / float(union)

    def _same_person(self, current: Detection, previous: Detection) -> bool:
        if self._box_iou(current.box, previous.box) >= self._match_iou:
            return True

        cx, cy, cw, ch = current.box
        px, py, pw, ph = previous.box
        current_center = (cx + cw * 0.5, cy + ch * 0.5)
        previous_center = (px + pw * 0.5, py + ph * 0.5)
        center_distance = (
            (current_center[0] - previous_center[0]) ** 2
            + (current_center[1] - previous_center[1]) ** 2
        ) ** 0.5
        area_ratio = (cw * ch) / float(max(1, pw * ph))
        return 0.45 <= area_ratio <= 2.20 and center_distance <= max(cw, ch, pw, ph) * 0.55

    def _passes_geometry(self, detection: Detection, image_width: int, image_height: int) -> bool:
        _, _, width, height = detection.box
        if width < 2 or height < 2:
            return False
        if height / float(max(1, image_height)) < self._min_height_ratio:
            return False
        if self._max_width_height_ratio > 0 and width / float(height) > self._max_width_height_ratio:
            return False
        return True

    def _confirm_people(self, detections: List[Detection]) -> List[Detection]:
        if not self._temporal_confirmation or self._confirmation_hits <= 1:
            self._tracks = []
            return detections

        next_tracks: List[_PersonTrack] = []
        confirmed_people: List[Detection] = []
        used_previous = set()
        for detection in detections:
            match_index = next(
                (
                    index
                    for index, track in enumerate(self._tracks)
                    if index not in used_previous and self._same_person(detection, track.detection)
                ),
                None,
            )
            if match_index is None:
                hits = 1
                score_sum = detection.score
                was_confirmed = False
            else:
                used_previous.add(match_index)
                previous = self._tracks[match_index]
                hits = previous.hits + 1
                score_sum = previous.score_sum + detection.score
                was_confirmed = previous.confirmed

            average_score = score_sum / float(hits)
            confirmed = (
                was_confirmed
                or detection.score >= self._instant_threshold
                or (
                    hits >= self._confirmation_hits
                    and average_score >= self._confirmation_threshold
                )
            )
            next_tracks.append(
                _PersonTrack(
                    detection=detection,
                    hits=hits,
                    score_sum=score_sum,
                    confirmed=confirmed,
                )
            )
            if confirmed:
                confirmed_people.append(detection)

        self._tracks = next_tracks
        return confirmed_people

    def detect(self, image) -> List[Detection]:
        people: List[Detection] = []
        image_array = np.asarray(image)
        if image_array.ndim < 2:
            self._tracks = []
            return people
        image_height, image_width = image_array.shape[:2]
        for detection in self._detector.detect(image):
            if detection.type_name != "person" and detection.label != "person":
                continue
            if not self._passes_geometry(detection, image_width, image_height):
                continue
            people.append(
                Detection(
                    label="行人",
                    raw_label="行人",
                    type_name="行人",
                    full_text=f"行人 {detection.score:.0%}",
                    score=detection.score,
                    box=detection.box,
                )
            )
        return self._confirm_people(people)

    def close(self) -> None:
        self._tracks = []
        self._detector.close()
