import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from pcie_bar import PcieBars


BAR_WINDOW_BYTES = 64 * 1024
HEADER_OFFSET = 0x000
FRAME_OFFSET = 0x100
CURRENT_SAFE_FRAME_BYTES = 7936
DEFAULT_SAFE_WIDTH = 112
DEFAULT_SAFE_HEIGHT = 64

REG_CONTROL = 0x000
REG_WIDTH = 0x010
REG_HEIGHT = 0x020
REG_THRESHOLD = 0x030
REG_ROI_XY = 0x040
REG_ROI_WH = 0x050
REG_MORPH_CFG = 0x060
REG_FRAME_BYTES = 0x070

PREPROC_INVERT = 1 << 0
PREPROC_PASSTHROUGH = 1 << 1
PREPROC_SOBEL = 1 << 2
PREPROC_DENOISE_SHIFT = 4
PREPROC_DENOISE_OFF = 0
PREPROC_DENOISE_GAUSS3X3 = 1
PREPROC_MORPH_SHIFT = 6
PREPROC_MORPH_OFF = 0
PREPROC_MORPH_OPEN = 1
PREPROC_MORPH_CLOSE = 2
PREPROC_MORPH_OPEN_THEN_CLOSE = 3

STATUS_SIGNATURE = 0x54504650
STATUS_BUSY = 1 << 0
STATUS_DONE = 1 << 1
STATUS_ERROR = 1 << 2
STATUS_CONTINUOUS = 1 << 3


@dataclass
class FpgaStatus:
    busy: bool
    done: bool
    error: bool
    continuous: bool
    width: int
    height: int
    frame_counter: int
    threshold: int
    morph_cfg: int
    roi: Tuple[int, int, int, int]
    frame_bytes: int
    active_pixels: int


class FpgaPreprocessClient:
    def __init__(self, resource_root: str, bar_size: int = BAR_WINDOW_BYTES) -> None:
        self.bars = PcieBars(resource_root, bar_size)
        self.resource_root = Path(resource_root)
        self.current_shape: Optional[Tuple[int, int]] = None
        self.single_bar_mode = (self.bars.bar1 is None) or (self.bars.bar2 is None)

    def close(self) -> None:
        self.bars.close()

    @staticmethod
    def build_morph_cfg(
        *,
        invert: bool = False,
        passthrough_gray: bool = False,
        enable_sobel: bool = False,
        denoise_mode: int = PREPROC_DENOISE_OFF,
        morph_mode: int = PREPROC_MORPH_OFF,
    ) -> int:
        cfg = 0
        if invert:
            cfg |= PREPROC_INVERT
        if passthrough_gray:
            cfg |= PREPROC_PASSTHROUGH
        if enable_sobel:
            cfg |= PREPROC_SOBEL
        cfg |= (denoise_mode & 0x3) << PREPROC_DENOISE_SHIFT
        cfg |= (morph_mode & 0x3) << PREPROC_MORPH_SHIFT
        return cfg

    def configure(
        self,
        width: int,
        height: int,
        threshold: int,
        roi: Optional[Tuple[int, int, int, int]] = None,
        morph_cfg: int = 0,
        continuous: bool = False,
    ) -> None:
        if roi is None:
            roi = (0, 0, 0, 0)

        frame_bytes = width * height
        self.validate_frame_size(width, height)
        self.current_shape = (height, width)

        reg_bar = self.bars.bar0 if self.single_bar_mode else self.bars.bar1
        reg_bar.write32(REG_WIDTH, width)
        reg_bar.write32(REG_HEIGHT, height)
        reg_bar.write32(REG_THRESHOLD, threshold & 0xFF)
        reg_bar.write32(REG_ROI_XY, (roi[1] << 16) | roi[0])
        reg_bar.write32(REG_ROI_WH, (roi[3] << 16) | roi[2])
        reg_bar.write32(REG_MORPH_CFG, morph_cfg & 0xFFFF)
        reg_bar.write32(REG_FRAME_BYTES, frame_bytes)

        control = (1 << 1) if continuous else 0
        reg_bar.write32(REG_CONTROL, control | (1 << 2))

    @staticmethod
    def validate_frame_size(width: int, height: int) -> None:
        frame_bytes = width * height
        if frame_bytes <= 0:
            raise ValueError("FPGA 帧尺寸必须大于 0")
        if frame_bytes > CURRENT_SAFE_FRAME_BYTES:
            raise ValueError(
                "当前 FPGA 版本安全帧区只有 "
                f"{CURRENT_SAFE_FRAME_BYTES} 字节，"
                f"你设置的是 {width}x{height}={frame_bytes} 字节，请降低分辨率"
            )

    def ensure_signature(self) -> FpgaStatus:
        return self.read_status()

    def write_grayscale_frame(self, gray: np.ndarray) -> None:
        if gray.dtype != np.uint8:
            raise ValueError("FPGA 输入必须是 uint8 灰度图")
        if gray.ndim != 2:
            raise ValueError("FPGA 输入必须是单通道灰度图")

        flat = gray.reshape(-1).tobytes()
        if FRAME_OFFSET + len(flat) > BAR_WINDOW_BYTES:
            raise ValueError("输入帧超出 BAR0 可用窗口，请降低分辨率")

        padded = flat
        padding = (16 - (len(flat) % 16)) % 16
        if padding:
            padded += b"\x00" * padding

        self.bars.bar0.write(FRAME_OFFSET, padded)

    def start(self, continuous: bool = False) -> None:
        control = 1
        if continuous:
            control |= (1 << 1)
        reg_bar = self.bars.bar0 if self.single_bar_mode else self.bars.bar1
        reg_bar.write32(REG_CONTROL, control)

    def read_status(self) -> FpgaStatus:
        status_bar = self.bars.bar0 if self.single_bar_mode else self.bars.bar2
        header = status_bar.read(HEADER_OFFSET, 48)
        dwords = struct.unpack("<12I", header)

        if dwords[0] != STATUS_SIGNATURE:
            raise RuntimeError(f"状态头签名错误: 0x{dwords[0]:08x}")

        status_bits = dwords[1]
        width = dwords[2] & 0xFFFF
        height = (dwords[2] >> 16) & 0xFFFF
        frame_counter = dwords[3]
        threshold = dwords[4] & 0xFF
        morph_cfg = (dwords[4] >> 16) & 0xFFFF
        roi_x = dwords[5] & 0xFFFF
        roi_y = (dwords[5] >> 16) & 0xFFFF
        roi_w = dwords[6] & 0xFFFF
        roi_h = (dwords[6] >> 16) & 0xFFFF
        frame_bytes = dwords[7]
        active_pixels = dwords[8]

        return FpgaStatus(
            busy=bool(status_bits & STATUS_BUSY),
            done=bool(status_bits & STATUS_DONE),
            error=bool(status_bits & STATUS_ERROR),
            continuous=bool(status_bits & STATUS_CONTINUOUS),
            width=width,
            height=height,
            frame_counter=frame_counter,
            threshold=threshold,
            morph_cfg=morph_cfg,
            roi=(roi_x, roi_y, roi_w, roi_h),
            frame_bytes=frame_bytes,
            active_pixels=active_pixels,
        )

    def wait_done(self, timeout_s: float = 0.2, poll_interval_s: float = 0.002) -> FpgaStatus:
        deadline = time.time() + timeout_s
        last_status: Optional[FpgaStatus] = None

        while time.time() < deadline:
            last_status = self.read_status()
            if last_status.error:
                raise RuntimeError("FPGA 预处理返回错误状态")
            if last_status.done and not last_status.busy:
                return last_status
            time.sleep(poll_interval_s)

        raise TimeoutError(f"等待 FPGA 完成超时，最后状态: {last_status}")

    def read_mask(self, width: int, height: int) -> np.ndarray:
        frame_bytes = width * height
        if FRAME_OFFSET + frame_bytes > BAR_WINDOW_BYTES:
            raise ValueError("输出掩码超出 BAR 可用窗口，请降低分辨率")

        data_bar = self.bars.bar0 if self.single_bar_mode else self.bars.bar2
        raw = data_bar.read(FRAME_OFFSET, frame_bytes)
        mask = np.frombuffer(raw, dtype=np.uint8).reshape(height, width)
        return mask.copy()
