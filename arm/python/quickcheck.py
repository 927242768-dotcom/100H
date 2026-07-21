import argparse
import struct
import time
from pathlib import Path

from pcie_bar import PcieBars


BAR_WINDOW_BYTES = 64 * 1024
HEADER_OFFSET = 0x000
FRAME_OFFSET = 0x100

REG_CONTROL = 0x000
REG_WIDTH = 0x010
REG_HEIGHT = 0x020
REG_THRESHOLD = 0x030
REG_ROI_XY = 0x040
REG_ROI_WH = 0x050
REG_MORPH_CFG = 0x060
REG_FRAME_BYTES = 0x070

STATUS_SIGNATURE = 0x54504650
STATUS_BUSY = 1 << 0
STATUS_DONE = 1 << 1
STATUS_ERROR = 1 << 2
STATUS_CONTINUOUS = 1 << 3


def build_test_frame(width: int, height: int) -> bytes:
    data = bytearray(width * height)
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            value = (x * 255) // max(width - 1, 1)
            if (width // 4) <= x < (width // 2) and (height // 4) <= y < (height // 2):
                value = 220
            data[idx] = value
    return bytes(data)


def write_pgm(path: Path, width: int, height: int, payload: bytes) -> None:
    header = f"P5\n{width} {height}\n255\n".encode("ascii")
    path.write_bytes(header + payload)


def dump_status(bar0) -> tuple[int, list[int]]:
    raw = bar0.read(HEADER_OFFSET, 48)
    dwords = list(struct.unpack("<12I", raw))
    return dwords[0], dwords


def print_status(prefix: str, dwords: list[int]) -> None:
    status_bits = dwords[1]
    width = dwords[2] & 0xFFFF
    height = (dwords[2] >> 16) & 0xFFFF
    frame_counter = dwords[3]
    threshold = dwords[4] & 0xFF
    morph_cfg = (dwords[4] >> 16) & 0xFFFF
    frame_bytes = dwords[7]
    active_pixels = dwords[8]

    print(
        f"{prefix}: busy={int(bool(status_bits & STATUS_BUSY))} "
        f"done={int(bool(status_bits & STATUS_DONE))} "
        f"error={int(bool(status_bits & STATUS_ERROR))} "
        f"continuous={int(bool(status_bits & STATUS_CONTINUOUS))} "
        f"width={width} height={height} frame_counter={frame_counter} "
        f"threshold={threshold} morph_cfg=0x{morph_cfg:04x} "
        f"frame_bytes={frame_bytes} active_pixels={active_pixels}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="FPGA BAR0 快速验板脚本")
    parser.add_argument("--resource-root", required=True, help="PCIe 设备 sysfs 根目录")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--threshold", type=int, default=96)
    parser.add_argument("--timeout", type=float, default=0.5)
    parser.add_argument("--output-dir", default="quickcheck_out")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_bytes = args.width * args.height
    if FRAME_OFFSET + frame_bytes > BAR_WINDOW_BYTES:
        raise ValueError("测试分辨率超出 64KB BAR 窗口，请降低 width/height")

    bars = PcieBars(args.resource_root, BAR_WINDOW_BYTES)
    try:
        signature, initial_dwords = dump_status(bars.bar0)
        print(f"header signature = 0x{signature:08x}")
        if signature != STATUS_SIGNATURE:
            raise RuntimeError("状态头签名不对，说明当前 sbit 不是我们这版 FPGA 逻辑")
        print_status("initial", initial_dwords)

        bars.bar0.write32(REG_WIDTH, args.width)
        bars.bar0.write32(REG_HEIGHT, args.height)
        bars.bar0.write32(REG_THRESHOLD, args.threshold & 0xFF)
        bars.bar0.write32(REG_ROI_XY, 0)
        bars.bar0.write32(REG_ROI_WH, 0)
        bars.bar0.write32(REG_MORPH_CFG, 0)
        bars.bar0.write32(REG_FRAME_BYTES, frame_bytes)
        bars.bar0.write32(REG_CONTROL, 1 << 2)

        frame = build_test_frame(args.width, args.height)
        padded = frame + (b"\x00" * ((16 - (len(frame) % 16)) % 16))
        bars.bar0.write(FRAME_OFFSET, padded)

        before_start_signature, before_start_dwords = dump_status(bars.bar0)
        if before_start_signature != STATUS_SIGNATURE:
            raise RuntimeError("写帧后状态头签名异常")
        print_status("before_start", before_start_dwords)

        bars.bar0.write32(REG_CONTROL, 1)

        deadline = time.time() + args.timeout
        final_dwords = before_start_dwords
        while time.time() < deadline:
            _, final_dwords = dump_status(bars.bar0)
            if final_dwords[1] & STATUS_ERROR:
                print_status("error", final_dwords)
                raise RuntimeError("FPGA 返回 error 状态")
            if final_dwords[1] & STATUS_DONE:
                break
            time.sleep(0.005)
        else:
            print_status("timeout", final_dwords)
            raise TimeoutError("等待 FPGA done 超时")

        print_status("final", final_dwords)

        mask = bars.bar0.read(FRAME_OFFSET, frame_bytes)
        active_pixels_host = sum(1 for item in mask if item != 0)
        print(f"mask active pixels counted on host = {active_pixels_host}")

        input_path = output_dir / "input.pgm"
        mask_path = output_dir / "mask.pgm"
        write_pgm(input_path, args.width, args.height, frame)
        write_pgm(mask_path, args.width, args.height, mask)

        print(f"input saved to {input_path}")
        print(f"mask saved to {mask_path}")
    finally:
        bars.close()


if __name__ == "__main__":
    main()
