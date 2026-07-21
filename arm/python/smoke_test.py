import argparse
from pathlib import Path

import cv2
import numpy as np

from fpga_client import FpgaPreprocessClient


def build_test_frame(width: int, height: int) -> np.ndarray:
    x = np.linspace(0, 255, width, dtype=np.uint8)
    frame = np.tile(x, (height, 1))
    cv2.rectangle(frame, (width // 4, height // 4), (width // 2, height // 2), 220, -1)
    cv2.circle(frame, (width * 3 // 4, height // 2), min(width, height) // 8, 40, -1)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="FPGA BAR 最小闭环测试")
    parser.add_argument("--resource-root", required=True, help="PCIe 设备 sysfs 根目录")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--threshold", type=int, default=96)
    parser.add_argument("--output-dir", default="smoke_out")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fpga = FpgaPreprocessClient(args.resource_root)
    try:
        gray = build_test_frame(args.width, args.height)
        fpga.configure(args.width, args.height, args.threshold)
        fpga.write_grayscale_frame(gray)
        fpga.start()
        status = fpga.wait_done()
        mask = fpga.read_mask(status.width, status.height)

        cv2.imwrite(str(output_dir / "input.png"), gray)
        cv2.imwrite(str(output_dir / "mask.png"), mask)

        print("status =", status)
        print("input saved to", output_dir / "input.png")
        print("mask saved to", output_dir / "mask.png")
    finally:
        fpga.close()


if __name__ == "__main__":
    main()

