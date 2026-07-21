import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="USB 摄像头连通性检查")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--output", default="camera_probe.png", help="抓拍保存路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError("摄像头打开失败")

    try:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("读取摄像头帧失败")
        cv2.imwrite(str(output_path), frame)
        print(f"camera ok, image saved to {output_path}")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
