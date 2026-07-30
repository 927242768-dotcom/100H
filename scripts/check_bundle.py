from __future__ import annotations

from pathlib import Path


REQUIRED = (
    "fpga/bitstream/pcie_dma_test.sbit",
    "models/plate/yolov8s.rknn",
    "models/person/yolov8n.rknn",
    "models/hyperlpr/r2_mobile/rpv3_mdict_160h.mnn",
    "third_party/wheels/rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    missing = [path for path in REQUIRED if not (root / path).is_file()]
    if missing:
        raise SystemExit("缺少部署文件：\n" + "\n".join(missing))
    print("部署文件检查通过。")


if __name__ == "__main__":
    main()
