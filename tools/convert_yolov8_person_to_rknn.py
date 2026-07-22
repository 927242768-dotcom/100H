from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将单输出 YOLOv8 行人模型转换为 RKNN")
    parser.add_argument("--onnx", required=True, help="输入 ONNX 模型路径")
    parser.add_argument("--output", required=True, help="输出 RKNN 模型路径")
    parser.add_argument("--target", default="rk3568", help="目标平台")
    parser.add_argument("--input-size", type=int, default=640, help="模型输入尺寸")
    parser.add_argument("--do-quantization", action="store_true", help="启用 INT8 量化")
    parser.add_argument("--dataset", default="", help="量化校准图片列表")
    parser.add_argument("--verbose", action="store_true", help="输出详细转换日志")
    return parser.parse_args()


def validate_onnx(onnx_path: Path) -> None:
    import onnx

    model = onnx.load(str(onnx_path))
    if len(model.graph.output) != 1:
        raise ValueError(
            "当前运行时需要 Ultralytics 原始单输出 ONNX，期望输出形状为 (1,84,8400)；"
            "请勿直接使用 Rockchip Model Zoo 的多分支优化模型。"
        )

    dims = [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim]
    if len(dims) != 3 or 84 not in dims:
        raise ValueError(f"ONNX 输出形状不是预期的 (1,84,8400)：{dims}")
    print(f"ONNX 输出检查通过：{dims}")


def main() -> None:
    from rknn.api import RKNN

    args = parse_args()
    onnx_path = Path(args.onnx).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"未找到 ONNX 模型：{onnx_path}")
    if args.do_quantization and not args.dataset:
        raise ValueError("启用量化时必须提供 --dataset")

    validate_onnx(onnx_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rknn = RKNN(verbose=args.verbose)
    try:
        if rknn.config(target_platform=args.target) != 0:
            raise RuntimeError("RKNN config 失败")
        if rknn.load_onnx(
            model=str(onnx_path),
            input_size_list=[[3, args.input_size, args.input_size]],
        ) != 0:
            raise RuntimeError("RKNN load_onnx 失败")
        if rknn.build(
            do_quantization=args.do_quantization,
            dataset=args.dataset if args.do_quantization else None,
        ) != 0:
            raise RuntimeError("RKNN build 失败")
        if rknn.export_rknn(str(output_path)) != 0:
            raise RuntimeError("RKNN export_rknn 失败")
    finally:
        rknn.release()

    print(f"RKNN 行人模型导出完成：{output_path}")


if __name__ == "__main__":
    main()
