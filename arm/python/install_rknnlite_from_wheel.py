import argparse
import shutil
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="将一个或多个 Python wheel 离线解压到工程 vendor 目录")
    parser.add_argument(
        "wheel",
        nargs="*",
        help="要解压的 wheel 文件路径；不传时自动使用仓库 third_party/wheels 目录",
    )
    parser.add_argument(
        "--target",
        default=str(Path(__file__).resolve().parent / "vendor"),
        help="解压目标目录，默认是当前脚本同级的 vendor/",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="解压前先清理目标目录里已有的 rknnlite / psutil / ruamel 相关目录和 dist-info",
    )
    args = parser.parse_args()

    target_dir = Path(args.target).expanduser().resolve()
    if args.wheel:
        wheel_paths = [Path(item).expanduser().resolve() for item in args.wheel]
    else:
        bundled_dir = Path(__file__).resolve().parents[2] / "third_party" / "wheels"
        wheel_paths = sorted(bundled_dir.glob("*.whl"))
        if not wheel_paths:
            raise FileNotFoundError(f"仓库中没有找到离线 wheel：{bundled_dir}")
    for wheel_path in wheel_paths:
        if not wheel_path.is_file():
            raise FileNotFoundError(f"未找到 wheel 文件：{wheel_path}")

    target_dir.mkdir(parents=True, exist_ok=True)

    if args.clean:
        for pattern in (
            "rknnlite",
            "psutil*",
            "ruamel*",
            "rknn_toolkit_lite2-*.dist-info",
            "psutil-*.dist-info",
            "ruamel*.dist-info",
        ):
            for path in target_dir.glob(pattern):
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                elif path.exists():
                    path.unlink()

    for wheel_path in wheel_paths:
        with zipfile.ZipFile(wheel_path) as zf:
            zf.extractall(target_dir)
        print(f"已解压：{wheel_path.name}")

    print(f"已解压到：{target_dir}")
    print(f"请确认存在：{target_dir / 'rknnlite'}")


if __name__ == "__main__":
    main()
