from __future__ import annotations

import pathlib
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print("用法: python3 patch_mnn_cpufixedpoint.py /path/to/CPUFixedPoint.hpp")
        return 1

    target = pathlib.Path(sys.argv[1])
    if not target.is_file():
        print(f"文件不存在: {target}")
        return 2

    original = target.read_text(encoding="utf-8")
    marker = "#include <stdint.h>"
    replacement = "#include <stdint.h>\n#include <cstdint>"

    if "#include <cstdint>" in original:
        print("已经包含 <cstdint>，文件保持不变。")
        return 0

    if marker not in original:
        print(f"没有找到标记头文件: {marker}")
        return 3

    patched = original.replace(marker, replacement, 1)
    backup = target.with_suffix(target.suffix + ".bak")
    backup.write_text(original, encoding="utf-8")
    target.write_text(patched, encoding="utf-8")
    print(f"已完成补丁: {target}")
    print(f"备份文件: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
