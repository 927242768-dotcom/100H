from __future__ import annotations

import pathlib
import re
import sys


PATTERNS = [
    re.compile(r"\bmov\s+(v\d+)\.4s,\s*(v\d+)\.4s\b"),
    re.compile(r"\bmov\s+(\\[A-Za-z0-9_]+\\\(\))\.4s,\s*(\\[A-Za-z0-9_]+\\\(\))\.4s\b"),
]


def main() -> int:
    if len(sys.argv) != 2:
        print("用法: python3 patch_mnn_armv86_mov.py /path/to/MNNGemmInt8AddBiasScale_ARMV86_Unit.S")
        return 1

    target = pathlib.Path(sys.argv[1])
    if not target.is_file():
        print(f"文件不存在: {target}")
        return 2

    original = target.read_text(encoding="utf-8")
    patched = original
    count = 0
    for pattern in PATTERNS:
        patched, delta = pattern.subn(r"mov \1.16b, \2.16b", patched)
        count += delta

    if count == 0:
        print("没有匹配到需要替换的 ARMV86 mov 指令，文件保持不变。")
        return 0

    backup = target.with_suffix(target.suffix + ".bak")
    backup.write_text(original, encoding="utf-8")
    target.write_text(patched, encoding="utf-8")
    print(f"已完成补丁: {target}")
    print(f"备份文件: {backup}")
    print(f"替换数量: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
