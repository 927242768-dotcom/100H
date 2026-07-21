import mmap
import os
import struct
from pathlib import Path


class PcieBarRegion:
    def __init__(self, path: Path, size: int) -> None:
        self.path = Path(path)
        self.size = size
        self.fd = os.open(str(self.path), os.O_RDWR | os.O_SYNC)
        self.mm = mmap.mmap(self.fd, self.size, access=mmap.ACCESS_WRITE)

    def close(self) -> None:
        try:
            self.mm.close()
        finally:
            os.close(self.fd)

    def write(self, offset: int, data: bytes) -> None:
        if offset < 0 or offset + len(data) > self.size:
            raise ValueError(f"BAR 写越界: offset=0x{offset:x}, len={len(data)}")
        self.mm.seek(offset)
        self.mm.write(data)
        self.mm.flush()

    def read(self, offset: int, size: int) -> bytes:
        if offset < 0 or offset + size > self.size:
            raise ValueError(f"BAR 读越界: offset=0x{offset:x}, len={size}")
        self.mm.seek(offset)
        return self.mm.read(size)

    def write32(self, offset: int, value: int) -> None:
        self.write(offset, struct.pack("<I", value & 0xFFFFFFFF))

    def read32(self, offset: int) -> int:
        return struct.unpack("<I", self.read(offset, 4))[0]


class PcieBars:
    def __init__(self, root: str, bar_size: int = 64 * 1024) -> None:
        root_path = Path(root)
        self.bar0 = PcieBarRegion(root_path / "resource0", bar_size)
        self.bar1 = PcieBarRegion(root_path / "resource1", bar_size) if (root_path / "resource1").exists() else None
        self.bar2 = PcieBarRegion(root_path / "resource2", bar_size) if (root_path / "resource2").exists() else None

    def close(self) -> None:
        self.bar0.close()
        if self.bar1 is not None:
            self.bar1.close()
        if self.bar2 is not None:
            self.bar2.close()
