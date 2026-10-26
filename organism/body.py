from __future__ import annotations

import asyncio
import ctypes
import json
import os
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class BodyMap:
    platform: str
    logical_cpus: int
    physical_cpus: int
    ram_gb: float
    numa_nodes: list[dict]
    warnings: list[str] = field(default_factory=list)
    loop_lag_ms: float = 0.0
    calibration: dict = field(default_factory=dict)
    measured_at: float = field(default_factory=time.time)

    @property
    def capacity_cores(self) -> float:
        return float(self.physical_cpus)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> BodyMap | None:
        try:
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return None


def inventory() -> BodyMap:
    logical = os.cpu_count() or 1
    physical, ram = logical, _ram_bytes()
    warnings = []
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or logical
        ram = psutil.virtual_memory().total
    except ImportError:
        warnings.append("psutil не установлен: физические ядра приняты равными логическим")
    ram_gb = round(ram / 2**30, 1)
    nodes = _numa_nodes() or [{"id": 0, "cpus": f"0-{logical - 1}", "mem_gb": ram_gb}]
    warnings += [f"NUMA-узел {n['id']} без своей памяти: его потоки читают чужие каналы"
                 for n in nodes if n["mem_gb"] < 0.5]
    return BodyMap(platform.platform(), logical, physical, ram_gb, nodes, warnings)


async def loop_lag_ms(samples: int = 20, interval: float = 0.005) -> float:
    worst = 0.0
    for _ in range(samples):
        t0 = time.perf_counter()
        await asyncio.sleep(interval)
        worst = max(worst, (time.perf_counter() - t0 - interval) * 1000)
    return round(worst, 2)


def _ram_bytes() -> int:
    if os.name == "nt":
        class MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                (n, ctypes.c_ulonglong) for n in ("total", "avail", "page_total", "page_avail",
                                                   "virt_total", "virt_avail", "ext_avail")]
        st = MemStatus()
        st.dwLength = ctypes.sizeof(MemStatus)
        return int(st.total) if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)) else 0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def _numa_nodes() -> list[dict]:
    base = Path("/sys/devices/system/node")
    nodes = []
    for d in sorted(base.glob("node[0-9]*")) if base.exists() else []:
        try:
            cpus = (d / "cpulist").read_text().strip()
            kb = next(int(line.split()[3]) for line in (d / "meminfo").read_text().splitlines() if "MemTotal" in line)
        except (OSError, StopIteration, ValueError, IndexError):
            continue
        nodes.append({"id": int(d.name[4:]), "cpus": cpus, "mem_gb": round(kb / 2**20, 1)})
    return nodes
