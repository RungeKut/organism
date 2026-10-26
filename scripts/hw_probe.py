from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from organism.calibrate import curve, knee, membw_gbps

CHANNELS_PER_SOCKET = 8
CURVE_THREADS = (1, 2, 4, 8, 16, 32, 64)
MIN_NODE_MB = 512
SYSFS_NODES = Path("/sys/devices/system/node")
PROC_MEMINFO = Path("/proc/meminfo")


def parse_cpulist(text: str) -> list[int]:
    cpus = []
    for part in text.replace(" ", ",").split(","):
        part = part.strip()
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        try:
            cpus.extend(range(int(lo), int(hi) + 1) if sep else [int(lo)])
        except ValueError:
            continue
    return sorted(set(cpus))


def format_cpulist(cpus: list[int]) -> str:
    out, start, prev = [], None, None
    for c in sorted(set(cpus)):
        if start is None:
            start = prev = c
        elif c == prev + 1:
            prev = c
        else:
            out.append(f"{start}-{prev}" if prev > start else str(start))
            start = prev = c
    if start is not None:
        out.append(f"{start}-{prev}" if prev > start else str(start))
    return ",".join(out)


def _size_mb(text: str) -> float | None:
    m = re.match(r"\s*([\d.]+)\s*([KMGT]i?B?|)", text, re.I)
    if not m:
        return None
    unit = m.group(2).upper()[:1]
    factor = {"K": 1 / 1024, "M": 1.0, "G": 1024.0, "T": 1024.0 ** 2, "": 1 / 2**20}[unit]
    return float(m.group(1)) * factor


def parse_lscpu(text: str) -> dict:
    kv, numa = {}, []
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key, value = key.strip(), value.strip()
        m = re.fullmatch(r"NUMA node(\d+) CPU\(s\)", key)
        if m:
            numa.append({"id": int(m.group(1)), "cpus": value})
        else:
            kv.setdefault(key, value)

    def num(key: str) -> int | None:
        try:
            return int(kv[key])
        except (KeyError, ValueError):
            return None

    flags = set(kv.get("Flags", "").split())
    l3 = kv.get("L3 cache")
    inst = re.search(r"\((\d+) instances?\)", l3 or "")
    return {
        "model_name": kv.get("Model name", ""),
        "cpus": num("CPU(s)"),
        "sockets": num("Socket(s)"),
        "cores_per_socket": num("Core(s) per socket"),
        "threads_per_core": num("Thread(s) per core"),
        "l3": l3,
        "l3_mb": round(_size_mb(l3), 1) if l3 and _size_mb(l3) is not None else None,
        "l3_instances": int(inst.group(1)) if inst else None,
        "flags": {"avx2": "avx2" in flags, "avx512f": "avx512f" in flags, "fma": "fma" in flags},
        "numa_nodes": sorted(numa, key=lambda n: n["id"]),
    }


def parse_numactl_hardware(text: str) -> dict:
    nodes: dict[int, dict] = {}
    distances: list[list[int]] = []
    in_dist = False

    def node(i: int) -> dict:
        return nodes.setdefault(i, {"id": i, "cpus": "", "size_mb": None, "free_mb": None})

    for line in text.splitlines():
        if line.strip().startswith("node distances"):
            in_dist = True
            continue
        if in_dist:
            m = re.match(r"\s*(\d+):\s*([\d\s]*)$", line)
            if m:
                distances.append([int(x) for x in m.group(2).split()])
            continue
        m = re.match(r"\s*node (\d+) cpus:(.*)$", line)
        if m:
            node(int(m.group(1)))["cpus"] = format_cpulist(parse_cpulist(m.group(2)))
            continue
        m = re.match(r"\s*node (\d+) (size|free):\s*(\d+)\s*MB", line)
        if m:
            node(int(m.group(1)))[f"{m.group(2)}_mb"] = int(m.group(3))
    return {"nodes": [nodes[i] for i in sorted(nodes)], "distances": distances}


def _dmi_blocks(text: str) -> list[dict]:
    blocks = []
    for chunk in re.split(r"\r?\n\s*\r?\n", text):
        lines = [ln for ln in chunk.splitlines() if ln.strip()]
        if not any(ln.strip() == "Memory Device" for ln in lines):
            continue
        fields = {}
        for ln in lines:
            key, sep, value = ln.strip().partition(":")
            if sep:
                fields.setdefault(key.strip(), value.strip())
        blocks.append(fields)
    return blocks


def _first_int(text: str | None) -> int | None:
    m = re.match(r"\s*(\d+)", text or "")
    return int(m.group(1)) if m else None


def parse_dmidecode_memory(text: str) -> list[dict]:
    slots = []
    for f in _dmi_blocks(text):
        if f.get("Size", "").lower().startswith("no module"):
            continue
        mb = _size_mb(f.get("Size", ""))
        speed = _first_int(f.get("Configured Memory Speed") or f.get("Configured Clock Speed") or f.get("Speed"))
        slots.append({
            "locator": f.get("Locator", ""),
            "bank": f.get("Bank Locator", ""),
            "size_gb": round(mb / 1024, 1) if mb is not None else None,
            "speed_mts": speed,
            "populated": True,
        })
    return slots


def count_dmidecode_empty(text: str) -> int:
    return sum(1 for f in _dmi_blocks(text) if f.get("Size", "").lower().startswith("no module"))


def parse_meminfo(text: str) -> dict:
    kb = {}
    for line in text.splitlines():
        m = re.match(r"\s*(?:Node \d+\s+)?(\w+):\s+(\d+)\s*kB", line)
        if m:
            kb.setdefault(m.group(1), int(m.group(2)))
    return {
        "total_mb": kb["MemTotal"] // 1024 if "MemTotal" in kb else None,
        "free_mb": kb["MemFree"] // 1024 if "MemFree" in kb else None,
        "available_mb": kb["MemAvailable"] // 1024 if "MemAvailable" in kb else None,
    }


def _node_gb(node: dict) -> float | None:
    if node.get("size_mb") is not None:
        return node["size_mb"] / 1024
    if node.get("mem_gb") is not None:
        return float(node["mem_gb"])
    return None


def _fmt_gb(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def channels_report(slots: list[dict] | None, nodes: list[dict], channels_per_node: int = CHANNELS_PER_SOCKET,
                    empty_slots: int | None = None) -> dict:
    warnings = []
    with_mem, without_mem = [], []
    for n in nodes:
        gb = _node_gb(n)
        if gb is None:
            continue
        (with_mem if gb * 1024 >= MIN_NODE_MB else without_mem).append(n["id"])
    if with_mem:
        warnings += [f"узел {i} без своей памяти: его потоки читают чужие каналы" for i in without_mem]
    needed = channels_per_node * max(1, len(with_mem))
    report = {
        "occupied": None, "empty": empty_slots, "total_slots": None, "total_gb": None,
        "sizes_gb": [], "speeds_mts": [], "channels_needed": needed,
        "nodes_with_memory": with_mem, "nodes_without_memory": without_mem, "warnings": warnings,
    }
    if not slots:
        return report
    occupied = len(slots)
    total = occupied + empty_slots if empty_slots is not None else None
    sizes = sorted({s["size_gb"] for s in slots if s.get("size_gb")})
    speeds = sorted({s["speed_mts"] for s in slots if s.get("speed_mts")})
    report.update(occupied=occupied, total_slots=total, sizes_gb=sizes, speeds_mts=speeds,
                  total_gb=round(sum(s.get("size_gb") or 0 for s in slots), 1))
    if occupied < needed:
        of = f" из {total}" if total else ""
        warnings.append(f"заняты только {occupied} слотов{of} — каналы памяти используются не полностью "
                        f"(для полной полосы нужно {needed})")
    if len(sizes) > 1:
        warnings.append("планки разного размера: " + ", ".join(_fmt_gb(s) for s in sizes) + " ГБ")
    if len(speeds) > 1:
        warnings.append("планки разной скорости: " + ", ".join(str(s) for s in speeds) + " МТ/с")
    return report


def read_sysfs_nodes(base: Path = SYSFS_NODES) -> list[dict]:
    nodes = []
    try:
        dirs = sorted(base.glob("node[0-9]*"), key=lambda d: int(d.name[4:]))
    except OSError:
        return []
    for d in dirs:
        try:
            cpus = (d / "cpulist").read_text().strip()
            mem = parse_meminfo((d / "meminfo").read_text())
        except OSError:
            continue
        nodes.append({"id": int(d.name[4:]), "cpus": cpus, "size_mb": mem["total_mb"], "free_mb": mem["free_mb"]})
    return nodes


def _which(name: str) -> str | None:
    return shutil.which(name)


def _run(cmd: list[str], timeout: float = 30, cwd=None) -> tuple[int, str] | None:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout, cwd=cwd)
    except (OSError, subprocess.SubprocessError):
        return None
    return p.returncode, p.stdout


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def bandwidth_cmd(numactl: str, cpu_node: int, mem_node: int, mb: int = 64, runs: int = 5,
                  python: str | None = None) -> list[str]:
    code = f"from organism.calibrate import membw_gbps; print(membw_gbps({int(mb)}, {int(runs)}))"
    return [numactl, f"--cpunodebind={cpu_node}", f"--membind={mem_node}", python or sys.executable, "-c", code]


def _bw_under_numactl(numactl: str, cpu_node: int, mem_node: int, root: Path, mb: int, runs: int) -> float | None:
    res = _run(bandwidth_cmd(numactl, cpu_node, mem_node, mb, runs), timeout=180, cwd=str(root))
    if not res or res[0] != 0:
        return None
    try:
        return round(float(res[1].strip().splitlines()[-1]), 2)
    except (ValueError, IndexError):
        return None


def measure_bandwidth(nodes: list[dict], numactl: str | None, root: Path, mb: int, runs: int,
                      skipped: list[str]) -> dict:
    mem_nodes = [n for n in nodes if (n.get("size_mb") or 0) >= MIN_NODE_MB]
    if not numactl or not mem_nodes:
        skipped.append("замер полосы по узлам пропущен (нет numactl или размеров узлов): сделан один общий замер")
        return {"mode": "общий", "local_gbps": {"all": round(membw_gbps(mb, runs), 2)}, "remote_gbps": {}}
    local, remote = {}, {}
    for n in mem_nodes:
        local[str(n["id"])] = _bw_under_numactl(numactl, n["id"], n["id"], root, mb, runs)
    target = mem_nodes[0]["id"]
    for n in nodes:
        if n.get("size_mb") == 0 and n.get("cpus"):
            remote[str(n["id"])] = _bw_under_numactl(numactl, n["id"], target, root, mb, runs)
    if any(v is None for v in list(local.values()) + list(remote.values())):
        skipped.append("часть замеров под numactl не удалась (значение null)")
    return {"mode": "numactl", "local_gbps": local, "remote_gbps": remote, "remote_target_node": target}


def _tool_output(cmd: list[str], skipped: list[str], reason: str) -> str | None:
    if not _which(cmd[0]):
        skipped.append(f"{cmd[0]} не найден: {reason}")
        return None
    res = _run(cmd)
    if not res or res[0] != 0 or not res[1].strip():
        skipped.append(f"{cmd[0]} не отработал: {reason}")
        return None
    return res[1]


def _dmidecode_text(skipped: list[str]) -> str | None:
    base = ["dmidecode", "-t", "memory"]
    if not _which("dmidecode"):
        skipped.append("dmidecode не найден: раскладка планок неизвестна")
        return None
    attempts = [base] + ([["sudo", "-n"] + base] if _which("sudo") else [])
    for cmd in attempts:
        res = _run(cmd)
        if res and res[0] == 0 and "Memory Device" in res[1]:
            return res[1]
    skipped.append("dmidecode пропущен: нужен root (запустите: sudo python3 scripts/hw_probe.py)")
    return None


def collect(root: Path = ROOT, bw_mb: int = 64, bw_runs: int = 5) -> dict:
    skipped: list[str] = []
    logical = os.cpu_count() or 1
    report: dict = {
        "host": socket.gethostname(), "platform": platform.platform(), "machine": platform.machine(),
        "python": sys.version.split()[0], "collected_at": time.time(), "logical_cpus": logical,
    }

    out = _tool_output(["lscpu"], skipped, "модель, сокеты и флаги неизвестны")
    lscpu = parse_lscpu(out) if out else {}
    report["lscpu"] = lscpu

    numactl = _which("numactl")
    out = _tool_output(["numactl", "--hardware"], skipped, "карта узлов берётся из /sys")
    hw = parse_numactl_hardware(out) if out else {"nodes": [], "distances": []}
    report["numactl"] = hw

    meminfo = parse_meminfo(_read_text(PROC_MEMINFO))
    report["meminfo"] = meminfo

    nodes = hw["nodes"] or read_sysfs_nodes()
    if not nodes and lscpu.get("numa_nodes"):
        nodes = [{"id": n["id"], "cpus": n["cpus"], "size_mb": None, "free_mb": None} for n in lscpu["numa_nodes"]]
    if not nodes:
        nodes = [{"id": 0, "cpus": f"0-{logical - 1}", "size_mb": meminfo["total_mb"], "free_mb": meminfo["free_mb"]}]
    if len(nodes) == 1 and nodes[0].get("size_mb") is None:
        nodes[0]["size_mb"], nodes[0]["free_mb"] = meminfo["total_mb"], meminfo["free_mb"]
    for n in nodes:
        n["cpu_count"] = len(parse_cpulist(n.get("cpus") or ""))
    report["nodes"] = nodes
    report["distances"] = hw["distances"]

    text = _dmidecode_text(skipped)
    slots = parse_dmidecode_memory(text) if text else None
    empty = count_dmidecode_empty(text) if text else None
    sockets = lscpu.get("sockets") or 1
    per_node = max(1, CHANNELS_PER_SOCKET // max(1, len(nodes) // sockets))
    report["dmidecode"] = {"slots": slots, "empty": empty}
    report["channels"] = channels_report(slots, nodes, per_node, empty)

    report["bandwidth"] = measure_bandwidth(nodes, numactl, root, bw_mb, bw_runs, skipped)

    threads = [t for t in CURVE_THREADS if t <= logical] or [1]
    points = curve(tuple(threads), tasks=max(24, 2 * max(threads)), repeats=2)
    report["curve"] = [[n, round(rate, 2)] for n, rate in points]
    report["knee"] = knee(points)

    report["warnings"] = list(report["channels"]["warnings"])
    report["skipped"] = skipped
    return report


def format_summary(report: dict) -> str:
    lines = [f"Хост: {report.get('host')} ({report.get('platform')})"]
    ls = report.get("lscpu") or {}
    if ls.get("model_name"):
        fl = ls["flags"]
        yn = lambda v: "да" if v else "нет"
        lines.append(f"Процессор: {ls['model_name']}; сокетов {ls['sockets']}, ядер на сокет {ls['cores_per_socket']}, "
                     f"потоков на ядро {ls['threads_per_core']}, L3 {ls['l3']}")
        lines.append(f"Команды: AVX2 {yn(fl['avx2'])}, AVX-512 {yn(fl['avx512f'])}, FMA {yn(fl['fma'])}")
    lines.append(f"Логических CPU: {report.get('logical_cpus')}")
    mem = report.get("meminfo", {}).get("total_mb")
    if mem:
        lines.append(f"Память всего: {mem / 1024:.1f} ГБ")
    lines.append("Узлы NUMA:")
    for n in report.get("nodes", []):
        size = f"{n['size_mb'] / 1024:.1f} ГБ" if n.get("size_mb") is not None else "размер неизвестен"
        lines.append(f"  узел {n['id']}: CPU {n.get('cpu_count', 0)}, память: {size}")
    ch = report.get("channels") or {}
    if ch.get("occupied"):
        total = f" из {ch['total_slots']}" if ch.get("total_slots") else ""
        sizes = "/".join(_fmt_gb(s) for s in ch["sizes_gb"])
        speeds = "/".join(str(s) for s in ch["speeds_mts"]) or "?"
        lines.append(f"Планки: занято {ch['occupied']}{total}, {sizes} ГБ, {speeds} МТ/с, всего {_fmt_gb(ch['total_gb'])} ГБ")
    else:
        lines.append("Планки: раскладка неизвестна")
    bw = report.get("bandwidth") or {}
    lines.append(f"Пропускная способность памяти ({bw.get('mode')}), ГБ/с:")
    for k, v in bw.get("local_gbps", {}).items():
        lines.append(f"  узел {k}, своя память: {v}")
    for k, v in bw.get("remote_gbps", {}).items():
        lines.append(f"  узел {k} читает память узла {bw.get('remote_target_node')}: {v}")
    lines.append("Кривая потоков (задач/с): " + ", ".join(f"{n}={r}" for n, r in report.get("curve", [])))
    lines.append(f"Излом кривой: {report.get('knee')} потоков")
    if report.get("warnings"):
        lines.append("Предупреждения:")
        lines += [f"  - {w}" for w in report["warnings"]]
    if report.get("skipped"):
        lines.append("Пропущено:")
        lines += [f"  - {s}" for s in report["skipped"]]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Зонд железа: NUMA, каналы памяти, пропускная способность")
    ap.add_argument("--out", default=None)
    ap.add_argument("--bw-mb", type=int, default=64)
    ap.add_argument("--bw-runs", type=int, default=5)
    args = ap.parse_args(argv)
    reconf = getattr(sys.stdout, "reconfigure", None)
    if reconf:
        reconf(errors="replace")
    report = collect(ROOT, args.bw_mb, args.bw_runs)
    host = re.sub(r"[^\w.-]", "_", str(report["host"]))
    out = Path(args.out) if args.out else ROOT / "state" / "hw" / f"probe_{host}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(format_summary(report))
    print(f"Отчёт: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
