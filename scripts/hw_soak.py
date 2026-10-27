from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from organism import generations
from organism.config import Genome, Territory
from organism.llm import AIMD, LlamaCpp
from organism.organism import Organism
from scripts.hw_live import PHRASES, healthy


def rss_mb() -> float:
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
                (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                               "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                               "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        c = Counters(cb=ctypes.sizeof(Counters))
        psapi, k32 = ctypes.WinDLL("psapi"), ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
        psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return round(c.WorkingSetSize / 2**20, 1)
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    return 0.0


def dir_mb(path: Path) -> float:
    return round(sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 2**20, 2)


async def soak(url: str, state: Path, hours: float, every_s: float, generations_every: int,
               out: Path | None, genome: Genome | None = None) -> dict:
    territory = replace(Territory.load(ROOT / "config" / "territory.json"),
                        generation_every_sleeps=generations_every)
    llm = LlamaCpp(url, AIMD(territory.max_llm_slots), territory.allow_network)
    org = Organism(state, llm, territory, genome or Genome(), seed=1)
    await org.start()
    t0, n, seen_sleeps = time.monotonic(), 0, 0
    totals = {"вопросов": 0, "пустых": 0, "отклонено": 0, "нарушений": 0}
    sleeps, rss = [], []
    try:
        while time.monotonic() - t0 < hours * 3600:
            text = PHRASES[n % len(PHRASES)]
            n += 1
            ts = time.perf_counter()
            if org.stem.accepts_input():
                reply = await org.say(text, timeout=120.0)
                totals["вопросов"] += 1
                totals["пустых"] += not reply
            else:
                reply = None
                totals["отклонено"] += 1
            if org.sleeps > seen_sleeps:
                seen_sleeps = org.sleeps
                report = org.last_sleep_report or {}
                totals["нарушений"] += int(report.get("нарушений", 0))
                sleeps.append({k: report.get(k) for k in ("эпизодов", "оценено", "рождений", "смертей",
                                                           "нарушений", "изменено", "срезано")})
            sample = {"t_мин": round((time.monotonic() - t0) / 60, 1), "фаза": org.stem.phase.value,
                      "сны": org.sleeps, "ответ_сек": round(time.perf_counter() - ts, 2), "ответ": bool(reply),
                      "lag_ms": round(org.body.loop_lag_ms, 1), "rss_mb": rss_mb(), "state_mb": dir_mb(state),
                      "потоки": sorted(org.streams), "tps": round(llm.last_tps, 2)}
            rss.append(sample["rss_mb"])
            if out:
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            await asyncio.sleep(max(0.0, every_s - (time.perf_counter() - ts)))
    finally:
        await org.stop()
    pending = generations.pending(state)
    saved = state / "genome.json"
    return {**totals, "геном_прежний": not saved.exists() or Genome.load(saved) == org.genome, "мин": round((time.monotonic() - t0) / 60, 1), "снов": org.sleeps, "сны": sleeps,
            "rss_mb": {"начало": rss[0] if rss else 0, "конец": rss[-1] if rss else 0, "макс": max(rss, default=0)},
            "state_mb": dir_mb(state), "поколение_ждёт": bool(pending),
            "детей": len(pending.get("kids", [])) if pending else 0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Долгий прогон организма на настоящем llama-server")
    ap.add_argument("--url", default=None)
    ap.add_argument("--state", required=True)
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--every", type=float, default=60.0, help="секунд между вопросами")
    ap.add_argument("--generations", type=int, default=0, help="generation_every_sleeps на время прогона")
    ap.add_argument("--log", default=None, help="jsonl с замерами")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    url = args.url or Territory.load(ROOT / "config" / "territory.json").llm_url
    if not healthy(url):
        print(f"llama-server не отвечает на {url}", file=sys.stderr)
        return 2
    report = asyncio.run(soak(url, Path(args.state), args.hours, args.every, args.generations,
                              Path(args.log) if args.log else None))
    text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
