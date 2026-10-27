from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from organism.config import Genome, Territory
from organism.llm import AIMD, LlamaCpp
from organism.organism import Organism
from organism.stem import Phase

PHRASES = (
    "Привет! Кто ты?",
    "Расскажи в двух предложениях, что ты умеешь.",
    "Выполни команду проверки версии git.",
    "Что ты помнишь из нашего разговора?",
)


def healthy(url: str) -> bool:
    try:
        urllib.request.urlopen(url.rstrip("/") + "/health", timeout=2).close()
    except OSError:
        return False
    return True


async def timed_say(org: Organism, text: str) -> dict:
    t0 = time.perf_counter()
    reply = await org.say(text, timeout=120.0)
    return {"вопрос": text, "ответ": reply, "сек": round(time.perf_counter() - t0, 2)}


async def timed_burst(org: Organism, phrases: tuple[str, ...], parallel: int) -> dict:
    t0 = time.perf_counter()
    burst = await asyncio.gather(*(timed_say(org, text) for text in (phrases * parallel)[:parallel]))
    return {"запросов": len(burst), "сек": round(time.perf_counter() - t0, 2), "ответы": burst}


async def nap(org: Organism, timeout: float = 900.0) -> dict:
    t0, wake_s = time.perf_counter(), org.genome.wake_s
    org.genome.wake_s = 0.0
    try:
        while org.stem.phase is not Phase.SLEEP:
            await asyncio.sleep(0.1)
    finally:
        org.genome.wake_s = wake_s
    while org.stem.phase is not Phase.WAKE and time.perf_counter() - t0 < timeout:
        await asyncio.sleep(0.5)
    return {"сек": round(time.perf_counter() - t0, 2), "фаза": org.stem.phase.value,
            "отчёт": org.last_sleep_report, "потоки": sorted(org.streams),
            "калибровка": org.body.calibration}


async def run(url: str, phrases: tuple[str, ...] = PHRASES, parallel: int = 4, cycle: bool = False,
              genome: Genome | None = None) -> dict:
    territory = Territory.load(ROOT / "config" / "territory.json")
    llm = LlamaCpp(url, AIMD(territory.max_llm_slots), territory.allow_network)
    with tempfile.TemporaryDirectory() as tmp:
        org = Organism(tmp, llm, territory, genome or Genome(), seed=1)
        await org.start()
        try:
            serial = [await timed_say(org, text) for text in phrases]
            burst = await timed_burst(org, phrases, parallel)
            extra = {}
            if cycle:
                extra["сон"] = await nap(org)
                extra["пачка_после_сна"] = await timed_burst(org, phrases, parallel)
            return {
                "url": url,
                "по_очереди": serial,
                "пачка": burst,
                **extra,
                "last_tps": llm.last_tps,
                "слоты": {"предел": llm.slots.limit, "потолок": llm.slots.ceiling},
                "lag_ms": round(org.body.loop_lag_ms, 2),
                "снимок": org.snapshot(),
            }
        finally:
            await org.stop()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Короткий прогон организма на настоящем llama-server")
    ap.add_argument("--url", default=None)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cycle", action="store_true", help="после пачки уснуть и повторить пачку")
    args = ap.parse_args(argv)
    url = args.url or Territory.load(ROOT / "config" / "territory.json").llm_url
    if not healthy(url):
        print(f"llama-server не отвечает на {url}", file=sys.stderr)
        return 2
    report = asyncio.run(run(url, parallel=args.parallel, cycle=args.cycle))
    text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
