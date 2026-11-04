from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from organism import proposals
from organism.config import Genome, Territory
from organism.llm import AIMD, LlamaCpp
from organism.organism import Organism


async def run(url: str, rounds: int, llm=None) -> dict:
    territory = Territory.load(ROOT / "config" / "territory.json")
    llm = llm or LlamaCpp(url, AIMD(territory.max_llm_slots), territory.allow_network)
    report = {"раунды": []}
    with tempfile.TemporaryDirectory() as tmp:
        state = Path(tmp)
        org = Organism(state, llm, territory, Genome())
        for _ in range(rounds):
            t0 = time.perf_counter()
            n0 = len(org.journal.all_events_since(0))
            await org._architect()
            events = [(topic, payload) for _, topic, _, payload in org.journal.all_events_since(0)[n0:]
                      if topic.startswith("архитектор")]
            report["раунды"].append({"сек": round(time.perf_counter() - t0, 1), "события": events})
        report["предложения"] = [{k: v for k, v in row.items() if k != "token"} for row in proposals.pending(state)]
        org.journal.close()
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Архитектор на настоящем llama-server (без записи в config/)")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    text = json.dumps(asyncio.run(run(args.url, args.rounds)), ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
