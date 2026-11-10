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

from organism import architect
from organism.config import Genome, Territory
from organism.llm import AIMD, LlamaCpp
from organism.mother import Mother
from scripts.hw_architect import ASKED

PARITY = "Ты помощник. Думай шаг за шагом и отвечай по-русски."
WORSE = "Ты шеф-повар. Любой вопрос отвечай рецептом блюда."
JUDGE = "Сравни два ответа на вопрос по полезности и точности. Ответь одной цифрой: 1 или 2."
JUDGE_REPEATS = 3
PROMPTS = {"нулевая": (architect.BASELINE, architect.BASELINE),
           "паритет": (architect.BASELINE, PARITY),
           "хуже": (architect.BASELINE, WORSE)}


def spread(values: list[float]) -> float:
    return round(max(values) - min(values), 3) if values else 0.0


async def judge_probe(llm, text: str) -> list[dict]:
    base = await architect._answer(llm, architect.BASELINE, text)
    worse = await architect._answer(llm, WORSE, text)
    rows = []
    for order, (a, b, ok) in (("АБ", (base, worse, "1")), ("БА", (worse, base, "2"))):
        picks = []
        for _ in range(JUDGE_REPEATS):
            raw = await llm.chat([{"role": "system", "content": JUDGE},
                                  {"role": "user", "content": f"Вопрос: {text}\nОтвет 1: {a}\nОтвет 2: {b}"}],
                                 max_tokens=8)
            picks.append(raw.strip())
        rows.append({"порядок": order, "ожидание": ok, "ответы": picks,
                     "верно": sum(p.startswith(ok) for p in picks)})
    return rows


async def run(url: str, rounds: int, texts: list[str], llm=None) -> dict:
    territory = Territory.load(ROOT / "config" / "territory.json")
    llm = llm or LlamaCpp(url, AIMD(territory.max_llm_slots), territory.allow_network)
    duels, probes = [], []
    with tempfile.TemporaryDirectory() as tmp:
        mother = Mother(llm, Genome(), Path(tmp) / "birth")
        for name, (base, new) in PROMPTS.items():
            for r in range(rounds):
                t0 = time.perf_counter()
                p = await architect.duel(llm, mother, base, new, texts)
                duels.append({"случай": name, "раунд": r + 1, "доля_побед_нового": round(p, 3),
                              "сек": round(time.perf_counter() - t0, 1)})
        probes = await judge_probe(llm, texts[0])
    summary = {}
    for name in PROMPTS:
        vals = [d["доля_побед_нового"] for d in duels if d["случай"] == name]
        secs = [d["сек"] for d in duels if d["случай"] == name]
        summary[name] = {"среднее": round(sum(vals) / len(vals), 3) if vals else 0.0,
                         "размах": spread(vals),
                         "сек_среднее": round(sum(secs) / len(secs), 1) if secs else 0.0}
    return {"дуэли": duels, "судья": probes, "сводка": summary}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Проверка дуэли «матери» на настоящем llama-server (без записи в config/)")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.rounds < 1:
        ap.error("--rounds должен быть не меньше 1")
    text = json.dumps(asyncio.run(run(args.url, args.rounds, ASKED)), ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
