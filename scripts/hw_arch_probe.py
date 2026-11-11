from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from organism import architect
from organism.config import Territory
from organism.llm import AIMD, LlamaCpp
from organism.organism import RESERVED, STREAM_TYPES

ASKED = ["Составь короткий договор аренды склада.", "Посчитай маржу: закупка 70, продажа 100, налог 6%.",
         "Как проверить, что pull request безопасен?", "Переведи на английский: «срок поставки изменён».",
         "Выполни команду проверки версии git."]


async def probe(url: str, rounds: int, llm=None) -> dict:
    territory = Territory.load(ROOT / "config" / "territory.json")
    llm = llm or LlamaCpp(url, AIMD(territory.max_llm_slots), territory.allow_network)
    before = (ROOT / "config" / "professions.json").read_text(encoding="utf-8")
    summary = architect.summary(["сон: эпизодов 10, оценено 10", "сон: эпизодов 10, изменено 3"],
                                {"задач": 1080, "снов": 108}, ASKED, {})
    reserved = RESERVED | frozenset(STREAM_TYPES)
    rows = []
    for i in range(rounds):
        raw = await llm.chat([{"role": "system", "content": architect.SYSTEM},
                              {"role": "user", "content": f"Профессии:\n{before}\n\nСводка:\n{summary}"}],
                             max_tokens=700)
        ext = architect.extract(raw)
        row = architect.changed(before, ext, reserved) if ext else None
        rows.append({"сырое": raw.strip(), "extract_ok": ext is not None, "changed_ok": row is not None,
                     "имя": row["name"] if row else None})
    return {"раунды": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Зонд архитектора: сырой ответ, extract и changed (без записи)")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    text = json.dumps(asyncio.run(probe(args.url, args.rounds)), ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
