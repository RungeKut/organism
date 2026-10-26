from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import urllib.request

from . import generations
from .body import inventory
from .config import ROOT, Genome, Territory
from .llm import AIMD, LLM, FakeLLM, LlamaCpp
from .organism import Organism
from .stem import Phase


def pick_llm(territory: Territory, fake: bool) -> LLM:
    if fake:
        return FakeLLM()
    try:
        urllib.request.urlopen(territory.llm_url.rstrip("/") + "/health", timeout=1).close()
    except OSError:
        print(f"[ствол] llama.cpp не отвечает на {territory.llm_url}: говорю заглушкой FakeLLM")
        return FakeLLM()
    return LlamaCpp(territory.llm_url, AIMD(territory.max_llm_slots), territory.allow_network)


async def chat(org: Organism) -> None:
    print("Команды: /состояние, /плюс, /минус, /выход")
    while True:
        try:
            line = (await asyncio.to_thread(input, "> ")).strip()
        except EOFError:
            return
        if line == "/выход":
            return
        if line == "/состояние":
            print(json.dumps(org.snapshot(), ensure_ascii=False, indent=2))
        elif line in ("/плюс", "/минус"):
            org.feedback(1.0 if line == "/плюс" else -1.0)
        elif line:
            reply = await org.say(line)
            print(reply if reply is not None else f"[{org.stem.phase.value}] ответа нет")


async def demo() -> None:
    genome = Genome(tact_s=0.05, wake_s=2.0, sleep_s=1.0, drowsy_max_s=5.0)
    with tempfile.TemporaryDirectory() as tmp:
        org = Organism(tmp, FakeLLM(), Territory.load(ROOT / "config" / "territory.json"), genome, seed=1)
        await org.start()
        print("фаза:", org.stem.phase.value)
        for text in ("привет", "выполни команду", "что ты умеешь"):
            print(f"> {text}\n{await org.say(text)}")
            org.feedback(1.0)
        while org.stem.phase is not Phase.SLEEP:
            await asyncio.sleep(0.1)
        print("фаза:", org.stem.phase.value, "| вход во сне:", await org.say("ты спишь?"))
        while org.stem.phase is not Phase.WAKE:
            await asyncio.sleep(0.1)
        print("фаза:", org.stem.phase.value)
        print("отчёт сна:", json.dumps(org.last_sleep_report, ensure_ascii=False))
        print(json.dumps(org.snapshot(), ensure_ascii=False, indent=2))
        await org.stop()


async def live(fake: bool) -> None:
    territory = Territory.load(ROOT / "config" / "territory.json")
    org = Organism(ROOT / "state", pick_llm(territory, fake), territory)
    await org.start()
    try:
        await chat(org)
    finally:
        await org.stop()


def main() -> None:
    parser = argparse.ArgumentParser(prog="organism")
    parser.add_argument("--fake", action="store_true", help="не ходить в llama.cpp, говорить заглушкой")
    parser.add_argument("--demo", action="store_true", help="короткий прогон бодрствование → сон → бодрствование")
    parser.add_argument("--body", action="store_true", help="показать карту тела и выйти")
    parser.add_argument("--approve", metavar="ТОКЕН", help="подтвердить смену поколения (организм остановлен)")
    args = parser.parse_args()
    if args.approve:
        print("смена поколения принята" if generations.approve(ROOT / "state", args.approve) else "токен не подошёл")
    elif args.body:
        print(json.dumps(inventory().__dict__, ensure_ascii=False, indent=2))
    elif args.demo:
        asyncio.run(demo())
    else:
        asyncio.run(live(args.fake))


if __name__ == "__main__":
    main()
