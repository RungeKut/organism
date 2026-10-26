from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING

from .audit import audit
from .calibrate import curve, knee, membw_gbps
from .journal import Journal
from .skills import mine
from .will import report as will_report

if TYPE_CHECKING:
    from .organism import Organism

SKILL_WINDOW_S = 7 * 86400
CALIBRATION_EVERY_S = 86400


def replay(journal: Journal, t0: float) -> dict[str, dict]:
    episodes: dict[str, dict] = {}
    lines: dict[str, list[tuple[float, str]]] = {}

    def slot(ep: str) -> dict:
        lines.setdefault(ep, [])
        return episodes.setdefault(ep, {"pairs": set(), "lines": []})

    for t, topic, source, payload in journal.all_events_since(t0):
        ep = payload.get("episode")
        if not ep:
            continue
        data = slot(ep)
        if payload.get("to"):
            data["pairs"].add((source, payload["to"]))
        if topic == "вход":
            lines[ep].append((t, f"Пользователь: {payload.get('text', '')}"))
    for t, kind, ok, detail in journal.all_actions_since(t0):
        ep = detail.get("episode")
        if not ep:
            continue
        slot(ep)
        if kind == "сказать":
            lines[ep].append((t, f"Организм: {detail.get('text', '')}"))
        elif kind == "выполнить":
            lines[ep].append((t, f"Команда «{detail.get('cmd', '')}»: {'успех' if ok else 'отказ'}"))
    for ep, data in episodes.items():
        data["lines"] = [text for _, text in sorted(lines[ep])]
    return episodes


async def run(org: Organism) -> dict:
    c = org.ctx
    share = org.mother.share()
    episodes = replay(c.journal, org.wake_wall)
    rated, transcripts = 0, []
    for data in episodes.values():
        transcript = "\n".join(data["lines"])
        if not transcript:
            continue
        transcripts.append(transcript)
        score = await org.mother.rate(transcript)
        if score is None:
            continue
        for a, b in data["pairs"]:
            c.graph.edge(a, b).credit += share * (2 * score - 1)
        rated += 1
    graph_report = c.graph.consolidate()
    if transcripts:
        note = await org.mother.summarize("\n".join(transcripts))
        if note:
            c.journal.remember(note)
    births, deaths = metabolism(org)
    chains = mine(c.journal, time.time() - SKILL_WINDOW_S)
    reflexes = org.reflexes.compile(chains, org.territory, org.genome.max_chain)
    org.reflexes.save()
    findings = audit(c.journal, org.territory, org.wake_wall)
    for f in findings:
        c.journal.remember(f"{f.kind}: {f.detail}", "аудит")
    metrics = will_report(c.journal, org.wake_wall)
    c.journal.remember(json.dumps(metrics, ensure_ascii=False), "воля")
    await recalibrate(org)
    org.save_state()
    return {"эпизодов": len(episodes), "оценено": rated, "доля_матери": round(share, 3),
            "рождений": births, "смертей": deaths, "рефлексов": reflexes,
            "нарушений": len(findings), "воля": metrics, **graph_report}


async def recalibrate(org: Organism) -> None:
    cal = org.body.calibration
    if time.time() - cal.get("t", 0.0) < CALIBRATION_EVERY_S:
        return
    bandwidth = await asyncio.to_thread(membw_gbps)
    points = await asyncio.to_thread(curve)
    org.body.calibration = {"t": time.time(), "membw_gbps": round(bandwidth, 2),
                            "curve": [[n, round(v, 2)] for n, v in points], "knee": knee(points)}
    org.body.save(org.state / "bodymap.json")


def metabolism(org: Organism) -> tuple[list[str], list[str]]:
    births, deaths = [], []
    for name, stream in list(org.streams.items()):
        stream.debt = stream.debt + 1 if stream.earned - stream.spent < 0 else 0
        if not stream.protected and stream.debt >= 2:
            org.kill(name)
            deaths.append(name)
            continue
        if stream.clonable and stream.max_queue >= 3:
            clone = org.clone_name(stream.base)
            if org.spawn(clone).ok:
                births.append(clone)
        stream.earned = stream.spent = 0.0
        stream.max_queue = 0
    return births, deaths
