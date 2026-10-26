from __future__ import annotations

from typing import TYPE_CHECKING

from .journal import Journal

if TYPE_CHECKING:
    from .organism import Organism


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
    org.save_state()
    return {"эпизодов": len(episodes), "оценено": rated, "доля_матери": round(share, 3),
            "рождений": births, "смертей": deaths, **graph_report}


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
