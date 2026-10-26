from __future__ import annotations

import copy
import dataclasses
import json
import secrets
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from .config import ROOT, Genome, Territory, _known, read_json
from .journal import Journal
from .llm import LLM
from .organism import Organism

BOUNDS: dict[str, tuple[float, float]] = {
    "eta": (0.01, 0.5),
    "decay": (0.0, 0.1),
    "trace_tau_s": (1.0, 300.0),
    "max_dw": (0.05, 1.0),
    "w_min": (0.0, 0.2),
    "explore": (0.0, 0.5),
    "tact_s": (0.05, 2.0),
    "sleep_s": (10.0, 600.0),
    "wake_s": (60.0, 7200.0),
    "max_chain": (1, 4),
}

PENDING = "generation_pending.json"


def _clip(name: str, value: float) -> float | int:
    lo, hi = BOUNDS[name]
    value = min(hi, max(lo, value))
    return int(round(value)) if isinstance(lo, int) else float(value)


def mutate(g: Genome, rng, rate: float = 0.2) -> Genome:
    kid = copy.deepcopy(g)
    for name, (lo, hi) in BOUNDS.items():
        value = getattr(kid, name)
        if rng.random() < rate:
            value = value + rng.gauss(0.0, 0.1 * (hi - lo))
        setattr(kid, name, _clip(name, value))
    return kid


def inputs(j: Journal, t0: float) -> list[str]:
    return [p["text"] for _, source, p in j.events_since(t0, "вход")
            if source == "пользователь" and isinstance(p.get("text"), str)]


async def trial(g: Genome, texts: list[str], llm: LLM, tmp: Path) -> float:
    if not texts:
        return 0.0
    territory = dataclasses.replace(Territory.load(ROOT / "config" / "territory.json"), exec_enabled=False)
    org = Organism(Path(tmp), llm, territory, copy.deepcopy(g))
    answered = 0
    try:
        await org.start()
        for text in texts:
            if await org.say(text) is not None:
                answered += 1
        actions = org.journal.all_actions_since(0)
    finally:
        await org.stop()
    expired = sum(1 for _, kind, _, _ in actions if kind == "просрочено")
    commands = [ok for _, kind, ok, _ in actions if kind == "выполнить"]
    done = sum(commands) / len(commands) if commands else 0.0
    n = len(texts)
    return answered / n - min(expired, n) / n + done


def select(scored: list[tuple[float, Genome]], parent_score: float,
           max_children: int = 1) -> list[tuple[float, Genome]]:
    better = sorted((s for s in scored if s[0] > parent_score), key=lambda s: s[0], reverse=True)
    return better[:max(0, max_children)]


async def breed(parent: Genome, texts: list[str], llm: LLM, rng, n: int = 2,
                max_children: int = 1, rate: float = 0.2) -> tuple[float, list[tuple[float, Genome]]]:
    with tempfile.TemporaryDirectory(prefix="поколение-") as work:
        base = await trial(parent, texts, llm, Path(work) / "родитель")
        scored = []
        for k in range(n):
            kid = mutate(parent, rng, rate)
            scored.append((await trial(kid, texts, llm, Path(work) / f"потомок-{k}"), kid))
    return base, select(scored, base, max_children)


def propose(state: Path, parent: Genome, kids: list[Genome], scores: list[float] | None = None,
            max_children: int = 1) -> Path:
    if not kids:
        raise ValueError("нет детей для смены поколения")
    scores = list(scores) if scores is not None else [0.0] * len(kids)
    if len(scores) != len(kids):
        raise ValueError("число счетов не равно числу детей")
    ranked = sorted(zip(scores, kids), key=lambda s: s[0], reverse=True)[:max(1, max_children)]
    path = Path(state) / PENDING
    data = {
        "token": secrets.token_hex(16),
        "t": time.time(),
        "parent": asdict(parent),
        "kids": [{"score": float(s), "genome": asdict(k)} for s, k in ranked],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def pending(state: Path) -> dict | None:
    path = Path(state) / PENDING
    return read_json(path) if path.exists() else None


def approve(state: Path, token: str) -> bool:
    path = Path(state) / PENDING
    data = pending(state)
    if data is None or not data.get("kids") or not isinstance(token, str):
        return False
    if not secrets.compare_digest(str(data.get("token", "")), token):
        return False
    best = max(data["kids"], key=lambda k: k["score"])
    genome = Genome(**_known(Genome, best["genome"]))
    for name in BOUNDS:
        setattr(genome, name, _clip(name, getattr(genome, name)))
    genome.save(Path(state) / "genome.json")
    path.unlink()
    return True
