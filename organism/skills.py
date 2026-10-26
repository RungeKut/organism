from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .config import Territory
from .journal import Journal

MIN_OK_RATE = 0.9


@dataclass(frozen=True)
class Chain:
    key: str
    cmds: tuple[str, ...]
    n: int
    ok_rate: float


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def mine(j: Journal, t0: float, min_count: int = 3) -> list[Chain]:
    texts: dict[str, str] = {}
    cmds: dict[str, list[str]] = defaultdict(list)
    oks: dict[str, list[bool]] = defaultdict(list)
    for _, topic, _, payload in j.all_events_since(t0):
        ep = payload.get("episode")
        if ep and topic == "вход" and ep not in texts:
            key = normalize(str(payload.get("text", "")))
            if key:
                texts[ep] = key
    for _, kind, ok, detail in j.all_actions_since(t0):
        ep = detail.get("episode")
        if ep and kind == "выполнить" and detail.get("cmd"):
            cmds[ep].append(str(detail["cmd"]))
            oks[ep].append(ok)
    groups: dict[tuple[str, tuple[str, ...]], list[bool]] = defaultdict(list)
    for ep, key in texts.items():
        if cmds.get(ep):
            groups[(key, tuple(cmds[ep]))].append(all(oks[ep]))
    chains = [Chain(key, seq, len(res), sum(res) / len(res))
              for (key, seq), res in groups.items() if len(res) >= min_count]
    return sorted(chains, key=lambda c: (-c.n, c.key, c.cmds))


class ReflexCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.reflexes: dict[str, tuple[str, ...]] = {}
        self.load()

    def load(self) -> None:
        self.reflexes = {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        for key, seq in data.items():
            if isinstance(seq, list) and seq and all(isinstance(c, str) for c in seq):
                self.reflexes[key] = tuple(seq)

    def save(self) -> None:
        data = {k: list(v) for k, v in self.reflexes.items()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def match(self, text: str) -> tuple[str, ...] | None:
        return self.reflexes.get(normalize(text))

    def compile(self, chains: list[Chain], territory: Territory, max_chain: int) -> int:
        allowed = set(territory.exec_allowlist)
        variants: dict[str, set[tuple[str, ...]]] = defaultdict(set)
        for c in chains:
            variants[normalize(c.key)].add(c.cmds)
        accepted = 0
        for c in chains:
            key = normalize(c.key)
            if not key or len(variants[key]) > 1:
                continue
            if not 0 < len(c.cmds) <= max_chain or c.ok_rate < MIN_OK_RATE:
                continue
            if not all(cmd in allowed for cmd in c.cmds):
                continue
            self.reflexes[key] = tuple(c.cmds)
            accepted += 1
        return accepted
