from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Proposal:
    stream: str
    effector: str
    payload: dict[str, Any]
    salience: float
    value: float
    born: float
    deadline: float

    @property
    def score(self) -> float:
        return self.salience * self.value


@dataclass
class Workspace:
    pending: list[Proposal] = field(default_factory=list)
    scene: deque = field(default_factory=lambda: deque(maxlen=8))

    def propose(self, p: Proposal) -> None:
        self.pending.append(p)

    def select(self, now: float) -> tuple[list[Proposal], list[Proposal]]:
        expired = [p for p in self.pending if p.deadline < now]
        self.pending = [p for p in self.pending if p.deadline >= now]
        winners: dict[str, Proposal] = {}
        for p in self.pending:
            best = winners.get(p.effector)
            if best is None or (p.score, -p.born) > (best.score, -best.born):
                winners[p.effector] = p
        taken = {id(p) for p in winners.values()}
        self.pending = [p for p in self.pending if id(p) not in taken]
        return list(winners.values()), expired

    def post(self, stream: str, effector: str, text: str) -> None:
        self.scene.append({"поток": stream, "орган": effector, "текст": text})
