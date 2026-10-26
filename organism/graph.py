from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .config import Genome


@dataclass
class Edge:
    w: float = 0.0
    trace: float = 0.0
    credit: float = 0.0
    uses: int = 0


class Graph:
    def __init__(self, genome: Genome, rng: random.Random | None = None):
        self.g = genome
        self.rng = rng or random.Random()
        self.edges: dict[tuple[str, str], Edge] = {}
        self.registry: dict[str, set[str]] = defaultdict(set)

    def edge(self, a: str, b: str) -> Edge:
        return self.edges.setdefault((a, b), Edge())

    def weight(self, a: str, b: str) -> float:
        e = self.edges.get((a, b))
        return e.w if e else 0.0

    def fire(self, a: str, b: str) -> None:
        e = self.edge(a, b)
        e.trace += 1.0
        e.uses += 1

    def decay_traces(self, dt: float) -> None:
        k = math.exp(-dt / self.g.trace_tau_s)
        for e in self.edges.values():
            e.trace *= k

    def reward(self, r: float) -> None:
        for e in self.edges.values():
            if e.trace > 1e-3:
                e.credit += r * e.trace

    def consolidate(self) -> dict[str, int]:
        changed = 0
        for e in self.edges.values():
            dw = self.g.eta * e.credit - self.g.decay * e.w
            dw = max(-self.g.max_dw, min(self.g.max_dw, dw))
            new = min(1.0, max(0.0, e.w + dw))
            changed += abs(new - e.w) > 1e-9
            e.w, e.credit, e.trace = new, 0.0, 0.0
        pruned = [k for k, e in self.edges.items() if e.uses == 0 and e.w < self.g.w_min]
        for k in pruned:
            del self.edges[k]
        for e in self.edges.values():
            e.uses = 0
        self._normalize_incoming()
        return {"изменено": changed, "срезано": len(pruned), "связей": len(self.edges)}

    def _normalize_incoming(self) -> None:
        total: dict[str, float] = defaultdict(float)
        for (_, b), e in self.edges.items():
            total[b] += e.w
        for (_, b), e in self.edges.items():
            if total[b] > 1.0:
                e.w /= total[b]

    def announce(self, address: str, professions: tuple[str, ...]) -> None:
        for p in professions:
            self.registry[p].add(address)

    def forget(self, address: str) -> None:
        for owners in self.registry.values():
            owners.discard(address)

    def route(self, source: str, profession: str) -> str | None:
        candidates = sorted(self.registry.get(profession, ()))
        if not candidates:
            return None
        if len(candidates) == 1 or self.rng.random() < self.g.explore:
            return self.rng.choice(candidates)
        best = max(self.weight(source, c) for c in candidates)
        return self.rng.choice([c for c in candidates if self.weight(source, c) == best])

    def save(self, path: Path) -> None:
        rows = [[a, b, round(e.w, 6)] for (a, b), e in sorted(self.edges.items()) if e.w > 0]
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    def load(self, path: Path) -> None:
        if path.exists():
            for a, b, w in json.loads(path.read_text(encoding="utf-8")):
                self.edge(a, b).w = float(w)
