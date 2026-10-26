from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from .body import BodyMap
from .config import Genome, Territory
from .events import Bus
from .executor import Executor
from .graph import Graph
from .journal import Journal
from .llm import LLM
from .stem import Stem
from .workspace import Workspace


@dataclass
class Ctx:
    territory: Territory
    genome: Genome
    stem: Stem
    graph: Graph
    bus: Bus
    journal: Journal
    llm: LLM
    workspace: Workspace
    executor: Executor
    body: BodyMap
    streams: dict = field(default_factory=dict)
    lag_ms: float = 0.0
    reflexes: Any = None
    stats: dict = field(default_factory=lambda: defaultdict(int))
    clock: Callable[[], float] = time.monotonic

    def reward(self, r: float, source: str, stream: str | None = None) -> None:
        self.journal.reward(r, source)
        self.graph.reward(r)
        if stream and r > 0 and stream in self.streams:
            self.streams[stream].earned += r
