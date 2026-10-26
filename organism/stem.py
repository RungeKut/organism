from __future__ import annotations

import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator

from .config import Genome, Territory


class Phase(Enum):
    WAKE = "бодрствование"
    DROWSY = "засыпание"
    SLEEP = "сон"


@dataclass
class Grant:
    ok: bool
    reason: str = ""


class Stem:
    def __init__(self, territory: Territory, genome: Genome, clock: Callable[[], float] = time.monotonic):
        self.t = territory
        self.g = genome
        self.clock = clock
        self.phase = Phase.WAKE
        self.since = clock()
        self.in_flight = 0
        self.events_since_sleep = 0
        self.reserved: dict[str, float] = {}
        self._actions: deque[float] = deque()
        self._sleep_done = False
        self._tact = genome.tact_s
        self.request_tact(genome.tact_s)

    @property
    def tact_s(self) -> float:
        return self._tact

    def request_tact(self, seconds: float) -> float:
        lo, hi = self.t.tact_bounds_s
        self._tact = min(hi, max(lo, float(seconds)))
        return self._tact

    def accepts_input(self) -> bool:
        return self.phase is Phase.WAKE

    @contextmanager
    def task(self) -> Iterator[None]:
        self.in_flight += 1
        try:
            yield
        finally:
            self.in_flight -= 1

    def note_event(self) -> None:
        self.events_since_sleep += 1

    def action_denied(self) -> str | None:
        now = self.clock()
        while self._actions and now - self._actions[0] > 3600:
            self._actions.popleft()
        if len(self._actions) >= self.t.max_actions_per_hour:
            return "лимит действий в час"
        self._actions.append(now)
        return None

    def request_stream(self, name: str, cores: float, capacity: float) -> Grant:
        if name in self.reserved:
            return Grant(True, "уже живёт")
        if len(self.reserved) >= self.t.max_streams:
            return Grant(False, "лимит потоков территории")
        if sum(self.reserved.values()) + cores > capacity * (1.0 - self.t.headroom):
            return Grant(False, "нет запаса ядер")
        self.reserved[name] = cores
        return Grant(True)

    def release_stream(self, name: str) -> None:
        self.reserved.pop(name, None)

    def sleep_done(self) -> None:
        self._sleep_done = True

    def tick(self, busy: int) -> Phase | None:
        elapsed = self.clock() - self.since
        if self.phase is Phase.WAKE:
            if elapsed >= self.g.wake_s or self.events_since_sleep >= self.g.sleep_pressure_events:
                return self._enter(Phase.DROWSY)
        elif self.phase is Phase.DROWSY:
            if busy == 0 or elapsed >= self.g.drowsy_max_s:
                return self._enter(Phase.SLEEP)
        elif self._sleep_done and elapsed >= self.g.sleep_s:
            return self._enter(Phase.WAKE)
        return None

    def _enter(self, phase: Phase) -> Phase:
        self.phase = phase
        self.since = self.clock()
        if phase is Phase.SLEEP:
            self._sleep_done = False
            self.events_since_sleep = 0
        return phase
