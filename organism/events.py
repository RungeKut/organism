from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .graph import Graph


@dataclass
class Event:
    topic: str
    source: str
    payload: dict[str, Any] = field(default_factory=dict)
    t: float = field(default_factory=time.time)


class Bus:
    def __init__(self, graph: Graph, sinks: list[Callable[[Event], None]] | None = None):
        self.graph = graph
        self.sinks = sinks or []
        self._queues: dict[str, asyncio.Queue] = {}
        self._topics: dict[str, set[str]] = {}

    def subscribe(self, owner: str, topics: tuple[str, ...], queue: asyncio.Queue) -> None:
        self._queues[owner] = queue
        for topic in topics:
            self._topics.setdefault(topic, set()).add(owner)

    def unsubscribe(self, owner: str) -> None:
        self._queues.pop(owner, None)
        for owners in self._topics.values():
            owners.discard(owner)

    def publish(self, ev: Event) -> int:
        self._record(ev)
        delivered = 0
        for owner in sorted(self._topics.get(ev.topic, ())):
            if owner != ev.source:
                self._deliver(ev, owner)
                delivered += 1
        return delivered

    def send(self, ev: Event, to: str) -> bool:
        if to not in self._queues:
            return False
        ev.payload.setdefault("to", to)
        self._record(ev)
        self._deliver(ev, to)
        return True

    def _record(self, ev: Event) -> None:
        for sink in self.sinks:
            sink(ev)

    def _deliver(self, ev: Event, owner: str) -> None:
        self.graph.fire(ev.source, owner)
        self._queues[owner].put_nowait(ev)
