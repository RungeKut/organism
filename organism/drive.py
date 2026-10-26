from __future__ import annotations

import hashlib
import random

from .context import Ctx
from .events import Event
from .stem import Phase
from .streams import Stream

DRIVE_CAP = 20
RECALL = "вспомнить"
SEEN_MAX = 512


class Drive(Stream):
    topics = ("результат",)
    professions = ()
    cores = 0.05
    protected = False
    clonable = False

    def __init__(self, name: str, ctx: Ctx):
        super().__init__(name, ctx)
        self.period = ctx.genome.vitals_every * 5
        self.rng = random.Random()
        self.counts: dict[str, int] = {}
        self.seen: dict[str, None] = {}
        self.acts = 0
        self.wake_since: float | None = None
        self.inputs = self._input_count()
        self.last_input: float | None = None
        self.busy_until = 0.0
        self.recalled: list[str] = []

    def _input_count(self) -> int:
        row = self.ctx.journal.db.execute("SELECT COUNT(*) FROM events WHERE topic = ? AND source = ?",
                                          ("вход", "пользователь")).fetchone()
        return row[0]

    def idle_s(self) -> float:
        return 5 * self.period * self.ctx.stem.tact_s

    def repertoire(self) -> list[str]:
        t = self.ctx.territory
        cmds = list(t.exec_allowlist) if t.exec_enabled else []
        return cmds + [RECALL]

    def choose(self) -> str:
        pool = self.repertoire()
        if self.rng.random() < self.ctx.genome.explore:
            cmd = self.rng.choice(pool)
        else:
            cmd = min(pool, key=lambda c: self.counts.get(c, 0))
        self.counts[cmd] = self.counts.get(cmd, 0) + 1
        return cmd

    async def pulse(self) -> None:
        c = self.ctx
        now = c.clock()
        n = self._input_count()
        if n != self.inputs:
            self.inputs = n
            self.last_input = now
        if c.stem.phase is not Phase.WAKE:
            return
        if c.stem.since != self.wake_since:
            self.wake_since = c.stem.since
            self.acts = 0
        if c.workspace.pending or now < self.busy_until or self.acts >= DRIVE_CAP:
            return
        if self.last_input is not None and now - self.last_input < self.idle_s():
            return
        cmd = self.choose()
        if cmd == RECALL:
            self.recalled = c.journal.memories()
            return
        self.acts += 1
        self.started = now
        self.busy_until = now + self.deadline_s
        self.propose("рука", {"cmd": cmd, "episode": "", "hops": 1, "reply_to": self.name}, salience=0.2)

    async def handle(self, ev: Event) -> None:
        self.busy_until = 0.0
        text = " ".join(str(ev.payload.get("text", "")).split())
        if not text:
            return
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if digest in self.seen:
            return
        self.seen[digest] = None
        if len(self.seen) > SEEN_MAX:
            self.seen.pop(next(iter(self.seen)))
        self.ctx.reward(0.1, "любопытство", stream=self.name)
