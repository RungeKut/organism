from __future__ import annotations

from collections import deque
from enum import Enum

from .context import Ctx

THINK_BELOW = 0.3
WINDOW = 50
HALF_LIFE_S = 5.0
RISE = 0.8
MISS_IMPULSE = 0.35

BUDGET = {"рефлекс": 0, "навык": 64, "размышление": 256}


class Rung(str, Enum):
    REFLEX = "рефлекс"
    SKILL = "навык"
    THINK = "размышление"


class Ladder:
    def __init__(self, ctx: Ctx):
        self.ctx = ctx
        self.level = 0.0
        self.seen_misses = ctx.stats["просрочено"]
        self.stamp = ctx.clock()
        self.recent: deque[Rung] = deque(maxlen=WINDOW)

    def instant(self) -> float:
        c = self.ctx
        pain_lag = max(c.genome.pain_lag_ms, 1e-9)
        x = max(0.0, c.lag_ms) / pain_lag
        lag = 0.6 * x * x if x <= 1.0 else 0.6 + 0.4 * min(1.0, x - 1.0)
        base = max(c.genome.tact_s, c.stem.t.tact_bounds_s[0], 1e-9)
        tact = min(1.0, max(0.0, (c.stem.tact_s / base - 1.0) / 2.0))
        return max(lag, tact)

    def pressure(self) -> float:
        c = self.ctx
        now = c.clock()
        dt = max(0.0, now - self.stamp)
        self.stamp = now
        self.level *= 0.5 ** (dt / HALF_LIFE_S)
        target = self.instant()
        if target > self.level:
            self.level += RISE * (target - self.level)
        misses = c.stats["просрочено"]
        fresh = max(0, misses - self.seen_misses)
        self.seen_misses = misses
        if fresh:
            self.level += MISS_IMPULSE * fresh
        self.level = min(1.0, max(0.0, self.level))
        return self.level

    def pick(self, text: str) -> Rung:
        reflexes = self.ctx.reflexes
        pressure = self.pressure()
        if reflexes is not None and reflexes.match(text) is not None:
            rung = Rung.REFLEX
        elif pressure < THINK_BELOW:
            rung = Rung.THINK
        else:
            rung = Rung.SKILL
        self.recent.append(rung)
        return rung

    def max_tokens(self, rung: Rung) -> int:
        return BUDGET[rung.value]

    def system_hint(self, rung: Rung) -> str:
        if rung is Rung.SKILL:
            return "Отвечай одной фразой, не более 20 слов."
        return ""

    def share(self) -> dict[Rung, float]:
        n = len(self.recent)
        return {r: (sum(1 for x in self.recent if x is r) / n if n else 0.0) for r in Rung}
