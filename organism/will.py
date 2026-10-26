from __future__ import annotations

import math
from collections import Counter

from .journal import Journal

_NOT_ACTS = {"отклонено", "сбой"}


def _label(kind: str, detail: dict) -> str:
    cmd = detail.get("cmd")
    return f"{kind}:{cmd}" if cmd else kind


def _acts(j: Journal, t0: float) -> list[tuple[str, dict]]:
    return [(_label(kind, d), d) for _, kind, _, d in j.all_actions_since(t0) if kind not in _NOT_ACTS]


def diversity(j: Journal, t0: float) -> float:
    counts = Counter(label for label, d in _acts(j, t0) if not d.get("episode"))
    total = sum(counts.values())
    if len(counts) < 2:
        return 0.0
    h = -sum(c / total * math.log(c / total) for c in counts.values())
    return h / math.log(len(counts))


def persistence(before: list[str], after: list[str]) -> float:
    a, b = set(before), set(after)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def trend(j: Journal, t0: float) -> float:
    rows = j.db.execute("SELECT t, value FROM rewards WHERE t >= ? ORDER BY id", (t0,)).fetchall()
    n = len(rows)
    if n < 2:
        return 0.0
    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den < 1e-12:
        xs = [float(i) for i in range(n)]
        mx = (n - 1) / 2
        den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den


def loop_score(j: Journal, t0: float, n: int = 3) -> float:
    seq = [label for label, _ in _acts(j, t0)]
    size = len(seq)
    if size == 0:
        return 0.0
    covered = [False] * size
    for p in range(1, n + 1):
        i = 0
        while i + p < size:
            if seq[i] != seq[i + p]:
                i += 1
                continue
            start = i
            while i + p < size and seq[i] == seq[i + p]:
                i += 1
            if i - start >= n:
                for k in range(start, i + p):
                    covered[k] = True
    return sum(covered) / size


def goals(j: Journal, before: float | None = None) -> list[str]:
    if before is None:
        rows = j.db.execute("SELECT text FROM memory WHERE kind = ? ORDER BY id", ("цель",))
    else:
        rows = j.db.execute("SELECT text FROM memory WHERE kind = ? AND t < ? ORDER BY id", ("цель", before))
    return [r[0] for r in rows]


def report(j: Journal, t0: float) -> dict:
    return {
        "diversity": diversity(j, t0),
        "persistence": persistence(goals(j, t0), goals(j)),
        "trend": trend(j, t0),
        "loop": loop_score(j, t0),
    }
