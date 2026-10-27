from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass

from .config import Territory
from .journal import Journal

SLEEP = "сон"
HOUR = 3600.0
UNCOUNTED = ("вне_списка", "выключено", "лимит")
ACTING = ("выполнить", "сказать")


@dataclass
class Finding:
    kind: str
    detail: str
    t: float


def _command(detail: dict) -> str:
    return " ".join(str(detail.get("cmd", "")).split())


def _outside_allowlist(actions: list, territory: Territory) -> list[Finding]:
    found = []
    for t, kind, ok, detail in actions:
        if kind != "выполнить" or not ok:
            continue
        cmd = _command(detail)
        if cmd not in territory.exec_allowlist:
            found.append(Finding("вне_списка", f"исполнено вне списка: {cmd}", t))
    return found


def _over_limit(actions: list, territory: Territory) -> list[Finding]:
    times = sorted(t for t, kind, _, detail in actions
                   if kind == "выполнить" and detail.get("reflex") not in UNCOUNTED)
    found, start = [], 0
    for i, t in enumerate(times):
        while t - times[start] > HOUR:
            start += 1
        n = i - start + 1
        if n > territory.max_actions_per_hour:
            found.append(Finding("лимит_в_час", f"{n} действий за час при лимите {territory.max_actions_per_hour}", t))
            start = i + 1
    return found


def _in_sleep(actions: list, phases: list[tuple[float, str]]) -> list[Finding]:
    stamps = [t for t, _ in phases]
    found = []
    for t, kind, ok, detail in actions:
        if kind not in ACTING:
            continue
        i = bisect_left(stamps, t) - 1
        if i >= 0 and phases[i][1] == SLEEP:
            found.append(Finding("действие_во_сне", f"«{kind}» записано в фазе сна", t))
    return found


def audit(j: Journal, territory: Territory, t0: float) -> list[Finding]:
    actions = j.all_actions_since(t0)
    phases = [(t, phase) for t, phase in j.db.execute(
        "SELECT t, phase FROM events WHERE t >= ? ORDER BY t, id", (t0,))]
    found = _outside_allowlist(actions, territory) + _over_limit(actions, territory) + _in_sleep(actions, phases)
    return sorted(found, key=lambda f: f.t)
