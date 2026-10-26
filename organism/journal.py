from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from .events import Event

SCHEMA = """
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, t REAL, phase TEXT, topic TEXT, source TEXT, payload TEXT);
CREATE TABLE IF NOT EXISTS actions(id INTEGER PRIMARY KEY, t REAL, kind TEXT, ok INTEGER, detail TEXT);
CREATE TABLE IF NOT EXISTS rewards(id INTEGER PRIMARY KEY, t REAL, value REAL, source TEXT);
CREATE TABLE IF NOT EXISTS memory(id INTEGER PRIMARY KEY, t REAL, kind TEXT, text TEXT);
"""


def _dump(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False)


class Journal:
    def __init__(self, path: Path):
        self.db = sqlite3.connect(str(path), isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        self.phase = "бодрствование"

    def event(self, ev: Event) -> None:
        self.db.execute("INSERT INTO events(t, phase, topic, source, payload) VALUES (?, ?, ?, ?, ?)",
                        (ev.t, self.phase, ev.topic, ev.source, _dump(ev.payload)))

    def action(self, kind: str, ok: bool, detail: dict) -> None:
        self.db.execute("INSERT INTO actions(t, kind, ok, detail) VALUES (?, ?, ?, ?)",
                        (time.time(), kind, int(ok), _dump(detail)))

    def reward(self, value: float, source: str) -> None:
        self.db.execute("INSERT INTO rewards(t, value, source) VALUES (?, ?, ?)", (time.time(), value, source))

    def remember(self, text: str, kind: str = "память") -> None:
        self.db.execute("INSERT INTO memory(t, kind, text) VALUES (?, ?, ?)", (time.time(), kind, text))

    def memories(self, kind: str = "память", limit: int = 5) -> list[str]:
        rows = self.db.execute("SELECT text FROM memory WHERE kind = ? ORDER BY id DESC LIMIT ?", (kind, limit))
        return [r[0] for r in rows][::-1]

    def events_since(self, t0: float, topic: str) -> list[tuple[float, str, dict]]:
        rows = self.db.execute("SELECT t, source, payload FROM events WHERE t >= ? AND topic = ? ORDER BY id",
                               (t0, topic))
        return [(t, s, json.loads(p)) for t, s, p in rows]

    def actions_since(self, t0: float, kind: str) -> list[tuple[float, bool, dict]]:
        rows = self.db.execute("SELECT t, ok, detail FROM actions WHERE t >= ? AND kind = ? ORDER BY id", (t0, kind))
        return [(t, bool(ok), json.loads(d)) for t, ok, d in rows]

    def rewards_since(self, t0: float, source: str) -> list[tuple[float, float]]:
        rows = self.db.execute("SELECT t, value FROM rewards WHERE t >= ? AND source = ? ORDER BY id", (t0, source))
        return list(rows)

    def all_events_since(self, t0: float) -> list[tuple[float, str, str, dict]]:
        rows = self.db.execute("SELECT t, topic, source, payload FROM events WHERE t >= ? ORDER BY id", (t0,))
        return [(t, topic, source, json.loads(p)) for t, topic, source, p in rows]

    def all_actions_since(self, t0: float) -> list[tuple[float, str, bool, dict]]:
        rows = self.db.execute("SELECT t, kind, ok, detail FROM actions WHERE t >= ? ORDER BY id", (t0,))
        return [(t, kind, bool(ok), json.loads(d)) for t, kind, ok, d in rows]

    def close(self) -> None:
        self.db.close()
