from __future__ import annotations

import json
from pathlib import Path

FILE = "holdout.jsonl"
LIMIT = 8


def load(state: Path) -> list[str]:
    path = Path(state) / FILE
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line)["text"])
    return rows


def add(state: Path, text: str) -> bool:
    text = text.strip()
    if not text or text in load(state):
        return False
    with open(Path(state) / FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    return True


def sample(state: Path) -> list[str]:
    return load(state)[-LIMIT:]
