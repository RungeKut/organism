from __future__ import annotations

import json
import secrets
import time
from pathlib import Path

PROPOSALS_DIR = "proposals"
DECISIONS_FILE = "decisions.jsonl"


def _key(kind: str, target: str, after: str) -> tuple[str, str, str]:
    return (kind, target, after)


def decisions(state: Path) -> list[dict]:
    path = Path(state) / DECISIONS_FILE
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def pending(state: Path) -> list[dict]:
    d = Path(state) / PROPOSALS_DIR
    if not d.exists():
        return []
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in d.glob("*.json")]
    return sorted(rows, key=lambda r: r.get("t", 0.0))


def rejected(state: Path, kind: str, target: str, after: str) -> bool:
    return any(not row.get("accept") and (row.get("kind"), row.get("target"), row.get("after")) == _key(kind, target, after)
               for row in decisions(state))


def propose(state: Path, kind: str, target: str, before: str, after: str,
            parent_score: float, child_score: float, reason: str) -> Path | None:
    key = _key(kind, target, after)
    if any((row.get("kind"), row.get("target"), row.get("after")) == key for row in decisions(state)):
        return None
    if any((row.get("kind"), row.get("target"), row.get("after")) == key for row in pending(state)):
        return None
    d = Path(state) / PROPOSALS_DIR
    d.mkdir(parents=True, exist_ok=True)
    proposal = {
        "id": secrets.token_hex(4),
        "token": secrets.token_hex(16),
        "t": time.time(),
        "kind": kind,
        "target": target,
        "before": before,
        "after": after,
        "parent_score": float(parent_score),
        "child_score": float(child_score),
        "reason": reason,
    }
    path = d / f"{proposal['id']}.json"
    while path.exists():
        proposal["id"] = secrets.token_hex(4)
        path = d / f"{proposal['id']}.json"
    path.write_text(json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def decide(state: Path, proposal_id: str, token: str, accept: bool, reason: str = "") -> bool:
    if not isinstance(token, str) or not isinstance(proposal_id, str) or not proposal_id.isalnum():
        return False
    path = Path(state) / PROPOSALS_DIR / f"{proposal_id}.json"
    if not path.exists():
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    if not secrets.compare_digest(str(data.get("token", "")).encode(), token.encode()):
        return False
    decision = {
        "t": time.time(),
        "id": data["id"],
        "kind": data["kind"],
        "target": data["target"],
        "after": data["after"],
        "accept": bool(accept),
        "reason": reason,
        "parent_score": data["parent_score"],
        "child_score": data["child_score"],
    }
    with open(Path(state) / DECISIONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(decision, ensure_ascii=False) + "\n")
    path.unlink()
    return True
