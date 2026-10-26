from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _known(cls, data: dict) -> dict:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in names}


@dataclass(frozen=True)
class Territory:
    max_streams: int = 8
    max_llm_slots: int = 4
    headroom: float = 0.25
    max_actions_per_hour: int = 120
    exec_enabled: bool = True
    exec_allowlist: tuple[str, ...] = ("python --version", "git --version", "git status", "uv --version")
    exec_timeout_s: float = 10.0
    allow_network: bool = False
    llm_url: str = "http://127.0.0.1:8080"
    mother_url: str | None = None
    tact_bounds_s: tuple[float, float] = (0.05, 60.0)
    max_children: int = 1
    generation_every_sleeps: int = 0

    @classmethod
    def load(cls, path: Path) -> Territory:
        data = _known(cls, read_json(path))
        for key in ("exec_allowlist", "tact_bounds_s"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)


@dataclass
class Genome:
    tact_s: float = 0.25
    wake_s: float = 600.0
    sleep_s: float = 60.0
    drowsy_max_s: float = 30.0
    sleep_pressure_events: int = 2000
    eta: float = 0.1
    decay: float = 0.01
    trace_tau_s: float = 30.0
    max_dw: float = 0.3
    w_min: float = 0.02
    explore: float = 0.1
    streams: list[str] = field(default_factory=lambda: ["речь", "моторика", "тело"])
    deadlines_s: dict[str, float] = field(default_factory=lambda: {"речь": 60.0, "моторика": 5.0, "тело": 1.0})
    max_chain: int = 2
    cost_per_core_s: float = 0.001
    pain_lag_ms: float = 200.0
    vitals_every: int = 4
    mother_share0: float = 0.8
    mother_tau_days: float = 30.0

    @classmethod
    def load(cls, path: Path) -> Genome:
        return cls(**_known(cls, read_json(path))) if path.exists() else cls()

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
