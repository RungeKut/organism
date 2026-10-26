from __future__ import annotations

import math
import re
import time
from pathlib import Path

from .config import Genome
from .llm import LLM, LLMError

NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


class Mother:
    def __init__(self, llm: LLM, genome: Genome, birth_file: Path, now=time.time):
        self.llm = llm
        self.g = genome
        self.now = now
        if not birth_file.exists():
            birth_file.write_text(str(now()), encoding="utf-8")
        self.birth = float(birth_file.read_text(encoding="utf-8"))

    def age_days(self) -> float:
        return max(0.0, (self.now() - self.birth) / 86400.0)

    def share(self) -> float:
        return self.g.mother_share0 * math.exp(-self.age_days() / self.g.mother_tau_days)

    async def rate(self, transcript: str) -> float | None:
        system = "Оцени ответы организма по полезности и точности. Ответь одним числом от 0 до 1."
        try:
            raw = await self.llm.chat([{"role": "system", "content": system},
                                       {"role": "user", "content": transcript}], max_tokens=8)
        except (LLMError, PermissionError):
            return None
        m = NUMBER.search(raw)
        return min(1.0, max(0.0, float(m.group().replace(",", ".")))) if m else None

    async def summarize(self, transcript: str) -> str | None:
        system = "Сожми разговор в одну короткую заметку для памяти."
        try:
            raw = await self.llm.chat([{"role": "system", "content": system},
                                       {"role": "user", "content": transcript}], max_tokens=96)
        except (LLMError, PermissionError):
            return None
        return raw.strip() or None
