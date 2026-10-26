from __future__ import annotations

import asyncio
from dataclasses import dataclass

from .config import Territory
from .stem import Stem


@dataclass
class Result:
    ok: bool
    text: str
    reflex: str = ""


class Executor:
    def __init__(self, territory: Territory, stem: Stem):
        self.t = territory
        self.stem = stem

    async def run(self, command: str) -> Result:
        cmd = " ".join(command.split())
        if not self.t.exec_enabled:
            return Result(False, "исполнение выключено территорией", "выключено")
        if cmd not in self.t.exec_allowlist:
            return Result(False, f"команда вне списка разрешённых: {cmd}", "вне_списка")
        denied = self.stem.action_denied()
        if denied:
            return Result(False, denied, "лимит")
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd.split(), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        except OSError as e:
            return Result(False, f"не запустилось: {e}", "запуск")
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), self.t.exec_timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return Result(False, "таймаут исполнения", "таймаут")
        return Result(proc.returncode == 0, out.decode("utf-8", "replace").strip()[:2000])
