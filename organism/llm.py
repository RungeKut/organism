from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from urllib.parse import urlparse

from .protocol import Envelope

LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1")


class LLMError(Exception):
    pass


class AIMD:
    def __init__(self, ceiling: int, start: int = 1):
        self.ceiling = max(1, ceiling)
        self.limit = max(1, min(start, self.ceiling))
        self.active = 0
        self._streak = 0
        self._cond = asyncio.Condition()

    async def acquire(self) -> None:
        async with self._cond:
            await self._cond.wait_for(lambda: self.active < self.limit)
            self.active += 1

    async def release(self, ok: bool) -> None:
        async with self._cond:
            self.active -= 1
            if not ok:
                self.limit, self._streak = max(1, self.limit // 2), 0
            else:
                self._streak += 1
                if self._streak >= self.limit:
                    self.limit, self._streak = min(self.ceiling, self.limit + 1), 0
            self._cond.notify_all()


class LLM:
    last_tps: float = 0.0

    async def chat(self, messages: list[dict], max_tokens: int = 256, lora: list[dict] | None = None,
                   deadline_s: float | None = None) -> str:
        raise NotImplementedError


class LlamaCpp(LLM):
    def __init__(self, url: str, slots: AIMD, allow_network: bool = False):
        if urlparse(url).hostname not in LOCAL_HOSTS and not allow_network:
            raise PermissionError(f"адрес {url} вне территории: allow_network выключен")
        self.url = url.rstrip("/")
        self.slots = slots

    async def chat(self, messages, max_tokens=256, lora=None, deadline_s=None) -> str:
        payload = {"messages": messages, "max_tokens": max_tokens, "temperature": 0.7}
        if lora:
            payload["lora"] = lora
        timeout = deadline_s or 120.0
        await self.slots.acquire()
        ok = False
        try:
            text = await asyncio.wait_for(asyncio.to_thread(self._post, payload, timeout), timeout)
            ok = True
            return text
        except (asyncio.TimeoutError, OSError, ValueError, KeyError) as e:
            raise LLMError(f"{type(e).__name__}: {e}") from e
        finally:
            await self.slots.release(ok)

    def _post(self, payload: dict, timeout: float) -> str:
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as e:
            e.close()
            raise
        self.last_tps = float(data.get("timings", {}).get("predicted_per_second", self.last_tps))
        return data["choices"][0]["message"]["content"]


class FakeLLM(LLM):
    def __init__(self, delay_s: float = 0.01):
        self.delay_s = delay_s
        self.calls = 0

    async def chat(self, messages, max_tokens=256, lora=None, deadline_s=None) -> str:
        self.calls += 1
        await asyncio.sleep(self.delay_s)
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        last = messages[-1]["content"]
        if "Оцени" in system:
            return "0.8"
        if "Сожми" in system:
            return "Заметка: днём был разговор с пользователем."
        if last.startswith("Моторика сообщает"):
            return Envelope("пользователь", "ответ", "Команда выполнена. " + last.splitlines()[-1], 0.9).render()
        if "команд" in last.lower():
            return Envelope("моторика", "просьба", "git --version", 0.9).render()
        return Envelope("пользователь", "ответ", f"Слышу: {last}", 0.9).render()
