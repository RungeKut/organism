from __future__ import annotations

import asyncio
from typing import Any

from .context import Ctx
from .events import Event
from .llm import LLMError
from .protocol import Envelope
from .workspace import Proposal


class Stream:
    topics: tuple[str, ...] = ()
    professions: tuple[str, ...] = ()
    cores: float = 0.25
    period: int = 0
    protected: bool = True
    clonable: bool = False

    def __init__(self, name: str, ctx: Ctx):
        self.name = name
        self.ctx = ctx
        self.base = name.split("-")[0]
        self.deadline_s = ctx.genome.deadlines_s.get(self.base, 30.0)
        self.queue: asyncio.Queue = asyncio.Queue()
        self.max_queue = 0
        self.earned = 0.0
        self.spent = 0.0
        self.debt = 0
        self.started = ctx.clock()
        self.task: asyncio.Task | None = None

    def start(self) -> None:
        self.ctx.bus.subscribe(self.name, self.topics, self.queue)
        self.ctx.graph.announce(self.name, self.professions)
        self.task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self.ctx.bus.unsubscribe(self.name)
        self.ctx.graph.forget(self.name)
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)

    async def _loop(self) -> None:
        c = self.ctx
        while True:
            ev = await self.queue.get()
            self.max_queue = max(self.max_queue, self.queue.qsize() + 1)
            t0 = self.started = c.clock()
            try:
                with c.stem.task():
                    await self.handle(ev)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                c.stats["сбой_потока"] += 1
                c.journal.action("сбой", False, {"поток": self.name, "ошибка": f"{type(e).__name__}: {e}"})
            finally:
                self.spent += (c.clock() - t0) * self.cores * c.genome.cost_per_core_s

    async def handle(self, ev: Event) -> None:
        raise NotImplementedError

    async def pulse(self) -> None:
        return None

    def propose(self, effector: str, payload: dict[str, Any], salience: float, value: float = 1.0) -> None:
        self.ctx.workspace.propose(Proposal(self.name, effector, payload, salience, value,
                                            self.ctx.clock(), self.started + self.deadline_s))

    def forward(self, env: Envelope, episode: str, hops: int, profession: str, reply_to: str) -> bool:
        c = self.ctx
        if hops + 1 > c.genome.max_chain:
            c.journal.event(Event("цепочка_оборвана", self.name, {"episode": episode, "hops": hops}))
            return False
        target = c.graph.route(self.name, profession)
        if target is None:
            return False
        env.sender = self.name
        payload = {"env": env.to_dict(), "episode": episode, "hops": hops + 1, "reply_to": reply_to}
        return c.bus.send(Event("просьба", self.name, payload), target)


class Speech(Stream):
    topics = ("вход", "результат")
    professions = ("речь",)
    clonable = True

    def __init__(self, name: str, ctx: Ctx):
        super().__init__(name, ctx)
        self.dialogs: dict[str, list[dict]] = {}

    def system_prompt(self) -> str:
        c = self.ctx
        memory = "\n".join(f"- {m}" for m in c.journal.memories()) or "- пока пусто"
        allowed = "; ".join(c.territory.exec_allowlist) or "нет"
        return ("Ты — организм, живущий на этой машине. Говори по-русски. Отвечай строго конвертом:\n"
                "кому: пользователь или моторика\nтип: ответ, просьба или отказ\n"
                "уверенность: число от 0 до 1\nтекст: сообщение\n"
                f"Моторике адресуй только точную команду из списка: {allowed}. "
                "Если не знаешь — скажи «не уверен» и поставь низкую уверенность.\n"
                f"Память:\n{memory}")

    async def think(self, episode: str, text: str) -> Envelope:
        c = self.ctx
        dialog = self.dialogs.setdefault(episode, [])
        dialog.append({"role": "user", "content": text})
        if len(self.dialogs) > 50:
            self.dialogs.pop(next(iter(self.dialogs)))
        messages = [{"role": "system", "content": self.system_prompt()}] + dialog
        try:
            raw = await c.llm.chat(messages, max_tokens=256, deadline_s=self.deadline_s)
        except (LLMError, PermissionError) as e:
            c.stats["сбой_llm"] += 1
            return Envelope("пользователь", "отказ", f"Не могу думать: {e}", 0.0)
        dialog.append({"role": "assistant", "content": raw})
        return Envelope.parse(raw) or Envelope("пользователь", "ответ", raw.strip(), 0.3)

    async def handle(self, ev: Event) -> None:
        p = ev.payload
        episode, hops = p.get("episode", ""), p.get("hops", 0)
        if ev.topic == "вход":
            text = p["text"]
        elif ev.topic == "результат":
            text = f"Моторика сообщает: {'успех' if p.get('ok') else 'отказ'}\n{p.get('text', '')}"
        else:
            return
        env = await self.think(episode, text)
        if env.to == "моторика":
            if self.forward(env, episode, hops, "моторика", self.name):
                return
            env = Envelope("пользователь", "отказ", "Не смог передать просьбу моторике", 1.0)
        self.dialogs.pop(episode, None)
        self.propose("голос", {"text": env.text, "episode": episode, "kind": env.kind,
                               "confidence": env.confidence}, env.confidence)


class Motor(Stream):
    topics = ("просьба",)
    professions = ("моторика",)

    async def handle(self, ev: Event) -> None:
        p = ev.payload
        env = Envelope.from_dict(p["env"])
        self.propose("рука", {"cmd": env.text, "episode": p.get("episode", ""), "hops": p.get("hops", 1),
                              "reply_to": p.get("reply_to")}, env.confidence)


class Body(Stream):
    period = 0
    cores = 0.1

    def __init__(self, name: str, ctx: Ctx):
        super().__init__(name, ctx)
        self.period = ctx.genome.vitals_every
        self.seen_misses = 0

    async def handle(self, ev: Event) -> None:
        return None

    async def pulse(self) -> None:
        c = self.ctx
        misses = c.stats["просрочено"]
        pain = c.lag_ms > c.genome.pain_lag_ms or misses > self.seen_misses
        self.seen_misses = misses
        depth = sum(s.queue.qsize() for s in c.streams.values())
        if pain:
            c.stem.request_tact(c.stem.tact_s * 1.5)
            c.reward(-0.3, "тело")
        elif c.stem.tact_s > c.genome.tact_s:
            c.stem.request_tact(max(c.genome.tact_s, c.stem.tact_s * 0.9))
        c.bus.publish(Event("ощущение", self.name, {"боль": pain, "lag_ms": round(c.lag_ms, 1),
                                                    "очередь": depth, "такт_с": round(c.stem.tact_s, 3)}))


STREAM_TYPES: dict[str, type[Stream]] = {"речь": Speech, "моторика": Motor, "тело": Body}
