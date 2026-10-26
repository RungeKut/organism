from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from .context import Ctx
from .events import Event
from .llm import LLMError
from .protocol import Envelope
from .society import certainty
from .streams import Stream

SURE = 0.5
SKIP = frozenset({"координатор", "речь", "моторика"})
BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
CHECK = ("Оцени, согласуется ли сводка ответов специалистов с вопросом. "
         "Ответь одним числом — уверенностью от 0 до 1.")


@dataclass(frozen=True)
class Step:
    profession: str
    text: str


class Unsure(Exception):
    pass


def plan(raw: str, known: set[str] | frozenset[str] | None = None) -> list[Step]:
    steps = []
    for line in str(raw or "").splitlines():
        head, sep, tail = BULLET.sub("", line).partition(":")
        profession, text = head.strip().strip("«»\"'*").strip().lower(), tail.strip()
        if not sep or not profession or not text or profession in SKIP:
            continue
        if known is not None and profession not in known:
            continue
        steps.append(Step(profession, text))
    return steps


def score(raw: str) -> float:
    m = NUMBER.search(raw or "")
    return max(0.0, min(1.0, float(m.group().replace(",", ".")))) if m else 0.0


class Inbox(asyncio.Queue):
    def __init__(self, waiting: dict[str, asyncio.Future]):
        super().__init__()
        self.waiting = waiting

    def put_nowait(self, ev: Event) -> None:
        fut = self.waiting.get(ev.payload.get("episode", "")) if ev.topic == "результат" else None
        if fut is None:
            super().put_nowait(ev)
        elif not fut.done():
            fut.set_result(ev)


class Coordinator(Stream):
    topics = ("просьба",)
    professions = ("координатор",)
    clonable = False
    protected = False

    def __init__(self, name: str, ctx: Ctx):
        super().__init__(name, ctx)
        self.waiting: dict[str, asyncio.Future] = {}
        self.queue = Inbox(self.waiting)

    def known(self) -> set[str]:
        return {p for p, owners in self.ctx.graph.registry.items() if owners and p not in SKIP}

    async def handle(self, ev: Event) -> None:
        if ev.topic != "просьба":
            return
        c, p = self.ctx, ev.payload
        episode, hops = p.get("episode", ""), p.get("hops", 0)
        reply_to = p.get("reply_to") or ev.source
        text = Envelope.from_dict(p["env"]).text if "env" in p else p.get("text", "")
        try:
            if hops + 1 > c.genome.max_chain:
                c.journal.event(Event("цепочка_оборвана", self.name, {"episode": episode, "hops": hops}))
                raise Unsure("цепочка слишком длинная")
            env = await self.solve(text, episode, hops)
        except Unsure as e:
            env = Envelope(reply_to, "отказ", f"не уверен: {e}", 0.0)
        except (LLMError, PermissionError) as e:
            c.stats["сбой_llm"] += 1
            env = Envelope(reply_to, "отказ", f"не уверен: не могу думать: {e}", 0.0)
        self.reply(reply_to, env, episode, hops)

    async def solve(self, text: str, episode: str, hops: int) -> Envelope:
        c = self.ctx
        known = self.known()
        limit = c.genome.max_chain
        system = ("Ты — координатор. Разложи задачу на короткие шаги, по одному в строке: "
                  f"«профессия: вопрос». Профессии: {', '.join(sorted(known)) or 'нет'}. "
                  f"Не больше {limit} шагов. Если разложить нельзя — ответь пустой строкой.")
        steps = plan(await self.think(system, text), known)
        if not steps:
            raise Unsure("не смог разложить задачу на вопросы специалистам")
        if len(steps) > limit:
            c.journal.event(Event("цепочка_оборвана", self.name, {"episode": episode, "шагов": len(steps)}))
            raise Unsure(f"план длиннее цепочки ({len(steps)} > {limit})")
        answers = []
        for k, step in enumerate(steps, 1):
            res = await self.ask(step, f"{episode}.{k}", hops)
            if not res.get("ok") or res.get("certainty") != "знаю":
                raise Unsure(f"{step.profession}: {res.get('text', '')}")
            answers.append(f"{step.profession}: {res.get('text', '')}")
        summary = "\n".join(answers)
        confidence = score(await self.think(CHECK, f"Вопрос: {text}\nСводка:\n{summary}"))
        if confidence < SURE:
            raise Unsure(f"сводка не согласуется с вопросом ({confidence:.2f})")
        return Envelope("спросивший", "ответ", summary, confidence)

    async def think(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return await self.ctx.llm.chat(messages, max_tokens=256, deadline_s=self.deadline_s)

    async def ask(self, step: Step, sub: str, hops: int) -> dict:
        c = self.ctx
        target = c.graph.route(self.name, step.profession)
        if target is None:
            raise Unsure(f"нет специалиста «{step.profession}»")
        env = Envelope(step.profession, "вопрос", step.text, 1.0, self.name)
        payload = {"env": env.to_dict(), "episode": sub, "hops": hops, "reply_to": self.name}
        fut = asyncio.get_running_loop().create_future()
        self.waiting[sub] = fut
        try:
            if not c.bus.send(Event("просьба", self.name, payload), target):
                raise Unsure(f"не смог спросить «{step.profession}»")
            ev = await asyncio.wait_for(fut, self.deadline_s)
        except asyncio.TimeoutError:
            raise Unsure(f"«{step.profession}» не ответил за {self.deadline_s:g} с") from None
        finally:
            self.waiting.pop(sub, None)
        return ev.payload

    def reply(self, to: str, env: Envelope, episode: str, hops: int) -> bool:
        env.to, env.sender = to, self.name
        ok = env.kind != "отказ"
        payload = {"ok": ok, "text": env.text, "env": env.to_dict(),
                   "certainty": certainty(env) if ok else "не уверен",
                   "profession": "координатор", "episode": episode, "hops": hops + 1}
        if self.ctx.bus.send(Event("результат", self.name, payload), to):
            return True
        self.ctx.stats["ответ_потерян"] += 1
        return False
