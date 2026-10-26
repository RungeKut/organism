from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

from .events import Event
from .llm import LLMError
from .protocol import Envelope
from .streams import Stream

KNOW = 0.7
RULES = ("Отвечай строго конвертом:\nкому: спросивший\nтип: ответ, факт или отказ\n"
         "уверенность: число от 0 до 1\nтекст: сообщение\n"
         "Если вопрос не твоей профессии — тип отказ. Если не уверен — поставь низкую уверенность.")


@dataclass(frozen=True)
class Profession:
    name: str
    prompt: str
    lora_id: int | str | None
    tools: tuple[str, ...] = ()


def load_professions(path: Path | str) -> list[Profession]:
    path = Path(path)
    if not path.exists():
        return []
    names = {f.name for f in fields(Profession)}
    result = []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return []
    for row in rows:
        data = {k: v for k, v in row.items() if k in names}
        data.setdefault("lora_id", None)
        data["tools"] = tuple(data.get("tools", ()))
        result.append(Profession(**data))
    return result


def certainty(env: Envelope) -> str:
    if env.kind == "отказ":
        return "не моя"
    return "знаю" if env.confidence >= KNOW else "не уверен"


class Specialist(Stream):
    topics = ("просьба",)
    clonable = False
    protected = False
    profession: Profession

    async def handle(self, ev: Event) -> None:
        c, p = self.ctx, ev.payload
        episode, hops = p.get("episode", ""), p.get("hops", 0)
        reply_to = p.get("reply_to") or ev.source
        if hops + 1 > c.genome.max_chain:
            c.journal.event(Event("цепочка_оборвана", self.name, {"episode": episode, "hops": hops}))
            self.reply(reply_to, Envelope(reply_to, "отказ", "Цепочка слишком длинная", 1.0),
                       episode, hops, "не уверен")
            return
        text = Envelope.from_dict(p["env"]).text if "env" in p else p.get("text", "")
        try:
            env = await self.answer(text)
        except (LLMError, PermissionError) as e:
            c.stats["сбой_llm"] += 1
            self.reply(reply_to, Envelope(reply_to, "отказ", f"Не могу думать: {e}", 0.0),
                       episode, hops, "не уверен")
            return
        self.reply(reply_to, env, episode, hops, certainty(env))

    async def answer(self, text: str) -> Envelope:
        prof = self.profession
        messages = [{"role": "system", "content": f"{prof.prompt}\n{RULES}"}, {"role": "user", "content": text}]
        lora = None if prof.lora_id is None else [{"id": prof.lora_id, "scale": 1.0}]
        raw = await self.ctx.llm.chat(messages, max_tokens=256, lora=lora, deadline_s=self.deadline_s)
        return Envelope.parse(raw) or Envelope("спросивший", "ответ", raw.strip(), 0.3)

    def reply(self, to: str, env: Envelope, episode: str, hops: int, verdict: str) -> bool:
        env.to, env.sender = to, self.name
        ok = env.kind != "отказ"
        payload = {"ok": ok, "text": env.text, "env": env.to_dict(), "certainty": verdict,
                   "profession": self.profession.name, "episode": episode, "hops": hops + 1}
        if self.ctx.bus.send(Event("результат", self.name, payload), to):
            return True
        self.ctx.stats["ответ_потерян"] += 1
        return False


def make_specialist(prof: Profession) -> type[Stream]:
    return type(f"Specialist_{prof.name}", (Specialist,), {"profession": prof, "professions": (prof.name,)})
