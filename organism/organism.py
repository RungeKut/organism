from __future__ import annotations

import asyncio
import itertools
import random
import time
from pathlib import Path
from typing import Callable

from .body import BodyMap, inventory, loop_lag_ms
from .config import ROOT, Genome, Territory
from .context import Ctx
from .events import Bus, Event
from .executor import Executor
from .graph import Graph
from .journal import Journal
from .llm import AIMD, LLM, FakeLLM, LlamaCpp
from .mother import Mother
from .sleep import run as sleep_run
from .stem import Grant, Phase, Stem
from .streams import STREAM_TYPES, Stream
from .workspace import Proposal, Workspace


class Organism:
    def __init__(self, state_dir: Path, llm: LLM | None = None, territory: Territory | None = None,
                 genome: Genome | None = None, clock: Callable[[], float] = time.monotonic,
                 seed: int | None = None):
        self.state = Path(state_dir)
        self.state.mkdir(parents=True, exist_ok=True)
        self.territory = territory or Territory.load(ROOT / "config" / "territory.json")
        saved = self.state / "genome.json"
        self.genome = genome or Genome.load(saved if saved.exists() else ROOT / "config" / "genome.json")
        self.llm = llm or FakeLLM()
        self.clock = clock
        self.rng = random.Random(seed)
        self.stem = Stem(self.territory, self.genome, clock)
        self.graph = Graph(self.genome, self.rng)
        self.journal = Journal(self.state / "organism.db")
        self.bus = Bus(self.graph, [self.journal.event, lambda ev: self.stem.note_event()])
        self.workspace = Workspace()
        self.executor = Executor(self.territory, self.stem)
        self.body = BodyMap.load(self.state / "bodymap.json") or inventory()
        self.ctx = Ctx(self.territory, self.genome, self.stem, self.graph, self.bus, self.journal, self.llm,
                       self.workspace, self.executor, self.body, clock=clock)
        self.streams: dict[str, Stream] = self.ctx.streams
        mother_llm = self.llm
        if self.territory.mother_url:
            mother_llm = LlamaCpp(self.territory.mother_url, AIMD(1), self.territory.allow_network)
        self.mother = Mother(mother_llm, self.genome, self.state / "birth.txt")
        self.wake_wall = time.time()
        self._episodes = itertools.count(1)
        self._waiting: dict[str, asyncio.Future] = {}
        self._acts: set[asyncio.Task] = set()
        self._ticker: asyncio.Task | None = None
        self._sleep_task: asyncio.Task | None = None
        self._last = clock()
        self._n = 0
        self.sleeps = 0
        self.last_sleep_report: dict = {}

    async def start(self) -> None:
        self.graph.load(self.state / "graph.json")
        self.body.loop_lag_ms = await loop_lag_ms()
        self.body.save(self.state / "bodymap.json")
        for name in self.genome.streams:
            grant = self.spawn(name)
            if not grant.ok:
                self.journal.event(Event("отказ_рождения", "ствол", {"поток": name, "причина": grant.reason}))
        self._last = self.clock()
        self.wake_wall = time.time()
        self._ticker = asyncio.create_task(self._tick_loop())

    async def stop(self) -> None:
        tasks = [t for t in (self._ticker, self._sleep_task, *self._acts) if t]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for stream in list(self.streams.values()):
            await stream.stop()
        self.save_state()
        self.journal.close()

    def spawn(self, name: str) -> Grant:
        cls = STREAM_TYPES.get(name.split("-")[0])
        if cls is None:
            return Grant(False, "неизвестный тип потока")
        grant = self.stem.request_stream(name, cls.cores, self.body.capacity_cores)
        if not grant.ok:
            return grant
        stream = cls(name, self.ctx)
        stream.protected = name == stream.base
        self.streams[name] = stream
        stream.start()
        self.journal.event(Event("рождение", "ствол", {"поток": name}))
        return grant

    def kill(self, name: str) -> None:
        stream = self.streams.pop(name, None)
        if stream is None:
            return
        self.stem.release_stream(name)
        self.graph.edges = {k: e for k, e in self.graph.edges.items() if name not in k}
        asyncio.create_task(stream.stop())
        self.journal.event(Event("смерть", "ствол", {"поток": name}))

    def clone_name(self, base: str) -> str:
        k = 2
        while f"{base}-{k}" in self.streams:
            k += 1
        return f"{base}-{k}"

    def save_state(self) -> None:
        self.graph.save(self.state / "graph.json")
        self.genome.save(self.state / "genome.json")

    def busy(self) -> int:
        queued = sum(s.queue.qsize() for s in self.streams.values())
        return self.stem.in_flight + queued + len(self.workspace.pending) + len(self._acts)

    async def say(self, text: str, timeout: float | None = None) -> str | None:
        if not self.stem.accepts_input():
            self.journal.action("отклонено", False, {"причина": self.stem.phase.value, "text": text})
            return None
        target = self.graph.route("пользователь", "речь")
        if target is None:
            return None
        episode = f"э{next(self._episodes)}"
        waiter = asyncio.get_running_loop().create_future()
        self._waiting[episode] = waiter
        self.bus.send(Event("вход", "пользователь", {"text": text, "episode": episode, "hops": 0}), target)
        limit = timeout or self.genome.deadlines_s.get("речь", 60.0) + 5.0
        try:
            return await asyncio.wait_for(waiter, limit)
        except asyncio.TimeoutError:
            return None
        finally:
            self._waiting.pop(episode, None)

    def feedback(self, value: float) -> None:
        self.ctx.reward(max(-1.0, min(1.0, value)), "человек")

    async def _tick_loop(self) -> None:
        while True:
            await asyncio.sleep(self.stem.tact_s)
            await self.step()

    async def step(self) -> None:
        now = self.clock()
        dt, self._last = now - self._last, now
        sample = max(0.0, (dt - self.stem.tact_s) * 1000.0)
        self.ctx.lag_ms = 0.7 * self.ctx.lag_ms + 0.3 * sample
        self.graph.decay_traces(dt)
        winners, expired = self.workspace.select(now)
        for p in expired:
            self._expire(p)
        for p in winners:
            self._act(p)
        self._n += 1
        for stream in list(self.streams.values()):
            if stream.period and self._n % stream.period == 0:
                await stream.pulse()
        phase = self.stem.tick(self.busy())
        if phase is Phase.DROWSY:
            self.journal.phase = phase.value
        elif phase is Phase.SLEEP:
            self.journal.phase = phase.value
            self._sleep_task = asyncio.create_task(self._sleep())
        elif phase is Phase.WAKE:
            self.journal.phase = phase.value
            self.wake_wall = time.time()

    async def _sleep(self) -> None:
        try:
            self.last_sleep_report = await sleep_run(self)
            self.sleeps += 1
            self.journal.remember(str(self.last_sleep_report), "сон")
        finally:
            self.stem.sleep_done()

    def _act(self, p: Proposal) -> None:
        self.workspace.post(p.stream, p.effector, str(p.payload.get("text") or p.payload.get("cmd", "")))
        self.bus.publish(Event("сцена", "селектор", {"поток": p.stream, "орган": p.effector}))
        if p.effector == "голос":
            self._speak(p)
        else:
            task = asyncio.create_task(self._do_hand(p))
            self._acts.add(task)
            task.add_done_callback(self._acts.discard)

    def _speak(self, p: Proposal) -> None:
        episode = p.payload.get("episode", "")
        text = p.payload.get("text", "")
        self.journal.action("сказать", True, {"text": text, "episode": episode, "stream": p.stream})
        self.ctx.reward(0.3 * p.payload.get("confidence", 1.0), "цель", p.stream)
        waiter = self._waiting.get(episode)
        if waiter and not waiter.done():
            waiter.set_result(text)

    async def _do_hand(self, p: Proposal) -> None:
        episode = p.payload.get("episode", "")
        res = await self.executor.run(p.payload["cmd"])
        self.journal.action("выполнить", res.ok, {"cmd": p.payload["cmd"], "text": res.text,
                                                   "reflex": res.reflex, "episode": episode, "stream": p.stream})
        self.ctx.reward(0.5 if res.ok else (-0.1 if res.reflex else -0.2), "цель", p.stream)
        reply = Event("результат", "выход", {"ok": res.ok, "text": res.text, "episode": episode,
                                            "hops": p.payload.get("hops", 1) + 1})
        if not self.bus.send(reply, p.payload.get("reply_to") or ""):
            waiter = self._waiting.get(episode)
            if waiter and not waiter.done():
                waiter.set_result(res.text)

    def _expire(self, p: Proposal) -> None:
        episode = p.payload.get("episode", "")
        self.ctx.stats["просрочено"] += 1
        self.journal.action("просрочено", False, {"episode": episode, "stream": p.stream, "орган": p.effector})
        self.ctx.reward(-0.3, "срок")
        waiter = self._waiting.get(episode)
        if waiter and not waiter.done():
            waiter.set_result(None)

    def snapshot(self) -> dict:
        edges = sorted(((e.w, a, b) for (a, b), e in self.graph.edges.items()), reverse=True)[:8]
        return {
            "фаза": self.stem.phase.value,
            "такт_с": round(self.stem.tact_s, 3),
            "сны": self.sleeps,
            "доля_матери": round(self.mother.share(), 3),
            "потоки": {n: {"очередь": s.queue.qsize(), "заработал": round(s.earned, 3),
                           "потратил": round(s.spent, 5)} for n, s in self.streams.items()},
            "связи": [f"{a}→{b}: {w:.3f}" for w, a, b in edges],
            "память": self.journal.memories(limit=3),
            "тело": {"ядер": self.body.physical_cpus, "ОЗУ_ГБ": self.body.ram_gb,
                     "предупреждения": self.body.warnings},
        }
