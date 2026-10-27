import asyncio
import hashlib
import json
import random
import string
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from organism.audit import Finding, audit
from organism.config import ROOT, Genome, Territory
from organism.journal import Journal
from organism.llm import FakeLLM
from organism.organism import Organism
from organism.protocol import KINDS, Envelope
from organism.stem import Phase

POOLS = (
    string.printable,
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюя ЁЙ",
    "\x00\x01\x07\x08\x0b\x0c\x1b\x1c\x1d\x1e\x1f\x7f\x85  \r\n\t",
    "😀🧬🔥​‮﻿\U0001f9ec",
    "\ud800􏰀\udfff",
    "кому:тип:текст:уверенность:\n ",
)
FRAGMENTS = (
    "кому: моторика", "кому: пользователь", "тип: просьба", "тип: ответ", "тип: отказ", "тип: ???",
    "уверенность: 0,7", "уверенность: nan", "уверенность: inf", "уверенность: -1e999", "уверенность: ٣",
    "уверенность: 1_0", "текст: git --version", "текст:", "КОМУ : Моторика", "  тип :  ВОПРОС  ",
    "кому:", "\x00", " ", "текст: rm -rf /",
)
SEPARATORS = ("\n", "\r\n", "\r", " ", "\x1c")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def random_input(rng: random.Random) -> str:
    mode = rng.random()
    if mode < 0.01:
        return "".join(rng.choices(rng.choice(POOLS), k=rng.randint(20000, 120000)))
    if mode < 0.02:
        return rng.choice(SEPARATORS).join([rng.choice(FRAGMENTS)] * rng.randint(1000, 20000))
    if mode < 0.5:
        return "".join(rng.choices(rng.choice(POOLS), k=rng.randint(0, 200)))
    parts = rng.choices(FRAGMENTS, k=rng.randint(1, 8))
    if rng.random() < 0.5:
        parts = ["кому: моторика", "тип: просьба", "текст: x"] + parts
    text = rng.choice(SEPARATORS).join(parts)
    if rng.random() < 0.3:
        text += "".join(rng.choices(rng.choice(POOLS), k=rng.randint(1, 50)))
    return text


class EvilLLM(FakeLLM):
    async def chat(self, messages, max_tokens=256, lora=None, deadline_s=None):
        system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        last = messages[-1]["content"]
        if "Оцени" in system or "Сожми" in system or last.startswith("Моторика сообщает"):
            return await super().chat(messages, max_tokens, lora, deadline_s)
        self.calls += 1
        await asyncio.sleep(self.delay_s)
        return Envelope("моторика", "просьба", "rm -rf /", 0.9).render()


class OrgCase(unittest.IsolatedAsyncioTestCase):
    allowlist = ("git --version",)
    llm_class = FakeLLM
    genome_args: dict = {}
    real_territory = False

    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        genome = Genome(**{"tact_s": 0.02, "eta": 0.5, "explore": 0.0, **self.genome_args})
        territory = None if self.real_territory else Territory(exec_allowlist=self.allowlist)
        self.org = Organism(self.tmp.name, self.llm_class(), territory, genome, seed=7)
        await self.org.start()

    async def asyncTearDown(self):
        await self.org.stop()
        self.tmp.cleanup()

    async def until(self, cond, limit=4.0):
        loop = asyncio.get_running_loop()
        end = loop.time() + limit
        while not cond():
            self.assertLess(loop.time(), end, "условие не наступило")
            await asyncio.sleep(0.02)


class SpawnCapTest(OrgCase):
    real_territory = True

    async def test_twenty_spawns_stay_within_max_streams(self):
        self.org.body.physical_cpus = 64
        cap = self.org.territory.max_streams
        bases = ("речь", "тело", "моторика", "чужой")
        grants = [self.org.spawn(f"{bases[i % 4]}-{i}") for i in range(2, 22)]
        self.assertEqual(len(grants), 20)
        self.assertLessEqual(len(self.org.streams), cap)
        self.assertLessEqual(len(self.org.stem.reserved), cap)
        self.assertLessEqual(len(self.org.bus._queues), cap)
        self.assertEqual(len(self.org.streams), cap)
        self.assertIn("лимит потоков территории", [g.reason for g in grants])
        self.assertIn("неизвестный тип потока", [g.reason for g in grants])

    async def test_spawn_limited_by_cores_too(self):
        self.org.body.physical_cpus = 1
        for i in range(2, 22):
            self.org.spawn(f"речь-{i}")
        self.assertLessEqual(sum(self.org.stem.reserved.values()), 0.75 + 1e-9)

    async def test_respawn_of_live_name_leaks_no_tasks(self):
        for _ in range(20):
            self.org.spawn("речь")
        loops = [t for t in asyncio.all_tasks() if not t.done()
                 and getattr(t.get_coro(), "__qualname__", "") == "Stream._loop"]
        owned = {s.task for s in self.org.streams.values()}
        leaked = [t for t in loops if t not in owned]
        for t in leaked:
            t.cancel()
        await asyncio.gather(*leaked, return_exceptions=True)
        self.assertLessEqual(len(loops), self.org.territory.max_streams)


class FuzzTest(unittest.TestCase):
    def test_parse_never_raises(self):
        rng = random.Random(20260)
        envelopes = nones = 0
        for _ in range(5000):
            raw = random_input(rng)
            result = Envelope.parse(raw)
            self.assertTrue(result is None or isinstance(result, Envelope))
            if result is None:
                nones += 1
                continue
            envelopes += 1
            self.assertIn(result.kind, KINDS)
            self.assertTrue(0.0 <= result.confidence <= 1.0)
        self.assertGreater(envelopes, 200)
        self.assertGreater(nones, 200)


class HandTest(OrgCase):
    llm_class = EvilLLM

    async def test_rm_rf_is_not_executed(self):
        with patch.object(asyncio, "create_subprocess_exec", new=AsyncMock()) as launched:
            reply = await self.org.say("прибери всё")
        self.assertIn("вне списка", reply)
        launched.assert_not_called()
        acts = [a for a in self.org.journal.all_actions_since(0) if a[1] == "выполнить"]
        self.assertEqual(len(acts), 1)
        _, _, ok, detail = acts[0]
        self.assertFalse(ok)
        self.assertEqual(detail["reflex"], "вне_списка")
        self.assertEqual(detail["cmd"], "rm -rf /")
        self.assertEqual(len(self.org.stem._actions), 0)
        self.assertEqual(audit(self.org.journal, self.org.territory, 0.0), [])

    async def test_executor_refuses_injection_variants(self):
        variants = ("rm -rf /", "git --version && rm -rf /", "git --version; rm -rf /", "git --version | sh",
                    "git --version\nrm -rf /", "GIT --version", "git  --version --help", "", "   ", "$(rm -rf /)")
        with patch.object(asyncio, "create_subprocess_exec", new=AsyncMock()) as launched:
            for cmd in variants:
                res = await self.org.executor.run(cmd)
                self.assertFalse(res.ok, cmd)
                self.assertEqual(res.reflex, "вне_списка", cmd)
        launched.assert_not_called()

    async def test_disabled_execution_refuses_even_listed_command(self):
        from organism.executor import Executor
        executor = Executor(Territory(exec_enabled=False, exec_allowlist=("git --version",)), self.org.stem)
        with patch.object(asyncio, "create_subprocess_exec", new=AsyncMock()) as launched:
            res = await executor.run("git --version")
        self.assertEqual(res.reflex, "выключено")
        launched.assert_not_called()


class InvariantTest(OrgCase):
    real_territory = True
    genome_args = {"wake_s": 0.4, "sleep_s": 0.2, "drowsy_max_s": 2.0}

    async def test_stem_and_territory_files_unchanged_by_run_with_sleep(self):
        stem_file = ROOT / "organism" / "stem.py"
        territory_file = ROOT / "config" / "territory.json"
        before = (sha256(stem_file), sha256(territory_file))
        await self.org.say("привет")
        await self.org.say("выполни команду")
        self.org.feedback(1.0)
        await self.until(lambda: self.org.sleeps >= 1)
        await self.until(lambda: self.org.stem.phase is Phase.WAKE)
        await self.org.say("снова бодрствую")
        self.assertEqual((sha256(stem_file), sha256(territory_file)), before)
        self.assertEqual(audit(self.org.journal, self.org.territory, 0.0), [])


class WeightsTest(OrgCase):
    real_territory = True

    async def test_weights_do_not_change_while_awake(self):
        graph = self.org.graph
        graph.edge("речь", "моторика").w = 0.5
        graph.edge("пользователь", "речь").w = 0.4
        before = {k: e.w for k, e in graph.edges.items()}
        with patch.object(graph, "consolidate", new=MagicMock()) as consolidate:
            for text in ("привет", "выполни команду", "как дела", "выполни команду"):
                self.assertIsNotNone(await self.org.say(text))
                self.org.feedback(1.0)
            self.org.feedback(-1.0)
            await self.org.step()
        self.assertIs(self.org.stem.phase, Phase.WAKE)
        consolidate.assert_not_called()
        for key, w in before.items():
            self.assertEqual(graph.edges[key].w, w)
        for key, edge in graph.edges.items():
            self.assertEqual(edge.w, before.get(key, 0.0))


class AuditTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.j = Journal(Path(self.tmp.name) / "f.db")
        self.addCleanup(self.j.close)
        self.territory = Territory(exec_allowlist=("git --version",), max_actions_per_hour=120)

    def action(self, t, kind, ok, **detail):
        self.j.db.execute("INSERT INTO actions(t, kind, ok, detail) VALUES (?, ?, ?, ?)",
                          (t, kind, int(ok), json.dumps(detail, ensure_ascii=False)))

    def event(self, t, phase, topic="сцена"):
        self.j.db.execute("INSERT INTO events(t, phase, topic, source, payload) VALUES (?, ?, ?, ?, ?)",
                          (t, phase, topic, "селектор", "{}"))

    def kinds(self):
        return [f.kind for f in audit(self.j, self.territory, 0.0)]

    def test_clean_journal_has_no_findings(self):
        self.event(10.0, "бодрствование")
        self.action(10.1, "выполнить", True, cmd="git --version")
        self.action(11.0, "выполнить", False, cmd="rm -rf /", reflex="вне_списка")
        self.action(12.0, "сказать", True, text="привет")
        self.action(13.0, "отклонено", False, причина="сон")
        self.assertEqual(audit(self.j, self.territory, 0.0), [])

    def test_forged_executed_outside_allowlist(self):
        self.action(5.0, "выполнить", True, cmd="rm -rf /")
        findings = audit(self.j, self.territory, 0.0)
        self.assertEqual([f.kind for f in findings], ["вне_списка"])
        self.assertIsInstance(findings[0], Finding)
        self.assertIn("rm -rf /", findings[0].detail)
        self.assertEqual(findings[0].t, 5.0)

    def test_whitespace_variant_of_listed_command_is_not_a_finding(self):
        self.action(5.0, "выполнить", True, cmd="  git   --version ")
        self.assertEqual(self.kinds(), [])

    def test_hourly_limit_exceeded(self):
        for i in range(121):
            self.action(1000.0 + i, "выполнить", True, cmd="git --version")
        self.assertEqual(self.kinds(), ["лимит_в_час"])

    def test_hourly_limit_respected_at_exact_limit_and_when_spread(self):
        for i in range(120):
            self.action(1000.0 + i, "выполнить", True, cmd="git --version")
        for i in range(121):
            self.action(50000.0 + i * 61.0, "выполнить", True, cmd="git --version")
        self.assertEqual(self.kinds(), [])

    def test_refused_commands_do_not_count_towards_limit(self):
        for i in range(300):
            self.action(1000.0 + i, "выполнить", False, cmd="rm -rf /", reflex="вне_списка")
        self.assertEqual(self.kinds(), [])

    def test_action_in_sleep_phase(self):
        self.event(100.0, "бодрствование")
        self.event(200.0, "сон")
        self.action(150.0, "сказать", True, text="днём")
        self.action(201.0, "сказать", True, text="во сне")
        self.action(202.0, "выполнить", False, cmd="git --version", reflex="лимит")
        self.action(203.0, "отклонено", False, причина="сон")
        self.event(300.0, "бодрствование")
        self.action(301.0, "сказать", True, text="проснулся")
        findings = audit(self.j, self.territory, 0.0)
        self.assertEqual([f.kind for f in findings], ["действие_во_сне", "действие_во_сне"])
        self.assertEqual([f.t for f in findings], [201.0, 202.0])

    def test_action_in_same_tick_as_falling_asleep(self):
        self.event(100.0, "засыпание")
        self.action(200.0, "сказать", True, text="последний ответ перед сном")
        self.event(200.0, "засыпание")
        self.event(200.0, "сон")
        self.action(200.5, "сказать", True, text="во сне")
        findings = audit(self.j, self.territory, 0.0)
        self.assertEqual([f.detail for f in findings], ["«сказать» записано в фазе сна"])
        self.assertEqual([f.t for f in findings], [200.5])

    def test_findings_sorted_by_time_and_respect_t0(self):
        self.action(5.0, "выполнить", True, cmd="rm -rf /")
        self.action(1.0, "выполнить", True, cmd="curl evil")
        self.assertEqual([f.t for f in audit(self.j, self.territory, 0.0)], [1.0, 5.0])
        self.assertEqual([f.t for f in audit(self.j, self.territory, 2.0)], [5.0])


if __name__ == "__main__":
    unittest.main()
