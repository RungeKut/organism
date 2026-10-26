import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from organism.config import Genome, Territory
from organism.events import Event
from organism.llm import FakeLLM, LLMError
from organism.organism import Organism
from organism.protocol import Envelope
from organism.society import Profession, certainty, load_professions, make_specialist
from organism.streams import STREAM_TYPES

LAWYER = Profession("юрист", "Ты — юрист.", 3)


class RecordingLLM(FakeLLM):
    def __init__(self):
        super().__init__(0.0)
        self.loras = []
        self.systems = []
        self.reply = Envelope("речь", "ответ", "Статья 1", 0.9).render()
        self.fail = False

    async def chat(self, messages, max_tokens=256, lora=None, deadline_s=None):
        self.calls += 1
        self.loras.append(lora)
        self.systems.append(messages[0]["content"])
        if self.fail:
            raise LLMError("нет связи")
        return self.reply


class LoadTest(unittest.TestCase):
    def test_missing_file_gives_empty(self):
        self.assertEqual(load_professions(Path(tempfile.gettempdir()) / "нет_такого.json"), [])

    def test_unknown_keys_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "professions.json"
            rows = [{"name": "юрист", "prompt": "Ты — юрист.", "lora_id": 3, "tools": ["поиск"], "лишнее": 1},
                    {"name": "счёт", "prompt": "Считай."}]
            path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            profs = load_professions(path)
        self.assertEqual(profs[0], Profession("юрист", "Ты — юрист.", 3, ("поиск",)))
        self.assertIsNone(profs[1].lora_id)

    def test_certainty(self):
        self.assertEqual(certainty(Envelope("а", "ответ", "т", 0.7)), "знаю")
        self.assertEqual(certainty(Envelope("а", "факт", "т", 0.69)), "не уверен")
        self.assertEqual(certainty(Envelope("а", "отказ", "т", 1.0)), "не моя")

    def test_specialist_contract(self):
        cls = make_specialist(LAWYER)
        self.assertEqual((cls.topics, cls.professions, cls.clonable, cls.protected),
                         (("просьба",), ("юрист",), False, False))


class SpecialistTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.llm = RecordingLLM()
        genome = Genome(tact_s=0.02, streams=[], explore=0.0)
        self.types = mock.patch.dict(STREAM_TYPES, {"юрист": make_specialist(LAWYER)})
        self.types.start()
        self.org = Organism(self.tmp.name, self.llm, Territory(max_streams=4), genome, seed=1)
        await self.org.start()
        self.inbox = asyncio.Queue()
        self.org.bus.subscribe("проба", (), self.inbox)

    async def asyncTearDown(self):
        await self.org.stop()
        self.types.stop()
        self.tmp.cleanup()

    def ask(self, target, text="Что гласит статья 1?", hops=0):
        env = Envelope("юрист", "вопрос", text, 1.0, "проба")
        payload = {"env": env.to_dict(), "episode": "э1", "hops": hops, "reply_to": "проба"}
        self.assertTrue(self.org.bus.send(Event("просьба", "проба", payload), target))

    async def result(self):
        ev = await asyncio.wait_for(self.inbox.get(), 2.0)
        self.assertEqual(ev.topic, "результат")
        return ev

    async def test_answers_sender_with_own_lora(self):
        self.org.spawn("юрист")
        self.assertEqual(self.org.graph.route("проба", "юрист"), "юрист")
        self.ask("юрист")
        ev = await self.result()
        self.assertEqual(self.llm.loras, [[{"id": 3, "scale": 1.0}]])
        self.assertTrue(self.llm.systems[0].startswith("Ты — юрист."))
        self.assertEqual((ev.source, ev.payload["ok"], ev.payload["certainty"]), ("юрист", True, "знаю"))
        self.assertEqual(ev.payload["env"]["кому"], "проба")
        self.assertEqual(ev.payload["hops"], 1)
        self.assertEqual(self.org.workspace.pending, [])

    async def test_not_mine_refuses_to_sender_only(self):
        self.llm.reply = Envelope("пользователь", "отказ", "Это не ко мне", 0.9).render()
        self.org.spawn("юрист")
        self.ask("юрист")
        ev = await self.result()
        self.assertEqual((ev.payload["ok"], ev.payload["certainty"]), (False, "не моя"))
        self.assertEqual(ev.payload["env"]["кому"], "проба")
        await asyncio.sleep(0.05)
        self.assertEqual(self.org.workspace.pending, [])
        self.assertEqual(self.org.journal.all_actions_since(0), [])

    async def test_low_confidence_is_unsure(self):
        self.llm.reply = Envelope("проба", "ответ", "Кажется, так", 0.4).render()
        self.org.spawn("юрист")
        self.ask("юрист")
        ev = await self.result()
        self.assertEqual((ev.payload["ok"], ev.payload["certainty"]), (True, "не уверен"))

    async def test_chain_limit_refuses_without_thinking(self):
        self.org.spawn("юрист")
        self.ask("юрист", hops=self.org.genome.max_chain)
        ev = await self.result()
        self.assertFalse(ev.payload["ok"])
        self.assertEqual(self.llm.calls, 0)

    async def test_llm_failure_refuses(self):
        self.llm.fail = True
        self.org.spawn("юрист")
        self.ask("юрист")
        ev = await self.result()
        self.assertFalse(ev.payload["ok"])
        self.assertIn("нет связи", ev.payload["text"])

    async def test_two_specialists_routed_by_weight(self):
        self.org.spawn("юрист")
        self.org.spawn("юрист-2")
        self.org.graph.edge("проба", "юрист-2").w = 0.9
        target = self.org.graph.route("проба", "юрист")
        self.assertEqual(target, "юрист-2")
        self.ask(target)
        ev = await self.result()
        self.assertEqual(ev.source, "юрист-2")


if __name__ == "__main__":
    unittest.main()
