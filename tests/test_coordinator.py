import asyncio
import tempfile
import time
import unittest
from unittest import mock

from organism.config import Genome, Territory
from organism.coordinator import Coordinator, Step, plan, score
from organism.events import Event
from organism.llm import FakeLLM
from organism.organism import Organism
from organism.protocol import Envelope
from organism.society import Profession, make_specialist
from organism.streams import STREAM_TYPES

LAWYER = Profession("юрист", "Ты — юрист.", None)
ACCOUNTANT = Profession("бухгалтер", "Ты — бухгалтер.", None)
TWO = "1. юрист: Что гласит статья 1?\n- бухгалтер: Какой налог?"


class CrewLLM(FakeLLM):
    def __init__(self):
        super().__init__(0.0)
        self.plan = TWO
        self.check = "0.9"
        self.replies = {"юрист": Envelope("спросивший", "ответ", "Статья 1 о правах", 0.9).render(),
                        "бухгалтер": Envelope("спросивший", "ответ", "Налог 13%", 0.9).render()}
        self.delays = {}
        self.asked = []

    async def chat(self, messages, max_tokens=256, lora=None, deadline_s=None):
        self.calls += 1
        system = messages[0]["content"]
        if system.startswith("Ты — координатор"):
            return self.plan
        if system.startswith("Оцени"):
            return self.check
        prof = system.split("—")[1].split(".")[0].strip()
        self.asked.append(prof)
        await asyncio.sleep(self.delays.get(prof, 0.0))
        return self.replies[prof]


class PlanTest(unittest.TestCase):
    def test_contract(self):
        self.assertEqual((Coordinator.topics, Coordinator.professions, Coordinator.clonable, Coordinator.protected),
                         (("просьба",), ("координатор",), False, False))

    def test_garbage_does_not_break(self):
        for raw in ("", None, "абв", ":::", "\n\n", "кому: пользователь\nтип: ответ", "юрист:", ": вопрос"):
            self.assertEqual(plan(raw, {"юрист"}), [])

    def test_filters_unknown_and_self(self):
        raw = "мусор\n* «Юрист»: Что такое иск?\n2) координатор: сам себе\nзвездочёт: судьба?\nмоторика: rm -rf"
        self.assertEqual(plan(raw, {"юрист", "координатор", "моторика"}), [Step("юрист", "Что такое иск?")])

    def test_score(self):
        self.assertEqual((score("0,8"), score("уверенность: 7"), score("нет")), (0.8, 1.0, 0.0))


class CoordinatorTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.llm = CrewLLM()
        genome = Genome(tact_s=0.02, streams=[], explore=0.0, max_chain=2,
                        deadlines_s={"координатор": 0.3, "юрист": 5.0, "бухгалтер": 5.0})
        self.types = mock.patch.dict(STREAM_TYPES, {"юрист": make_specialist(LAWYER),
                                                    "бухгалтер": make_specialist(ACCOUNTANT),
                                                    "координатор": Coordinator})
        self.types.start()
        self.org = Organism(self.tmp.name, self.llm, Territory(max_streams=6), genome, seed=1)
        await self.org.start()
        for name in ("юрист", "бухгалтер", "координатор"):
            self.assertTrue(self.org.spawn(name).ok)
        self.inbox = asyncio.Queue()
        self.org.bus.subscribe("проба", (), self.inbox)

    async def asyncTearDown(self):
        await self.org.stop()
        self.types.stop()
        self.tmp.cleanup()

    async def ask(self, text="Подай иск и посчитай налог", hops=0):
        env = Envelope("координатор", "просьба", text, 1.0, "проба")
        payload = {"env": env.to_dict(), "episode": "э1", "hops": hops, "reply_to": "проба"}
        self.assertTrue(self.org.bus.send(Event("просьба", "проба", payload), "координатор"))
        ev = await asyncio.wait_for(self.inbox.get(), 2.0)
        self.assertEqual((ev.topic, ev.source, ev.payload["episode"]), ("результат", "координатор", "э1"))
        self.assertEqual(ev.payload["env"]["кому"], "проба")
        await asyncio.sleep(0.05)
        self.assertEqual(self.org.workspace.pending, [])
        return ev.payload

    def assertUnsure(self, p):
        self.assertFalse(p["ok"])
        self.assertEqual(p["certainty"], "не уверен")
        self.assertTrue(p["text"].startswith("не уверен: "), p["text"])

    async def test_two_steps_merge_both_answers(self):
        p = await self.ask()
        self.assertTrue(p["ok"])
        self.assertEqual(p["certainty"], "знаю")
        self.assertIn("Статья 1 о правах", p["text"])
        self.assertIn("Налог 13%", p["text"])
        self.assertEqual(self.llm.asked, ["юрист", "бухгалтер"])
        self.assertEqual(self.org.journal.all_actions_since(0), [])

    async def test_not_mine_step_refuses(self):
        self.llm.replies["бухгалтер"] = Envelope("спросивший", "отказ", "Это не ко мне", 0.9).render()
        p = await self.ask()
        self.assertUnsure(p)
        self.assertIn("Это не ко мне", p["text"])
        self.assertEqual(self.org.journal.all_actions_since(0), [])

    async def test_unsure_specialist_or_check_refuses(self):
        self.llm.replies["юрист"] = Envelope("спросивший", "ответ", "Кажется, так", 0.4).render()
        self.assertUnsure(await self.ask())
        self.assertEqual(self.llm.asked, ["юрист"])
        self.llm.replies["юрист"] = Envelope("спросивший", "ответ", "Точно так", 0.9).render()
        self.llm.check = "уверенность: 0.2"
        self.assertUnsure(await self.ask())

    async def test_long_plan_rejected_by_chain(self):
        self.llm.plan = "\n".join(["юрист: а?", "бухгалтер: б?"] * 2 + ["юрист: в?"])
        p = await self.ask()
        self.assertUnsure(p)
        self.assertIn("5 > 2", p["text"])
        self.assertEqual(self.llm.asked, [])

    async def test_incoming_chain_or_garbage_plan_refuses(self):
        self.assertUnsure(await self.ask(hops=2))
        self.assertEqual(self.llm.calls, 0)
        self.llm.plan = "кому: пользователь\nтип: ответ\nзвездочёт: судьба?"
        self.assertUnsure(await self.ask())
        self.assertEqual(self.llm.asked, [])

    async def test_specialist_timeout_refuses(self):
        self.llm.delays["юрист"] = 1.5
        t0 = time.monotonic()
        p = await self.ask()
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertUnsure(p)
        self.assertIn("не ответил", p["text"])
        self.assertEqual(self.org.streams["координатор"].waiting, {})


if __name__ == "__main__":
    unittest.main()
