import asyncio
import tempfile
import unittest

from organism.config import Genome, Territory
from organism.llm import FakeLLM
from organism.organism import Organism
from organism.stem import Phase


class OrganismTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        genome = Genome(tact_s=0.02, wake_s=1.5, sleep_s=0.4, drowsy_max_s=2.0, eta=0.5, explore=0.0)
        territory = Territory(exec_allowlist=("git --version",), max_streams=4)
        self.org = Organism(self.tmp.name, FakeLLM(), territory, genome, seed=7)
        await self.org.start()

    async def asyncTearDown(self):
        await self.org.stop()
        self.tmp.cleanup()

    async def until(self, cond, limit=8.0):
        loop = asyncio.get_running_loop()
        end = loop.time() + limit
        while not cond():
            self.assertLess(loop.time(), end, "условие не наступило")
            await asyncio.sleep(0.02)

    async def test_three_default_streams_born(self):
        self.assertEqual(set(self.org.streams), {"речь", "моторика", "тело"})
        self.assertAlmostEqual(sum(self.org.stem.reserved.values()), 0.6)

    async def test_plain_reply(self):
        self.assertIn("Слышу: привет", await self.org.say("привет"))

    async def test_command_goes_through_motor_and_single_output(self):
        reply = await self.org.say("выполни команду")
        self.assertIn("Команда выполнена", reply)
        self.assertIn("git version", reply)
        kinds = [a[1] for a in self.org.journal.all_actions_since(0)]
        self.assertEqual(kinds.count("выполнить"), 1)
        self.assertIn(("речь", "моторика"), self.org.graph.edges)

    async def test_chain_limit_cuts_loops(self):
        self.org.genome.max_chain = 0
        reply = await self.org.say("выполни команду")
        self.assertIn("Не смог передать", reply)

    async def test_reflex_blocks_unlisted_command(self):
        from organism.executor import Executor
        self.org.executor = Executor(Territory(exec_allowlist=()), self.org.stem)
        reply = await self.org.say("выполни команду")
        self.assertIn("вне списка", reply)

    async def test_sleep_blocks_input_and_consolidates(self):
        await self.org.say("выполни команду")
        self.org.feedback(1.0)
        await self.until(lambda: self.org.stem.phase is Phase.SLEEP)
        self.assertIsNone(await self.org.say("ты спишь?"))
        await self.until(lambda: self.org.stem.phase is Phase.WAKE)
        report = self.org.last_sleep_report
        self.assertEqual(self.org.sleeps, 1)
        self.assertGreaterEqual(report["оценено"], 1)
        self.assertGreater(self.org.graph.weight("речь", "моторика"), 0.0)
        self.assertTrue(self.org.journal.memories())
        self.assertIn("Слышу", await self.org.say("снова бодрствую"))

    async def test_breed_timeout_from_territory(self):
        from dataclasses import replace
        self.org.territory = replace(self.org.territory, generation_every_sleeps=1, breed_timeout_s=0.05)
        self.org.llm.delay_s = 0.2
        await self.org.say("привет")
        await self.until(lambda: self.org.sleeps >= 1 and self.org.stem.phase is Phase.WAKE)
        faults = [d for _, kind, _, d in self.org.journal.all_actions_since(0) if kind == "сбой"]
        self.assertIn({"сон": "поколение: таймаут"}, faults)

    async def test_slow_llm_misses_deadline(self):
        self.org.llm.delay_s = 0.5
        self.org.streams["речь"].deadline_s = 0.1
        self.assertIsNone(await self.org.say("медленно", timeout=3.0))
        self.assertGreaterEqual(self.org.ctx.stats["просрочено"], 1)

    async def test_overloaded_stream_divides(self):
        speech = self.org.streams["речь"]
        speech.max_queue = 5
        from organism.sleep import metabolism
        births, deaths = metabolism(self.org)
        self.assertEqual(births, ["речь-2"])
        self.assertIn("речь-2", self.org.streams)
        self.assertFalse(self.org.streams["речь-2"].protected)

    async def test_useless_clone_dies_after_two_sleeps(self):
        self.org.spawn("речь-2")
        from organism.sleep import metabolism
        clone = self.org.streams["речь-2"]
        clone.spent = 0.01
        metabolism(self.org)
        clone.spent = 0.01
        _, deaths = metabolism(self.org)
        self.assertEqual(deaths, ["речь-2"])
        self.assertNotIn("речь-2", self.org.streams)

    async def test_territory_caps_stream_count(self):
        self.assertTrue(self.org.spawn("речь-2").ok)
        self.assertFalse(self.org.spawn("речь-3").ok)


if __name__ == "__main__":
    unittest.main()
