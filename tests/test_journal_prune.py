import asyncio
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

from organism.config import Genome, Territory
from organism.events import Event
from organism.journal import Journal
from organism.llm import FakeLLM
from organism.organism import Organism
from organism.stem import Phase


def add(journal, topic: str, t: float, text: str = "текст") -> None:
    journal.event(Event(topic, "тело", {"text": text}, t=t))


class PruneEventsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.j = Journal(Path(self.tmp.name) / "j.db")
        self.now = time.time()
        add(self.j, "ощущение", self.now - 10 * 3600)
        add(self.j, "ощущение", self.now - 7 * 3600)
        add(self.j, "ощущение", self.now - 5 * 3600)
        add(self.j, "ощущение", self.now - 0.1)
        add(self.j, "вход", self.now - 10 * 3600)
        add(self.j, "фаза", self.now - 10 * 3600)

    def tearDown(self):
        self.j.close()
        self.tmp.cleanup()

    def test_prunes_only_old_rows_of_topic(self):
        removed = self.j.prune_events("ощущение", self.now - 6 * 3600)
        self.assertEqual(removed, 2)
        rows = self.j.all_events_since(0)
        ts = [t for t, topic, _, _ in rows if topic == "ощущение"]
        self.assertEqual(len(ts), 2)
        self.assertTrue(all(t >= self.now - 6 * 3600 for t in ts))
        self.assertEqual([t for t, topic, _, _ in rows if topic == "вход"], [self.now - 10 * 3600])
        self.assertEqual([t for t, topic, _, _ in rows if topic == "фаза"], [self.now - 10 * 3600])

    def test_nothing_old_returns_zero_and_is_idempotent(self):
        self.assertEqual(self.j.prune_events("ощущение", self.now - 24 * 3600), 0)
        self.assertEqual(self.j.prune_events("ощущение", self.now - 6 * 3600), 2)
        self.assertEqual(self.j.prune_events("ощущение", self.now - 6 * 3600), 0)

    def test_default_keep_is_six_hours(self):
        self.assertEqual(Territory().sense_keep_hours, 6.0)


class SleepPruneTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        genome = Genome(tact_s=0.02, wake_s=1.5, sleep_s=0.4, drowsy_max_s=2.0, eta=0.5, explore=0.0)
        self.org = Organism(self.tmp.name, FakeLLM(), Territory(exec_allowlist=("git --version",)), genome, seed=3)
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

    async def test_sleep_prunes_old_sensations_keeps_new(self):
        now = time.time()
        add(self.org.journal, "ощущение", now - 10 * 3600, "старое")
        add(self.org.journal, "ощущение", now - 0.1, "свежее")
        await self.until(lambda: self.org.sleeps >= 1 and self.org.stem.phase is Phase.WAKE)
        rows = [p for _, topic, _, p in self.org.journal.all_events_since(0) if topic == "ощущение"]
        texts = [p.get("text") for p in rows]
        self.assertNotIn("старое", texts)
        self.assertIn("свежее", texts)

    async def test_sleep_without_prune_when_zero(self):
        self.org.territory = replace(self.org.territory, sense_keep_hours=0.0)
        now = time.time()
        add(self.org.journal, "ощущение", now - 10 * 3600, "старое")
        await self.until(lambda: self.org.sleeps >= 1 and self.org.stem.phase is Phase.WAKE)
        rows = [p for _, topic, _, p in self.org.journal.all_events_since(0) if topic == "ощущение"]
        self.assertIn("старое", [p.get("text") for p in rows])


if __name__ == "__main__":
    unittest.main()
