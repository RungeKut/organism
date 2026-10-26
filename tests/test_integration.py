import asyncio
import dataclasses
import tempfile
import unittest

from organism.config import Genome, Territory
from organism.llm import FakeLLM
from organism.organism import Organism
from organism.society import Profession, make_specialist
from organism.stem import Phase
from organism.streams import STREAM_TYPES
from organism.workspace import Proposal


def genome():
    return Genome(tact_s=0.02, wake_s=1.2, sleep_s=0.3, drowsy_max_s=2.0, eta=0.5, explore=0.0)


def territory():
    return Territory(exec_allowlist=("git --version",), max_streams=6)


class IntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.llm = FakeLLM()
        self.org = Organism(self.tmp.name, self.llm, territory(), genome(), seed=3)
        await self.org.start()

    async def asyncTearDown(self):
        await self.org.stop()
        self.tmp.cleanup()

    async def until(self, cond, limit=10.0):
        loop = asyncio.get_running_loop()
        end = loop.time() + limit
        while not cond():
            self.assertLess(loop.time(), end, "условие не наступило")
            await asyncio.sleep(0.02)

    async def sleep_cycle(self):
        target = self.org.sleeps + 1
        await self.until(lambda: self.org.sleeps >= target)
        await self.until(lambda: self.org.stem.phase is Phase.WAKE)

    def episodes(self, journal):
        return {p["episode"] for _, topic, _, p in journal.all_events_since(0) if topic == "вход"}

    async def test_episode_ids_unique_across_runs(self):
        await self.org.say("привет")
        first = self.episodes(self.org.journal)
        await self.org.stop()
        second_org = Organism(self.tmp.name, FakeLLM(), territory(), genome(), seed=4)
        await second_org.start()
        try:
            await second_org.say("привет")
            both = first | self.episodes(second_org.journal)
            self.assertEqual(len(both), 2)
        finally:
            await second_org.stop()
            self.org = Organism(self.tmp.name, FakeLLM(), territory(), genome(), seed=5)

    async def test_reflex_learned_in_sleep_skips_llm(self):
        for _ in range(3):
            self.assertIn("git version", await self.org.say("выполни команду"))
        await self.sleep_cycle()
        self.assertGreaterEqual(self.org.last_sleep_report["рефлексов"], 1)
        before = self.llm.calls
        reply = await self.org.say("выполни команду")
        self.assertIn("git version", reply)
        self.assertEqual(self.llm.calls, before)

    async def test_reflex_dropped_when_allowlist_narrows(self):
        for _ in range(3):
            await self.org.say("выполни команду")
        await self.sleep_cycle()
        self.org.ctx.territory = dataclasses.replace(self.org.territory, exec_allowlist=())
        before = self.llm.calls
        await self.org.say("выполни команду")
        self.assertGreater(self.llm.calls, before)

    async def test_stale_reflex_removed_on_next_sleep(self):
        for _ in range(3):
            await self.org.say("выполни команду")
        await self.sleep_cycle()
        self.assertTrue(self.org.reflexes.reflexes)
        self.org.journal.db.execute("DELETE FROM actions")
        await self.sleep_cycle()
        self.assertFalse(self.org.reflexes.reflexes)

    async def test_specialist_is_mortal_and_core_is_not(self):
        STREAM_TYPES["юрист"] = make_specialist(Profession("юрист", "Ты юрист.", None))
        try:
            self.assertTrue(self.org.spawn("юрист").ok)
            self.assertFalse(self.org.streams["юрист"].protected)
            self.assertTrue(self.org.streams["речь"].protected)
        finally:
            del STREAM_TYPES["юрист"]

    async def test_wave2_types_registered_and_drive_off_by_default(self):
        self.assertIn("координатор", STREAM_TYPES)
        self.assertIn("влечение", STREAM_TYPES)
        self.assertNotIn("влечение", self.org.streams)
        self.assertTrue(self.org.spawn("влечение").ok)

    async def test_no_hand_action_during_sleep(self):
        self.org.stem.phase = Phase.SLEEP
        now = self.org.clock()
        self.org.workspace.propose(Proposal("моторика", "рука", {"cmd": "git --version", "episode": "", "hops": 1},
                                            1.0, 1.0, now, now + 60))
        await asyncio.sleep(0.3)
        self.assertEqual(len(self.org.workspace.pending), 1)
        self.assertFalse([a for a in self.org.journal.all_actions_since(0) if a[1] == "выполнить"])
        self.org.stem.phase = Phase.WAKE
        await self.until(lambda: not self.org.workspace.pending)
        await self.until(lambda: any(a[1] == "выполнить" for a in self.org.journal.all_actions_since(0)))

    async def test_phase_changes_are_journaled(self):
        await self.sleep_cycle()
        phases = [p["фаза"] for _, topic, _, p in self.org.journal.all_events_since(0) if topic == "фаза"]
        self.assertEqual(phases[:3], ["засыпание", "сон", "бодрствование"])

    async def test_broken_sleep_does_not_hang_organism(self):
        async def boom(org):
            raise RuntimeError("сбой")
        from organism import organism as module
        original = module.sleep_run
        module.sleep_run = boom
        try:
            await self.until(lambda: self.org.stem.phase is Phase.SLEEP)
            await self.until(lambda: self.org.stem.phase is Phase.WAKE)
        finally:
            module.sleep_run = original
        failures = [a for a in self.org.journal.all_actions_since(0) if a[1] == "сбой"]
        self.assertTrue(failures)


if __name__ == "__main__":
    unittest.main()
