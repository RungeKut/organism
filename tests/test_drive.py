import asyncio
import tempfile
import unittest
from unittest import mock

from organism import will
from organism.config import Genome, Territory
from organism.drive import DRIVE_CAP, RECALL, Drive
from organism.events import Event
from organism.organism import Organism
from organism.stem import Phase
from organism.streams import STREAM_TYPES
from organism.workspace import Proposal

ALLOWED = ("git --version", "git status")


class DriveCase(unittest.IsolatedAsyncioTestCase):
    explore = 0.0
    territory_args: dict = {}

    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.now = 1000.0
        patcher = mock.patch.dict(STREAM_TYPES, {"влечение": Drive})
        patcher.start()
        self.addCleanup(patcher.stop)
        genome = Genome(tact_s=0.02, wake_s=600.0, explore=self.explore, streams=[])
        territory = Territory(exec_allowlist=ALLOWED, **self.territory_args)
        self.org = Organism(self.tmp.name, None, territory, genome, clock=lambda: self.now, seed=3)
        self.assertTrue(self.org.spawn("влечение").ok)
        self.drive = self.org.streams["влечение"]
        self.step_s = self.drive.period * self.org.stem.tact_s
        self.addAsyncCleanup(self.org.stop)

    async def tick(self):
        self.now += self.step_s
        await self.drive.pulse()

    async def settle(self):
        await self.org.step()
        if self.org._acts:
            await asyncio.gather(*self.org._acts)
        await asyncio.sleep(0.01)

    async def cycle(self):
        await self.tick()
        await self.settle()

    def commands(self):
        return [d["cmd"] for _, _, _, d in self.org.journal.all_actions_since(0) if d.get("cmd")]


class DriveTest(DriveCase):
    def test_class_flags(self):
        self.assertEqual(self.drive.period, self.org.genome.vitals_every * 5)
        self.assertEqual(Drive.cores, 0.05)
        self.assertFalse(Drive.protected)
        self.assertFalse(Drive.clonable)
        self.assertEqual(Drive.topics, ("результат",))
        self.assertEqual(Drive.professions, ())

    async def test_proposes_allowed_action_on_empty_organism(self):
        for _ in range(4):
            await self.tick()
            if self.org.workspace.pending:
                break
        self.assertEqual(len(self.org.workspace.pending), 1)
        p = self.org.workspace.pending[0]
        self.assertEqual(p.effector, "рука")
        self.assertEqual(p.stream, "влечение")
        self.assertIn(p.payload["cmd"], ALLOWED)
        self.assertEqual(p.payload["episode"], "")
        self.assertEqual(p.payload["reply_to"], "влечение")
        self.assertEqual(p.payload["hops"], 1)
        self.assertAlmostEqual(p.salience, 0.2)

    async def test_diversity_grows_without_user(self):
        for _ in range(4):
            await self.cycle()
        self.assertEqual(set(self.commands()), set(ALLOWED))
        self.assertGreater(will.diversity(self.org.journal, 0), 0.0)

    async def test_novelty_prefers_least_chosen(self):
        picks = [self.drive.choose() for _ in range(6)]
        self.assertEqual(sorted(picks[:3]), sorted([*ALLOWED, RECALL]))
        self.assertEqual(sorted(picks[3:]), sorted([*ALLOWED, RECALL]))

    async def test_silent_with_pending(self):
        stub = Proposal("кто-то", "голос", {"text": "x"}, 0.5, 1.0, self.now, self.now + 100)
        self.org.workspace.propose(stub)
        for _ in range(4):
            await self.tick()
        self.assertEqual(self.org.workspace.pending, [stub])

    async def test_silent_after_recent_user_input(self):
        await self.tick()
        self.org.workspace.pending.clear()
        self.drive.busy_until = 0.0
        self.org.journal.event(Event("вход", "пользователь", {"text": "привет", "episode": "э1", "hops": 0}))
        await self.tick()
        self.assertEqual(self.org.workspace.pending, [])
        self.assertEqual(self.drive.last_input, self.now)
        idle = self.drive.idle_s()
        self.now += idle * 0.5
        await self.drive.pulse()
        self.assertEqual(self.org.workspace.pending, [])
        self.now += idle
        await self.drive.pulse()
        self.assertEqual(len(self.org.workspace.pending), 1)

    async def test_silent_outside_wake(self):
        self.org.stem.phase = Phase.SLEEP
        for _ in range(4):
            await self.tick()
        self.assertEqual(self.org.workspace.pending, [])

    async def test_cap_respected(self):
        with mock.patch("organism.drive.DRIVE_CAP", 3):
            for _ in range(12):
                await self.cycle()
        proposals = [c for c in self.commands() if c in ALLOWED]
        self.assertEqual(len(proposals), 3)

    async def test_cap_resets_on_new_wake(self):
        self.assertEqual(DRIVE_CAP, 20)
        with mock.patch("organism.drive.DRIVE_CAP", 2):
            for _ in range(8):
                await self.cycle()
            self.assertEqual(self.drive.acts, 2)
            self.org.stem.since += 1.0
            for _ in range(3):
                await self.cycle()
        self.assertEqual(self.drive.acts, 2)
        self.assertEqual(len([c for c in self.commands() if c in ALLOWED]), 4)

    async def test_recall_is_not_an_action(self):
        self.drive.counts = {c: 5 for c in ALLOWED}
        self.org.journal.remember("факт")
        await self.tick()
        self.assertEqual(self.org.workspace.pending, [])
        self.assertEqual(self.drive.recalled, ["факт"])
        self.assertEqual(self.drive.acts, 0)
        self.assertEqual(self.org.journal.all_actions_since(0), [])

    async def test_no_wait_for_unanswered_proposal(self):
        await self.tick()
        self.assertEqual(len(self.org.workspace.pending), 1)
        self.org.workspace.pending.clear()
        await self.tick()
        self.assertEqual(self.org.workspace.pending, [])
        self.now += self.drive.deadline_s
        await self.drive.pulse()
        self.assertEqual(len(self.org.workspace.pending), 1)

    async def test_reward_only_for_new_output(self):
        def ev(text):
            return Event("результат", "выход", {"ok": True, "text": text, "episode": "", "hops": 2})
        await self.drive.handle(ev("версия 1"))
        await self.drive.handle(ev("версия 1"))
        await self.drive.handle(ev("версия   1\n"))
        rewards = self.org.journal.rewards_since(0, "любопытство")
        self.assertEqual([v for _, v in rewards], [0.1])
        await self.drive.handle(ev("версия 2"))
        await self.drive.handle(ev(""))
        rewards = self.org.journal.rewards_since(0, "любопытство")
        self.assertEqual([v for _, v in rewards], [0.1, 0.1])
        self.assertAlmostEqual(self.drive.earned, 0.2)

    async def test_repeated_real_output_not_rewarded(self):
        for _ in range(8):
            await self.cycle()
        outputs = {d["text"] for _, _, _, d in self.org.journal.all_actions_since(0) if d.get("cmd") in ALLOWED}
        rewards = self.org.journal.rewards_since(0, "любопытство")
        self.assertEqual(len(rewards), len(outputs))


class DriveExploreTest(DriveCase):
    explore = 1.0

    async def test_never_proposes_outside_allowlist(self):
        seen = set()
        for _ in range(60):
            await self.tick()
            for p in self.org.workspace.pending:
                seen.add(p.payload["cmd"])
            self.org.workspace.pending.clear()
            self.drive.busy_until = 0.0
            self.drive.acts = 0
        self.assertTrue(seen)
        self.assertLessEqual(seen, set(ALLOWED))


class DriveNoExecTest(DriveCase):
    territory_args = {"exec_enabled": False}

    async def test_only_recall_when_exec_off(self):
        for _ in range(4):
            await self.tick()
        self.assertEqual(self.org.workspace.pending, [])
        self.assertEqual(self.drive.repertoire(), [RECALL])


if __name__ == "__main__":
    unittest.main()
