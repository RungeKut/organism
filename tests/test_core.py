import asyncio
import random
import unittest

from organism.config import Genome, Territory
from organism.executor import Executor
from organism.graph import Graph
from organism.protocol import Envelope
from organism.stem import Phase, Stem
from organism.workspace import Proposal, Workspace


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class ProtocolTest(unittest.TestCase):
    def test_roundtrip(self):
        env = Envelope("Моторика", "просьба", "git status\nвторая строка", 0.7)
        back = Envelope.parse(env.render())
        self.assertEqual((back.to, back.kind, back.confidence), ("моторика", "просьба", 0.7))
        self.assertEqual(back.text, "git status\nвторая строка")

    def test_garbage(self):
        self.assertIsNone(Envelope.parse("просто болтовня без полей"))
        self.assertIsNone(Envelope.parse("кому: x\nтип: чушь\nтекст: y"))

    def test_confidence_clamped(self):
        self.assertEqual(Envelope("а", "факт", "б", 7).confidence, 1.0)
        self.assertEqual(Envelope.parse("кому: а\nтип: факт\nуверенность: abc\nтекст: б").confidence, 0.5)


class GraphTest(unittest.TestCase):
    def setUp(self):
        self.g = Genome(eta=0.5, decay=0.0, explore=0.0)
        self.graph = Graph(self.g, random.Random(1))

    def test_reward_grows_weight_of_fired_edge_only(self):
        self.graph.fire("а", "б")
        self.graph.edge("а", "в")
        self.graph.reward(1.0)
        self.graph.consolidate()
        self.assertGreater(self.graph.weight("а", "б"), 0.0)
        self.assertEqual(self.graph.weight("а", "в"), 0.0)

    def test_unused_zero_edge_pruned(self):
        self.graph.edge("а", "в")
        self.graph.consolidate()
        self.assertNotIn(("а", "в"), self.graph.edges)

    def test_step_is_capped(self):
        self.graph.fire("а", "б")
        self.graph.reward(100.0)
        self.graph.consolidate()
        self.assertLessEqual(self.graph.weight("а", "б"), self.g.max_dw)

    def test_incoming_normalized(self):
        for src in "абв":
            self.graph.edge(src, "г").w = 0.9
        self.graph.fire("а", "г")
        self.graph._normalize_incoming()
        self.assertAlmostEqual(sum(self.graph.weight(s, "г") for s in "абв"), 1.0)

    def test_route_prefers_stronger_and_forgets_dead(self):
        self.graph.announce("р1", ("речь",))
        self.graph.announce("р2", ("речь",))
        self.graph.edge("п", "р2").w = 0.5
        self.assertEqual({self.graph.route("п", "речь") for _ in range(20)}, {"р2"})
        self.graph.forget("р2")
        self.assertEqual(self.graph.route("п", "речь"), "р1")
        self.assertIsNone(self.graph.route("п", "нет такой"))

    def test_route_skips_busy_clone(self):
        self.graph.announce("р", ("речь",))
        self.graph.announce("р-2", ("речь",))
        self.graph.edge("п", "р").w = 0.5
        load = {"р": 0, "р-2": 0}
        self.graph.load = load.get
        self.assertEqual(self.graph.route("п", "речь"), "р")
        load["р"] = 1
        self.assertEqual({self.graph.route("п", "речь") for _ in range(20)}, {"р-2"})
        load["р-2"] = 1
        self.assertEqual({self.graph.route("п", "речь") for _ in range(20)}, {"р"})


class StemTest(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.g = Genome(wake_s=10, sleep_s=5, drowsy_max_s=3, sleep_pressure_events=5)
        self.stem = Stem(Territory(max_streams=2, headroom=0.25), self.g, self.clock)

    def test_wake_to_sleep_to_wake(self):
        self.assertTrue(self.stem.accepts_input())
        self.clock.t = 10
        self.assertIs(self.stem.tick(busy=0), Phase.DROWSY)
        self.assertFalse(self.stem.accepts_input())
        self.assertIs(self.stem.tick(busy=0), Phase.SLEEP)
        self.clock.t = 100
        self.assertIsNone(self.stem.tick(busy=0))
        self.stem.sleep_done()
        self.assertIs(self.stem.tick(busy=0), Phase.WAKE)

    def test_drowsy_waits_for_work_then_gives_up(self):
        self.clock.t = 10
        self.stem.tick(busy=0)
        self.clock.t = 11
        self.assertIsNone(self.stem.tick(busy=2))
        self.clock.t = 13.5
        self.assertIs(self.stem.tick(busy=2), Phase.SLEEP)

    def test_event_pressure_forces_sleep(self):
        for _ in range(5):
            self.stem.note_event()
        self.assertIs(self.stem.tick(busy=0), Phase.DROWSY)

    def test_tact_bounds(self):
        self.assertEqual(self.stem.request_tact(1000), 60.0)
        self.assertEqual(self.stem.request_tact(0.0001), 0.05)

    def test_stream_limits(self):
        self.assertTrue(self.stem.request_stream("а", 1.0, capacity=4).ok)
        self.assertFalse(self.stem.request_stream("б", 2.5, capacity=4).ok)
        self.assertTrue(self.stem.request_stream("в", 1.0, capacity=4).ok)
        self.assertFalse(self.stem.request_stream("г", 0.1, capacity=100).ok)
        self.stem.release_stream("а")
        self.assertTrue(self.stem.request_stream("г", 0.1, capacity=100).ok)

    def test_action_rate_limit(self):
        stem = Stem(Territory(max_actions_per_hour=2), self.g, self.clock)
        self.assertIsNone(stem.action_denied())
        self.assertIsNone(stem.action_denied())
        self.assertIsNotNone(stem.action_denied())
        self.clock.t = 3601
        self.assertIsNone(stem.action_denied())


class WorkspaceTest(unittest.TestCase):
    def prop(self, stream, effector, salience, born=0.0, deadline=10.0):
        return Proposal(stream, effector, {}, salience, 1.0, born, deadline)

    def test_best_per_effector_losers_wait(self):
        ws = Workspace()
        ws.propose(self.prop("а", "голос", 0.4))
        ws.propose(self.prop("б", "голос", 0.9))
        ws.propose(self.prop("в", "рука", 0.1))
        winners, expired = ws.select(now=1.0)
        self.assertEqual({p.stream for p in winners}, {"б", "в"})
        self.assertEqual(expired, [])
        self.assertEqual([p.stream for p in ws.pending], ["а"])

    def test_expiry(self):
        ws = Workspace()
        ws.propose(self.prop("а", "голос", 0.4, deadline=1.0))
        winners, expired = ws.select(now=2.0)
        self.assertEqual((winners, [p.stream for p in expired]), ([], ["а"]))


class ExecutorTest(unittest.IsolatedAsyncioTestCase):
    def make(self, **kw):
        t = Territory(exec_allowlist=("git --version",), **kw)
        return Executor(t, Stem(t, Genome(), Clock()))

    async def test_allowed_command_runs(self):
        res = await self.make().run("git   --version")
        self.assertTrue(res.ok, res.text)
        self.assertIn("git version", res.text)

    async def test_reflex_refuses_outside_allowlist(self):
        res = await self.make().run("git push --force")
        self.assertFalse(res.ok)
        self.assertEqual(res.reflex, "вне_списка")

    async def test_disabled_and_rate_limited(self):
        self.assertEqual((await self.make(exec_enabled=False).run("git --version")).reflex, "выключено")
        ex = self.make(max_actions_per_hour=1)
        self.assertTrue((await ex.run("git --version")).ok)
        self.assertEqual((await ex.run("git --version")).reflex, "лимит")


if __name__ == "__main__":
    unittest.main()
