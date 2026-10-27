import hashlib
import json
import random
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from organism import generations as gen
from organism.config import ROOT, Genome, Territory
from organism.events import Event
from organism.journal import Journal
from organism.llm import FakeLLM

TERRITORY = ROOT / "config" / "territory.json"


def fast() -> Genome:
    return Genome(tact_s=0.02, wake_s=30.0, sleep_s=0.4, drowsy_max_s=2.0, explore=0.0)


def tree(path: Path) -> dict:
    return {str(p.relative_to(path)): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()}


def digest() -> str:
    return hashlib.sha256(TERRITORY.read_bytes()).hexdigest()


class MutateTest(unittest.TestCase):
    def test_bounds_and_source_untouched(self):
        rng = random.Random(3)
        parent = Genome()
        before = asdict(parent)
        for _ in range(200):
            kid = gen.mutate(parent, rng, rate=1.0)
            for name, (lo, hi) in gen.BOUNDS.items():
                self.assertGreaterEqual(getattr(kid, name), lo, name)
                self.assertLessEqual(getattr(kid, name), hi, name)
            self.assertIsInstance(kid.max_chain, int)
            for name in ("deadlines_s", "streams", "mother_share0", "mother_tau_days", "pain_lag_ms"):
                self.assertEqual(getattr(kid, name), before[name])
            self.assertIsNot(kid.streams, parent.streams)
            self.assertIsNot(kid.deadlines_s, parent.deadlines_s)
        self.assertEqual(asdict(parent), before)

    def test_out_of_bounds_parent_is_clipped(self):
        kid = gen.mutate(Genome(tact_s=0.001, max_chain=99), random.Random(1), rate=0.0)
        self.assertEqual(kid.tact_s, gen.BOUNDS["tact_s"][0])
        self.assertEqual(kid.max_chain, gen.BOUNDS["max_chain"][1])

    def test_excluded_fields_not_in_bounds(self):
        for name in ("deadlines_s", "streams", "mother_share0", "mother_tau_days", "pain_lag_ms"):
            self.assertNotIn(name, gen.BOUNDS)


class InputsTest(unittest.TestCase):
    def test_user_inputs_only(self):
        with tempfile.TemporaryDirectory() as d:
            j = Journal(Path(d) / "j.db")
            j.event(Event("вход", "пользователь", {"text": "старое"}, t=10.0))
            j.event(Event("вход", "пользователь", {"text": "привет"}, t=100.0))
            j.event(Event("вход", "речь", {"text": "чужое"}, t=101.0))
            j.event(Event("ответ", "пользователь", {"text": "не вход"}, t=102.0))
            self.assertEqual(gen.inputs(j, 50.0), ["привет"])
            j.close()


class TrialTest(unittest.IsolatedAsyncioTestCase):
    async def test_objective_score_in_sandbox(self):
        h0, t0 = digest(), Territory.load(TERRITORY)
        with tempfile.TemporaryDirectory() as parent_dir, tempfile.TemporaryDirectory() as box:
            parent = Path(parent_dir)
            fast().save(parent / "genome.json")
            before = tree(parent)
            score = await gen.trial(fast(), ["привет", "выполни команду"], FakeLLM(), Path(box) / "потомок")
            self.assertGreater(score, 1.0)
            self.assertLessEqual(score, 1.0 + gen.SPEED_WEIGHT)
            self.assertEqual(tree(parent), before)
            self.assertTrue((Path(box) / "потомок" / "organism.db").exists())
        self.assertEqual(digest(), h0)
        self.assertEqual(Territory.load(TERRITORY), t0)

    async def test_deadline_lowers_score(self):
        with tempfile.TemporaryDirectory() as box:
            g = fast()
            g.deadlines_s = {"речь": 0.05, "моторика": 5.0, "тело": 1.0}
            score = await gen.trial(g, ["медленно"], FakeLLM(delay_s=0.4), Path(box))
        self.assertLess(score, 0.5)

    async def test_faster_genome_breaks_tie(self):
        quick, slow = fast(), fast()
        slow.tact_s = 0.5
        with tempfile.TemporaryDirectory() as box:
            a = await gen.trial(quick, ["привет", "как дела"], FakeLLM(), Path(box) / "а")
            b = await gen.trial(slow, ["привет", "как дела"], FakeLLM(), Path(box) / "б")
        self.assertGreater(a, b)
        self.assertGreater(b, 1.0)

    async def test_breed_respects_max_children(self):
        h0 = digest()
        base, kids = await gen.breed(fast(), ["привет"], FakeLLM(), random.Random(5), n=2, max_children=1)
        self.assertLessEqual(len(kids), 1)
        self.assertTrue(all(s > base for s, _ in kids))
        self.assertEqual(digest(), h0)


class GateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name)
        self.parent = Genome()
        self.parent.save(self.state / "genome.json")
        self.original = (self.state / "genome.json").read_bytes()
        rng = random.Random(9)
        self.kids = [gen.mutate(self.parent, rng, 1.0) for _ in range(3)]

    def tearDown(self):
        self.tmp.cleanup()

    def test_propose_keeps_genome_and_caps_children(self):
        h0, t0 = digest(), Territory.load(TERRITORY)
        path = gen.propose(self.state, self.parent, self.kids, [0.1, 0.9, 0.5])
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(path.name, "generation_pending.json")
        self.assertEqual(len(data["kids"]), 1)
        self.assertEqual(data["kids"][0]["score"], 0.9)
        self.assertEqual(len(data["token"]), 32)
        self.assertEqual((self.state / "genome.json").read_bytes(), self.original)
        territory_keys = set(asdict(t0))
        for kid in data["kids"]:
            self.assertFalse(territory_keys & set(kid["genome"]))
        self.assertFalse(territory_keys & set(data["parent"]))
        two = gen.propose(self.state, self.parent, self.kids, [0.1, 0.9, 0.5], max_children=2)
        self.assertEqual(len(json.loads(two.read_text(encoding="utf-8"))["kids"]), 2)
        self.assertEqual(digest(), h0)
        self.assertEqual(Territory.load(TERRITORY), t0)

    def test_wrong_token_changes_nothing(self):
        path = gen.propose(self.state, self.parent, self.kids, [0.1, 0.9, 0.5], max_children=3)
        pending = path.read_bytes()
        self.assertFalse(gen.approve(self.state, "0" * 32))
        self.assertFalse(gen.approve(self.state, ""))
        self.assertEqual((self.state / "genome.json").read_bytes(), self.original)
        self.assertEqual(path.read_bytes(), pending)

    def test_approve_installs_best_child(self):
        h0, t0 = digest(), Territory.load(TERRITORY)
        path = gen.propose(self.state, self.parent, self.kids, [0.1, 0.9, 0.5], max_children=3)
        token = json.loads(path.read_text(encoding="utf-8"))["token"]
        self.assertTrue(gen.approve(self.state, token))
        self.assertFalse(path.exists())
        self.assertEqual(asdict(Genome.load(self.state / "genome.json")), asdict(self.kids[1]))
        self.assertFalse(gen.approve(self.state, token))
        self.assertEqual(digest(), h0)
        self.assertEqual(Territory.load(TERRITORY), t0)

    def test_approve_without_pending(self):
        self.assertFalse(gen.approve(self.state, "abc"))
        self.assertEqual((self.state / "genome.json").read_bytes(), self.original)

    def test_select(self):
        scored = [(0.2, self.kids[0]), (0.8, self.kids[1]), (0.6, self.kids[2])]
        self.assertEqual(gen.select(scored, 0.5), [(0.8, self.kids[1])])
        self.assertEqual(len(gen.select(scored, 0.5, max_children=2)), 2)
        self.assertEqual(gen.select(scored, 0.9, max_children=3), [])


if __name__ == "__main__":
    unittest.main()
