import tempfile
import unittest

from organism.config import Genome, Territory
from organism.ladder import Ladder, Rung
from organism.llm import FakeLLM
from organism.organism import Organism


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class LadderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        genome = Genome(tact_s=0.1, streams=[], explore=0.0)
        self.org = Organism(self.tmp.name, FakeLLM(0.0), Territory(max_streams=4), genome, seed=1)
        self.ctx = self.org.ctx
        self.clock = Clock()
        self.ctx.clock = self.clock
        self.ladder = Ladder(self.ctx)

    def tearDown(self):
        self.org.journal.close()
        self.tmp.cleanup()

    def series(self, n=50, prefix="вопрос"):
        return [self.ladder.pick(f"{prefix} {i}") for i in range(n)]

    def test_calm_unknown_text_thinks(self):
        self.assertLess(self.ladder.pressure(), 0.3)
        self.assertIs(self.ladder.pick("что нового?"), Rung.THINK)

    def test_lag_pushes_to_skill(self):
        self.ctx.lag_ms = self.ctx.genome.pain_lag_ms * 1.5
        self.assertIs(self.ladder.pick("что нового?"), Rung.SKILL)
        self.ctx.lag_ms = self.ctx.genome.pain_lag_ms * 3
        self.assertIs(self.ladder.pick("что нового?"), Rung.SKILL)
        self.assertGreaterEqual(self.ladder.pressure(), 0.7)

    def test_misses_push_to_skill(self):
        self.ctx.stats["просрочено"] += 1
        self.assertIs(self.ladder.pick("что нового?"), Rung.SKILL)

    def test_tact_growth_adds_pressure(self):
        self.ctx.stem.request_tact(self.ctx.genome.tact_s * 4)
        self.assertGreaterEqual(self.ladder.pressure(), 0.3)

    def test_pain_series_drops_think_share_then_recovers(self):
        self.series(10)
        self.assertEqual(self.ladder.share()[Rung.THINK], 1.0)
        self.ctx.lag_ms = self.ctx.genome.pain_lag_ms * 2
        for i in range(50):
            self.ctx.stats["просрочено"] += 1
            self.clock.t += 0.1
            self.ladder.pick(f"боль {i}")
        share = self.ladder.share()
        self.assertLess(share[Rung.THINK], 0.2)
        self.assertGreater(share[Rung.SKILL], 0.8)
        self.ctx.lag_ms = 0.0
        self.clock.t += 120.0
        self.assertIs(self.ladder.pick("тишина"), Rung.THINK)
        self.series(49, "покой")
        self.assertEqual(self.ladder.share()[Rung.THINK], 1.0)

    def test_pressure_decays_without_pain(self):
        self.ctx.lag_ms = self.ctx.genome.pain_lag_ms * 2
        high = self.ladder.pressure()
        self.ctx.lag_ms = 0.0
        self.clock.t += 5.0
        mid = self.ladder.pressure()
        self.clock.t += 60.0
        low = self.ladder.pressure()
        self.assertGreater(high, mid)
        self.assertGreater(mid, low)
        self.assertLess(low, 0.01)

    def test_reflex_wins_under_any_pain(self):
        self.ctx.reflexes.reflexes["привет"] = ("echo привет",)
        self.ctx.lag_ms = self.ctx.genome.pain_lag_ms * 5
        self.ctx.stats["просрочено"] += 10
        self.assertIs(self.ladder.pick("  Привет "), Rung.REFLEX)
        self.assertIs(self.ladder.pick("другое"), Rung.SKILL)

    def test_pressure_bounded_and_monotone_in_lag(self):
        pain = self.ctx.genome.pain_lag_ms
        prev = 0.0
        for k in range(0, 61):
            self.ctx.lag_ms = pain * k / 10
            p = self.ladder.pressure()
            self.assertGreaterEqual(p, prev)
            self.assertLessEqual(p, 1.0)
            prev = p
        self.assertGreaterEqual(prev, 0.9)
        for k in (0, 5, 10, 20, 60):
            fresh = Ladder(self.ctx)
            self.ctx.lag_ms = pain * k / 10
            self.assertTrue(0.0 <= fresh.pressure() <= 1.0)

    def test_pressure_capped_by_misses(self):
        self.ctx.stats["просрочено"] += 1000
        self.assertEqual(self.ladder.pressure(), 1.0)

    def test_budgets_and_hints(self):
        self.assertEqual(self.ladder.max_tokens(Rung.THINK), 256)
        self.assertEqual(self.ladder.max_tokens(Rung.SKILL), 64)
        self.assertEqual(self.ladder.max_tokens(Rung.REFLEX), 0)
        self.assertIn("20 слов", self.ladder.system_hint(Rung.SKILL))
        self.assertEqual(self.ladder.system_hint(Rung.THINK), "")
        self.assertEqual(self.ladder.system_hint(Rung.REFLEX), "")

    def test_share_sums_to_one_over_window(self):
        self.ctx.reflexes.reflexes["привет"] = ("echo привет",)
        self.assertEqual(sum(self.ladder.share().values()), 0.0)
        self.series(30)
        self.ladder.pick("привет")
        self.ctx.lag_ms = self.ctx.genome.pain_lag_ms * 2
        self.series(40, "боль")
        share = self.ladder.share()
        self.assertEqual(len(self.ladder.recent), 50)
        self.assertAlmostEqual(sum(share.values()), 1.0)
        self.assertAlmostEqual(share[Rung.REFLEX], 0.02)

    def test_works_without_reflex_cache(self):
        self.ctx.reflexes = None
        self.assertIs(self.ladder.pick("что нового?"), Rung.THINK)


if __name__ == "__main__":
    unittest.main()
