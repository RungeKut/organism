import itertools
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from organism import will
from organism.journal import Journal


class WillTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.j = Journal(Path(self.tmp.name) / "j.db")
        self.clock = itertools.count(1000)
        patcher = mock.patch("organism.journal.time.time", lambda: float(next(self.clock)))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(self.j.close)

    def act(self, kind, episode="", **extra):
        self.j.action(kind, True, {"episode": episode, **extra})

    def test_empty(self):
        self.assertEqual(will.diversity(self.j, 0), 0.0)
        self.assertEqual(will.loop_score(self.j, 0), 0.0)
        self.assertEqual(will.trend(self.j, 0), 0.0)
        self.assertEqual(will.persistence([], []), 1.0)

    def test_pure_cycle(self):
        for _ in range(4):
            for kind in ("сказать", "сказать", "сказать"):
                self.act(kind)
        self.assertAlmostEqual(will.diversity(self.j, 0), 0.0)
        self.assertAlmostEqual(will.loop_score(self.j, 0), 1.0)

    def test_cycle_of_three_kinds(self):
        for _ in range(4):
            for cmd in ("a", "b", "c"):
                self.act("выполнить", cmd=cmd)
        self.assertAlmostEqual(will.loop_score(self.j, 0), 1.0)
        self.assertAlmostEqual(will.diversity(self.j, 0), 1.0)

    def test_period_two(self):
        for _ in range(5):
            self.act("сказать")
            self.act("выполнить", cmd="ls")
        self.assertAlmostEqual(will.loop_score(self.j, 0), 1.0)

    def test_diverse_sequence(self):
        for cmd in "abcdefghijkl":
            self.act("выполнить", cmd=cmd)
        self.assertAlmostEqual(will.loop_score(self.j, 0), 0.0)
        self.assertAlmostEqual(will.diversity(self.j, 0), 1.0)

    def test_user_episodes_not_self_initiated(self):
        for i in range(5):
            self.act("сказать", episode=f"э{i}")
        self.act("выполнить", cmd="ls")
        self.assertEqual(will.diversity(self.j, 0), 0.0)
        self.act("сказать")
        self.assertGreater(will.diversity(self.j, 0), 0.9)

    def test_t0_filters(self):
        self.act("сказать")
        self.act("выполнить", cmd="ls")
        t0 = next(self.clock)
        self.act("сказать")
        self.assertEqual(will.diversity(self.j, t0), 0.0)
        self.assertGreater(will.diversity(self.j, 0), 0.9)

    def test_trend(self):
        for v in (0.1, 0.2, 0.3, 0.5):
            self.j.reward(v, "цель")
        self.assertGreater(will.trend(self.j, 0), 0)
        t0 = next(self.clock)
        for v in (0.5, 0.2, 0.0, -0.3):
            self.j.reward(v, "человек")
        self.assertLess(will.trend(self.j, t0), 0)

    def test_trend_single_point(self):
        self.j.reward(1.0, "цель")
        self.assertEqual(will.trend(self.j, 0), 0.0)

    def test_trend_equal_timestamps(self):
        self.j.db.executemany("INSERT INTO rewards(t, value, source) VALUES (5.0, ?, 'цель')", [(0.1,), (0.4,), (0.9,)])
        self.assertGreater(will.trend(self.j, 0), 0)

    def test_persistence(self):
        self.assertEqual(will.persistence([], []), 1.0)
        self.assertEqual(will.persistence(["a", "b"], ["c"]), 0.0)
        self.assertEqual(will.persistence(["a", "b"], ["b", "a"]), 1.0)
        self.assertAlmostEqual(will.persistence(["a", "b"], ["b", "c"]), 1 / 3)
        self.assertEqual(will.persistence(["a"], []), 0.0)

    def test_report(self):
        self.j.remember("учиться", "цель")
        self.j.remember("шум", "память")
        t0 = next(self.clock)
        self.j.remember("помнить", "цель")
        for _ in range(6):
            self.act("сказать")
            self.j.reward(0.3, "цель")
        r = will.report(self.j, t0)
        self.assertEqual(set(r), {"diversity", "persistence", "trend", "loop"})
        self.assertAlmostEqual(r["persistence"], 0.5)
        self.assertAlmostEqual(r["loop"], 1.0)
        self.assertAlmostEqual(r["diversity"], 0.0)
        self.assertAlmostEqual(r["trend"], 0.0)


if __name__ == "__main__":
    unittest.main()
