import tempfile
import unittest
from pathlib import Path

from organism import generations as gen
from organism import holdout
from organism.config import Genome
from organism.llm import FakeLLM
import random


class HoldoutTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_add_dedupes_and_limits_sample(self):
        self.assertTrue(holdout.add(self.state, " привет "))
        self.assertFalse(holdout.add(self.state, "привет"))
        self.assertFalse(holdout.add(self.state, "  "))
        for i in range(12):
            holdout.add(self.state, f"фраза {i}")
        self.assertEqual(len(holdout.load(self.state)), 13)
        self.assertEqual(holdout.sample(self.state)[-1], "фраза 11")
        self.assertEqual(len(holdout.sample(self.state)), holdout.LIMIT)

    def test_empty_when_no_file(self):
        self.assertEqual(holdout.load(self.state), [])


class HoldoutBreedTest(unittest.IsolatedAsyncioTestCase):
    async def test_kid_worse_on_holdout_is_dropped(self):
        scores = iter([1.0, 2.0, 1.5, 1.0, 0.5, 1.5])
        calls = []

        async def fake_trial(g, texts, llm, tmp):
            calls.append(list(texts))
            return next(scores)

        orig = gen.trial
        gen.trial = fake_trial
        try:
            base, kids = await gen.breed(Genome(), ["а"], FakeLLM(), random.Random(1), n=2,
                                         max_children=2, margin=0.03, holdout=["э"])
        finally:
            gen.trial = orig
        self.assertEqual(base, 1.0)
        self.assertEqual(len(kids), 1)
        self.assertEqual(kids[0][0], 1.5)
        self.assertEqual(calls[-3:], [["э"], ["э"], ["э"]])
