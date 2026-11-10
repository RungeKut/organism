import asyncio
import unittest

from organism.llm import FakeLLM
from scripts import hw_duel


class HwDuelTest(unittest.TestCase):
    def test_report_shape_on_fake_llm(self):
        report = asyncio.run(hw_duel.run("", 2, ["Вопрос один?"], FakeLLM(delay_s=0)))
        self.assertEqual(len(report["дуэли"]), 6)
        self.assertEqual({d["случай"] for d in report["дуэли"]}, {"нулевая", "паритет", "хуже"})
        for d in report["дуэли"]:
            self.assertEqual(d["доля_побед_нового"], 0.5)
            self.assertGreaterEqual(d["сек"], 0)
        self.assertEqual(len(report["судья"]), 2)
        self.assertEqual({row["порядок"] for row in report["судья"]}, {"АБ", "БА"})
        self.assertEqual(report["сводка"]["нулевая"]["среднее"], 0.5)

    def test_external_judge(self):
        report = asyncio.run(hw_duel.run("", 1, ["Вопрос один?"], FakeLLM(delay_s=0), FakeLLM(delay_s=0)))
        self.assertEqual(len(report["дуэли"]), 3)
        self.assertEqual(report["сводка"]["хуже"]["среднее"], 0.5)
