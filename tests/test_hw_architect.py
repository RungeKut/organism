import asyncio
import unittest

from organism.llm import FakeLLM
from scripts import hw_architect


class HwArchitectTest(unittest.TestCase):
    def test_report_shape_on_fake_llm(self):
        report = asyncio.run(hw_architect.run("", 2, FakeLLM(delay_s=0)))
        self.assertEqual(len(report["раунды"]), 2)
        for r in report["раунды"]:
            self.assertEqual([t for t, _ in r["события"]], ["архитектор_без_предложения"])
        self.assertEqual(report["предложения"], [])
