from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from organism.config import Genome
from scripts import hw_soak
from tests.openai_stub import Stub


class HwSoakTest(unittest.IsolatedAsyncioTestCase):
    async def test_soak_samples_and_sleeps(self):
        stub = Stub()
        self.addCleanup(stub.close)
        genome = Genome(tact_s=0.05, wake_s=1.0, sleep_s=0.5, drowsy_max_s=1.0)
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "soak.jsonl"
            report = await hw_soak.soak(stub.url, Path(tmp) / "state", 6 / 3600, 0.3, 1, log, genome)
            rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        self.assertGreaterEqual(report["снов"], 1)
        self.assertEqual(len(report["сны"]), report["снов"])
        self.assertEqual(report["вопросов"] + report["отклонено"], len(rows))
        self.assertGreater(report["rss_mb"]["макс"], 0)
        self.assertTrue(report["геном_прежний"])
        self.assertNotIn("token", json.dumps(report, ensure_ascii=False))
        self.assertIn("state_mb", rows[-1])
        self.assertEqual(report["сбои"], [])
        self.assertEqual(len(report["аудит"]), report["нарушений"])
        self.assertTrue(report["разведения"])
        self.assertEqual(report["поколение_ждёт"], any(b["итог"] == "поколение_предложено"
                                                       for b in report["разведения"]))

    def test_main_without_server(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(hw_soak.main(["--url", "http://127.0.0.1:9", "--state", tmp]), 2)


if __name__ == "__main__":
    unittest.main()
