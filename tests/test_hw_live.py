from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from organism.config import Territory
from scripts import hw_live, llama_server
from tests.openai_stub import Stub


class HwLiveTest(unittest.IsolatedAsyncioTestCase):
    async def test_run_reports_speed_and_slots(self):
        stub = Stub(tps=41.5)
        self.addCleanup(stub.close)
        report = await hw_live.run(stub.url, phrases=("привет", "как дела"), parallel=2)
        self.assertEqual(len(report["по_очереди"]), 2)
        self.assertEqual(report["пачка"]["запросов"], 2)
        self.assertEqual(report["last_tps"], 41.5)
        self.assertGreaterEqual(report["слоты"]["предел"], 1)
        self.assertIn("lag_ms", report)
        json.dumps(report, ensure_ascii=False)

    def test_main_without_server(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            code = hw_live.main(["--url", "http://127.0.0.1:9"])
        self.assertEqual(code, 2)
        self.assertIn("не отвечает", err.getvalue())

    def test_main_writes_out(self):
        stub = Stub()
        self.addCleanup(stub.close)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "live.json"
            with contextlib.redirect_stdout(io.StringIO()):
                code = hw_live.main(["--url", stub.url, "--parallel", "1", "--out", str(out)])
            self.assertEqual(code, 0)
            self.assertIn("по_очереди", json.loads(out.read_text(encoding="utf-8")))


class ThreadsFlagTest(unittest.TestCase):
    def test_threads_in_cmd(self):
        t = replace(Territory(), max_llm_slots=2)
        self.assertNotIn("-t", llama_server.build_cmd("m.gguf", t))
        cmd = llama_server.build_cmd("m.gguf", t, threads=16)
        self.assertEqual(cmd[cmd.index("-t") + 1], "16")
