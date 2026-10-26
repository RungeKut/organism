from __future__ import annotations

import contextlib
import io
import json
import time
import unittest
import urllib.request
from dataclasses import replace
from unittest import mock

from organism.config import ROOT, Territory
from organism.llm import AIMD, LlamaCpp, LLMError
from scripts import llama_server
from tests.openai_stub import Stub

MSG = [{"role": "user", "content": "привет"}]


class LlamaCppHttpTest(unittest.IsolatedAsyncioTestCase):
    def stub(self, **kw) -> Stub:
        s = Stub(**kw)
        self.addCleanup(s.close)
        return s

    async def test_reply(self):
        s = self.stub()
        llm = LlamaCpp(s.url, AIMD(4))
        self.assertEqual(await llm.chat(MSG, max_tokens=7), "ответ заглушки")
        self.assertEqual(s.requests[0]["path"], "/v1/chat/completions")
        self.assertEqual(s.requests[0]["body"]["messages"], MSG)
        self.assertEqual(s.requests[0]["body"]["max_tokens"], 7)

    async def test_health(self):
        s = self.stub()
        with urllib.request.urlopen(s.url + "/health", timeout=2) as resp:
            self.assertEqual(json.load(resp)["status"], "ok")

    async def test_timings_to_last_tps(self):
        s = self.stub(tps=37.25)
        llm = LlamaCpp(s.url, AIMD(4))
        self.assertEqual(llm.last_tps, 0.0)
        await llm.chat(MSG)
        self.assertEqual(llm.last_tps, 37.25)

    async def test_deadline(self):
        s = self.stub(delay_s=1.5)
        slots = AIMD(4)
        llm = LlamaCpp(s.url, slots)
        t0 = time.monotonic()
        with self.assertRaises(LLMError):
            await llm.chat(MSG, deadline_s=0.2)
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertEqual(slots.active, 0)

    async def test_http_500_halves_limit(self):
        s = self.stub(fail=True)
        slots = AIMD(8, start=8)
        llm = LlamaCpp(s.url, slots)
        with self.assertRaises(LLMError):
            await llm.chat(MSG)
        self.assertEqual(slots.limit, 4)
        self.assertEqual(slots.active, 0)

    async def test_lora_in_body(self):
        s = self.stub()
        llm = LlamaCpp(s.url, AIMD(4))
        lora = [{"id": 0, "scale": 0.5}]
        await llm.chat(MSG, lora=lora)
        await llm.chat(MSG)
        self.assertEqual(s.requests[0]["body"]["lora"], lora)
        self.assertNotIn("lora", s.requests[1]["body"])

    async def test_foreign_host_denied(self):
        with self.assertRaises(PermissionError):
            LlamaCpp("http://10.0.0.5:8080", AIMD(4))
        with self.assertRaises(PermissionError):
            LlamaCpp("http://example.com", AIMD(4), allow_network=False)
        self.assertEqual(LlamaCpp("http://10.0.0.5:8080/", AIMD(4), allow_network=True).url, "http://10.0.0.5:8080")


class LlamaServerScriptTest(unittest.TestCase):
    def test_build_cmd(self):
        t = replace(Territory(), max_llm_slots=3, llm_url="http://127.0.0.1:9090")
        cmd = llama_server.build_cmd("m.gguf", t, ("a.gguf", "b.gguf"))
        self.assertEqual(cmd[0], "llama-server")
        self.assertEqual(cmd[cmd.index("-m") + 1], "m.gguf")
        self.assertEqual(cmd[cmd.index("-np") + 1], "3")
        self.assertEqual(cmd[cmd.index("--port") + 1], "9090")
        self.assertIn("--lora-init-without-apply", cmd)
        self.assertEqual([cmd[i + 1] for i, x in enumerate(cmd) if x == "--lora"], ["a.gguf", "b.gguf"])

    def test_dry_run_does_not_spawn(self):
        out = io.StringIO()
        with mock.patch("subprocess.call") as call, mock.patch("subprocess.Popen") as popen, \
                contextlib.redirect_stdout(out):
            code = llama_server.main(["m.gguf", "--dry-run", "--territory", str(ROOT / "config" / "territory.json")])
        self.assertEqual(code, 0)
        call.assert_not_called()
        popen.assert_not_called()
        self.assertIn("-np 4", out.getvalue())
        self.assertIn("--lora-init-without-apply", out.getvalue())


if __name__ == "__main__":
    unittest.main()
