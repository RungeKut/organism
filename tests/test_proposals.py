import json
import tempfile
import unittest
from pathlib import Path

from organism import proposals as pr


class ProposalsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def propose(self, **kw):
        args = dict(kind="порог", target="territory.json", before="0.5", after="0.4",
                    parent_score=1.0, child_score=1.2, reason="тест")
        args.update(kw)
        return pr.propose(self.state, **args)

    def read(self, path) -> dict:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def test_propose_creates_file_with_fields(self):
        path = self.propose()
        self.assertIsNotNone(path)
        self.assertEqual(path.parent, self.state / "proposals")
        data = self.read(path)
        self.assertEqual(len(data["id"]), 8)
        int(data["id"], 16)
        self.assertEqual(len(data["token"]), 32)
        self.assertIsInstance(data["t"], float)
        self.assertEqual(data["kind"], "порог")
        self.assertEqual(data["target"], "territory.json")
        self.assertEqual(data["before"], "0.5")
        self.assertEqual(data["after"], "0.4")
        self.assertEqual(data["parent_score"], 1.0)
        self.assertEqual(data["child_score"], 1.2)
        self.assertEqual(data["reason"], "тест")

    def test_pending_by_time_keeps_token(self):
        first = self.propose(after="0.4")
        second = self.propose(kind="модель", after="q6")
        data = self.read(first)
        data["t"] = self.read(second)["t"] + 10.0
        first.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        rows = pr.pending(self.state)
        self.assertEqual([r["id"] for r in rows], [self.read(second)["id"], data["id"]])
        self.assertIn("token", rows[0])

    def test_duplicate_pending_returns_none(self):
        self.assertIsNotNone(self.propose())
        self.assertIsNone(self.propose())
        self.assertEqual(len(list((self.state / "proposals").glob("*.json"))), 1)

    def test_repeat_after_rejection_returns_none(self):
        path = self.propose()
        data = self.read(path)
        self.assertTrue(pr.decide(self.state, data["id"], data["token"], accept=False, reason="нет"))
        self.assertIsNone(self.propose())
        self.assertTrue(pr.rejected(self.state, "порог", "territory.json", "0.4"))
        self.assertFalse(pr.rejected(self.state, "порог", "territory.json", "0.6"))

    def test_repeat_after_accept_returns_none(self):
        path = self.propose()
        data = self.read(path)
        self.assertTrue(pr.decide(self.state, data["id"], data["token"], accept=True, reason="ок"))
        self.assertIsNone(self.propose())

    def test_wrong_tokens_change_nothing(self):
        path = self.propose()
        data = self.read(path)
        raw = path.read_bytes()
        for bad in ("0" * 32, "", "неверный", "токен-кириллица", None):
            self.assertFalse(pr.decide(self.state, data["id"], bad, accept=True))
        self.assertFalse(pr.decide(self.state, "неттакого", data["token"], accept=True))
        self.assertEqual(path.read_bytes(), raw)
        self.assertEqual(pr.decisions(self.state), [])
        self.assertFalse((self.state / "decisions.jsonl").exists())

    def test_decisions_written_without_token(self):
        first = self.propose()
        second = self.propose(kind="модель", target="модель.json", after="q6")
        d1, d2 = self.read(first), self.read(second)
        self.assertTrue(pr.decide(self.state, d1["id"], d1["token"], accept=True, reason="ок"))
        self.assertFalse(first.exists())
        self.assertTrue(pr.decide(self.state, d2["id"], d2["token"], accept=False))
        self.assertFalse(second.exists())
        rows = pr.decisions(self.state)
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertNotIn("token", row)
        self.assertEqual((rows[0]["id"], rows[0]["accept"], rows[0]["reason"]), (d1["id"], True, "ок"))
        self.assertEqual((rows[1]["id"], rows[1]["accept"], rows[1]["reason"]), (d2["id"], False, ""))
        self.assertEqual(rows[0]["kind"], "порог")
        self.assertEqual(rows[0]["target"], "territory.json")
        self.assertEqual(rows[0]["after"], "0.4")
        self.assertEqual(rows[0]["parent_score"], 1.0)
        self.assertEqual(rows[0]["child_score"], 1.2)
        text = (self.state / "decisions.jsonl").read_text(encoding="utf-8")
        self.assertIn("порог", text)
        self.assertEqual(len(text.strip().splitlines()), 2)
        self.assertFalse(pr.decide(self.state, d1["id"], d1["token"], accept=True))


if __name__ == "__main__":
    unittest.main()
