from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from scripts import win_group

WIN = os.name == "nt"


class MaskTest(unittest.TestCase):
    def test_mask_bits(self):
        self.assertEqual(win_group.mask_bits(0), [])
        self.assertEqual(win_group.mask_bits(0b1011), [0, 1, 3])
        self.assertEqual(len(win_group.mask_bits((1 << 64) - 1)), 64)


@unittest.skipUnless(WIN, "только Windows")
class WindowsTest(unittest.TestCase):
    def test_topology(self):
        t = win_group.topology()
        self.assertGreaterEqual(len(t["groups"]), 1)
        self.assertEqual(sum(g["processors"] for g in t["groups"]), t["all_processors"])
        self.assertTrue(t["nodes"])
        self.assertTrue(any(n["available_mb"] for n in t["nodes"]))
        self.assertEqual(sum(n["processors"] for n in t["nodes"]), t["all_processors"])

    def test_whereami(self):
        w = win_group.whereami()
        self.assertEqual(w["pid"], os.getpid())
        self.assertIn(w["group"], range(len(win_group.topology()["groups"])))

    def test_launch_on_node_writes_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, err = Path(tmp) / "o.txt", Path(tmp) / "e.txt"
            code = win_group.main(["--node", "0", "--out", str(out), "--err", str(err), "--",
                                   sys.executable, "-c", "import sys; print(42); sys.exit(3)"])
            self.assertEqual(code, 3)
            self.assertIn("42", out.read_text())

    def test_child_threads_land_on_mask(self):
        code = ("import json, sys, threading; sys.path.insert(0, sys.argv[1]); from scripts import win_group as w; "
                "seen = []; ts = [threading.Thread(target=lambda: seen.extend(w.whereami()['processor'] "
                "for _ in range(50))) for _ in range(4)]; [t.start() for t in ts]; [t.join() for t in ts]; "
                "print(json.dumps({'main': w.whereami(), 'threads': sorted(set(seen))}))")
        root = str(Path(win_group.__file__).resolve().parent.parent)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "w.json"
            rc = win_group.main(["--group", "0", "--mask", "1", "--out", str(out), "--",
                                 getattr(sys, "_base_executable", sys.executable), "-c", code, root])
            self.assertEqual(rc, 0)
            w = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual((w["main"]["group"], w["main"]["processor"]), (0, 0))
            self.assertEqual(w["threads"], [0])

    def test_cli_needs_target(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            win_group.main(["--", "x"])


if __name__ == "__main__":
    unittest.main()
