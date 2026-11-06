import json
import sys
import tempfile
import unittest
from pathlib import Path

from organism import workshop

GENERATIONS = 'def approve(token):\n    return token == "ok"\n'
STEM = "PHASE = 'бодрствование'\n"
TESTS_X = 'print("раз")\n'

PATCH_EXTRA = """diff --git a/organism/extra.py b/organism/extra.py
new file mode 100644
--- /dev/null
+++ b/organism/extra.py
@@ -0,0 +1 @@
+print("добавлен")
"""

PATCH_BAD_CTX = """diff --git a/tests/x.py b/tests/x.py
--- a/tests/x.py
+++ b/tests/x.py
@@ -1 +1,2 @@
 print("x")
+print("y")
"""

PATCH_APPROVE = """diff --git a/organism/generations.py b/organism/generations.py
--- a/organism/generations.py
+++ b/organism/generations.py
@@ -1,2 +1,2 @@
 def approve(token):
-    return token == "ok"
+    return True
"""

PATCH_MIXED = """diff --git a/tests/x.py b/tests/x.py
--- a/tests/x.py
+++ b/tests/x.py
@@ -1 +1 @@
-print("раз")
+print("два")
diff --git a/organism/extra.py b/organism/extra.py
--- /dev/null
+++ b/organism/extra.py
@@ -0,0 +1 @@
+print("добавлен")
"""

PATCH_STEM = """diff --git a/organism/stem.py b/organism/stem.py
--- a/organism/stem.py
+++ b/organism/stem.py
@@ -1 +1 @@
-PHASE = 'бодрствование'
+PHASE = 'сон'
"""

PATCH_DOTDOT = """diff --git a/../evil.py b/../evil.py
--- /dev/null
+++ b/../evil.py
@@ -0,0 +1 @@
+evil
"""


class ProblemsTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(workshop.problems(""), ["патч пуст"])
        self.assertEqual(workshop.problems("   \n"), ["патч пуст"])

    def test_protected(self):
        self.assertIn("охраняемый файл: organism/stem.py", workshop.problems(PATCH_STEM))

    def test_dotdot(self):
        self.assertIn("недопустимый путь: ../evil.py", workshop.problems(PATCH_DOTDOT))

    def test_binary(self):
        self.assertIn("бинарный патч не поддерживается", workshop.problems(PATCH_STEM + "\nGIT binary patch\n"))

    def test_mixed_rule(self):
        self.assertIn("правки organism/ и проверок (tests/, audit.py) в одном патче", workshop.problems(PATCH_MIXED))

    def test_clean_pass(self):
        self.assertEqual(workshop.problems(PATCH_EXTRA), [])
        self.assertEqual(workshop.problems(PATCH_BAD_CTX), [])

    def test_patch_files_order_unique(self):
        self.assertEqual(workshop.patch_files(PATCH_MIXED), ["tests/x.py", "organism/extra.py"])
        self.assertEqual(workshop.patch_files(PATCH_EXTRA), ["organism/extra.py"])

    def test_approve_source(self):
        with tempfile.TemporaryDirectory() as d:
            org = Path(d) / "organism"
            org.mkdir()
            (org / "generations.py").write_bytes(GENERATIONS.encode("utf-8"))
            src = workshop.approve_source(Path(d))
            self.assertTrue(src.startswith("def approve"))
            self.assertIn('return token == "ok"', src)
            (org / "generations.py").write_bytes("нет функции\n".encode("utf-8"))
            self.assertEqual(workshop.approve_source(Path(d)), "")


class RunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rep = Path(self.tmp.name) / "rep"
        (self.rep / "organism").mkdir(parents=True)
        (self.rep / "tests").mkdir()
        (self.rep / "organism" / "generations.py").write_bytes(GENERATIONS.encode("utf-8"))
        (self.rep / "organism" / "stem.py").write_bytes(STEM.encode("utf-8"))
        (self.rep / "tests" / "x.py").write_bytes(TESTS_X.encode("utf-8"))
        self.outbox = Path(self.tmp.name) / "outbox"

    def tearDown(self):
        self.tmp.cleanup()

    def tree(self, path: Path) -> dict:
        return {str(p.relative_to(path)): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()}

    def run_patch(self, patch, tests_cmd=None, demo_cmd=None) -> dict:
        return workshop.run(patch, root=self.rep, outbox=self.outbox,
                            tests_cmd=tests_cmd or [sys.executable, "-c", "print('Ran 3 tests')"],
                            demo_cmd=demo_cmd or [sys.executable, "-c", "print('демо ок')"])

    def test_good_patch_applies_in_copy(self):
        before = self.tree(self.rep)
        r = self.run_patch(PATCH_EXTRA)
        self.assertEqual(r["problems"], [])
        self.assertTrue(r["applied"])
        self.assertTrue(r["tests_ok"])
        self.assertEqual(r["tests_ran"], 3)
        self.assertIn("Ran 3 tests", r["tests_tail"])
        self.assertTrue(r["demo_ok"])
        self.assertLess(r["seconds"], 100)
        self.assertEqual(self.tree(self.rep), before)
        self.assertFalse((self.rep / "organism" / "extra.py").exists())
        box = self.outbox / r["id"]
        self.assertEqual((box / "patch.diff").read_bytes(), PATCH_EXTRA.encode("utf-8"))
        saved = json.loads((box / "report.json").read_text(encoding="utf-8"))
        self.assertEqual((saved["id"], saved["applied"], saved["tests_ok"]), (r["id"], True, True))

    def test_unappliable_patch(self):
        before = self.tree(self.rep)
        r = self.run_patch(PATCH_BAD_CTX)
        self.assertEqual(r["problems"], [])
        self.assertFalse(r["applied"])
        self.assertFalse(r["tests_ok"])
        self.assertEqual(r["tests_ran"], 0)
        self.assertEqual(self.tree(self.rep), before)

    def test_approve_patch_flagged(self):
        r = self.run_patch(PATCH_APPROVE)
        self.assertTrue(r["applied"])
        self.assertIn("тронута охраняемая часть", r["problems"])

    def test_failing_tests(self):
        r = self.run_patch(PATCH_EXTRA, tests_cmd=[sys.executable, "-c", "import sys; sys.exit(3)"])
        self.assertTrue(r["applied"])
        self.assertFalse(r["tests_ok"])
        self.assertEqual(r["tests_ran"], 0)
        self.assertTrue(r["demo_ok"])

    def test_problem_patch_no_copy(self):
        before = self.tree(self.rep)
        r = self.run_patch(PATCH_STEM)
        self.assertIn("охраняемый файл: organism/stem.py", r["problems"])
        self.assertFalse(r["applied"])
        self.assertFalse(r["tests_ok"])
        self.assertEqual(self.tree(self.rep), before)
        self.assertTrue((self.outbox / r["id"] / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
