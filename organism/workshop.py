from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path, PurePosixPath, PureWindowsPath

from .config import ROOT

PROTECTED = ("organism/stem.py", "config/territory.json", "organism/audit.py")
EXCLUDE = (".git", ".venv", "state", "result", "outbox", ".relay", ".zcode", "__pycache__")
RAN = re.compile(r"Ran (\d+) tests?")


def patch_files(patch: str) -> list[str]:
    out: list[str] = []
    for line in patch.splitlines():
        m = re.match(r"diff --git a/(.*) b/(.*)$", line)
        if m:
            for path in (m.group(1), m.group(2)):
                if path not in out:
                    out.append(path)
            continue
        m = re.match(r"--- a/(.*)$", line) or re.match(r"\+\+\+ b/(.*)$", line)
        if m and m.group(1) != "/dev/null" and m.group(1) not in out:
            out.append(m.group(1))
    return out


def _bad_path(path: str) -> bool:
    return (PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts
            or PureWindowsPath(path).is_absolute())


def problems(patch: str) -> list[str]:
    if not patch.strip():
        return ["патч пуст"]
    out: list[str] = []
    if "GIT binary patch" in patch:
        out.append("бинарный патч не поддерживается")
    files = patch_files(patch)
    for path in files:
        if _bad_path(path):
            out.append(f"недопустимый путь: {path}")
        elif path in PROTECTED:
            out.append(f"охраняемый файл: {path}")
    checks = [f for f in files if f.startswith("tests/") or f == "organism/audit.py"]
    code = [f for f in files if f.startswith("organism/") and f != "organism/audit.py"]
    if checks and code:
        out.append("правки organism/ и проверок (tests/, audit.py) в одном патче")
    return out


def approve_source(root: Path) -> str:
    path = Path(root) / "organism" / "generations.py"
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError, ValueError):
        return ""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "approve":
            return ast.get_source_segment(source, node) or ""
    return ""


def _manifest(copy: Path) -> dict[str, str]:
    manifest = {}
    for name in PROTECTED:
        path = copy / name
        manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else hashlib.sha256(b"").hexdigest()
    manifest["approve"] = hashlib.sha256(approve_source(copy).encode("utf-8")).hexdigest()
    return manifest


def _execute(cmd: list[str], cwd: Path, env: dict, timeout_s: float) -> tuple[str, bool, bool]:
    try:
        res = subprocess.run(cmd, cwd=cwd, capture_output=True, timeout=timeout_s, env=env)
    except subprocess.TimeoutExpired as e:
        raw = (e.stdout or b"") + (e.stderr or b"")
        return raw.decode("utf-8", "replace"), False, True
    raw = (res.stdout or b"") + (res.stderr or b"")
    return raw.decode("utf-8", "replace"), res.returncode == 0, False


def run(patch: str, root: Path = ROOT, outbox: Path = ROOT / "outbox", timeout_s: float = 900,
        tests_cmd: list[str] | None = None, demo_cmd: list[str] | None = None) -> dict:
    t0 = time.time()
    root, outbox = Path(root), Path(outbox)
    tests_cmd = list(tests_cmd) if tests_cmd else [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."]
    demo_cmd = list(demo_cmd) if demo_cmd else [sys.executable, "-m", "organism", "--demo"]
    report = {"id": secrets.token_hex(4), "t": t0, "files": patch_files(patch), "problems": problems(patch),
              "applied": False, "tests_ok": False, "tests_ran": 0, "tests_tail": "", "demo_ok": False, "seconds": 0.0}
    tmp = None
    try:
        if not report["problems"]:
            tmp = Path(tempfile.mkdtemp(prefix="workshop-"))
            copy = tmp / "repo"
            shutil.copytree(root, copy, ignore=shutil.ignore_patterns(*EXCLUDE))
            before = _manifest(copy)
            env = {**os.environ, "PYTHONUTF8": "1"}
            try:
                check = subprocess.run(["git", "apply", "--check", "-"], input=patch.encode("utf-8"), cwd=copy,
                                       capture_output=True, timeout=timeout_s, env=env)
                if check.returncode == 0:
                    applied = subprocess.run(["git", "apply", "-"], input=patch.encode("utf-8"), cwd=copy,
                                             capture_output=True, timeout=timeout_s, env=env)
                    report["applied"] = applied.returncode == 0
            except subprocess.TimeoutExpired:
                report["problems"].append("тайм-аут")
            if report["applied"]:
                if _manifest(copy) != before:
                    report["problems"].append("тронута охраняемая часть")
                out, ok, expired = _execute(tests_cmd, copy, env, timeout_s)
                if expired:
                    report["problems"].append("тайм-аут")
                report["tests_ok"] = ok
                ran = RAN.search(out)
                report["tests_ran"] = int(ran.group(1)) if ran else 0
                report["tests_tail"] = "\n".join(out.splitlines()[-40:])
                _, ok, expired = _execute(demo_cmd, copy, env, timeout_s)
                if expired:
                    report["problems"].append("тайм-аут")
                report["demo_ok"] = ok
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    report["seconds"] = round(time.time() - t0, 1)
    box = outbox / report["id"]
    box.mkdir(parents=True, exist_ok=True)
    (box / "patch.diff").write_bytes(patch.encode("utf-8"))
    (box / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
