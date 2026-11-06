from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from organism import workshop


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Проверка патча к коду организма в копии репозитория")
    ap.add_argument("patch", type=Path, help="файл с патчем")
    ap.add_argument("--outbox", type=Path, default=ROOT / "outbox")
    ap.add_argument("--timeout", type=float, default=900.0)
    args = ap.parse_args(argv)
    report = workshop.run(args.patch.read_text(encoding="utf-8"), outbox=args.outbox, timeout_s=args.timeout)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["problems"] == [] and report["applied"] and report["tests_ok"] and report["demo_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
