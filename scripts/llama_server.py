from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from organism.config import Territory


def build_cmd(model: str, territory: Territory, lora=(), binary: str = "llama-server") -> list[str]:
    parsed = urlparse(territory.llm_url)
    cmd = [binary, "-m", str(model), "-np", str(territory.max_llm_slots),
           "--host", parsed.hostname or "127.0.0.1", "--port", str(parsed.port or 8080),
           "--lora-init-without-apply"]
    for path in lora:
        cmd += ["--lora", str(path)]
    return cmd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Команда запуска llama-server по территории")
    ap.add_argument("model")
    ap.add_argument("--territory", default=str(ROOT / "config" / "territory.json"))
    ap.add_argument("--lora", action="append", default=[])
    ap.add_argument("--binary", default="llama-server")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    cmd = build_cmd(args.model, Territory.load(Path(args.territory)), args.lora, args.binary)
    if args.dry_run:
        print(shlex.join(cmd))
        return 0
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
