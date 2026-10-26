from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from organism.calibrate import knee
from organism.config import Territory
from scripts.llama_server import build_cmd

REPO_URL = "https://github.com/ggerganov/llama.cpp"
NUMA_MODES = ("none", "node0", "interleave")
THREADS = (8, 16, 24, 32, 48)
THREAD_KEYS = ("n_threads", "threads", "n_thread", "nthreads")
TPS_KEYS = ("avg_ts", "avg_tps", "tps", "t/s", "ts", "tokens_per_second")
PROMPT_KEYS = ("n_prompt", "prompt", "pp")
GEN_KEYS = ("n_gen", "gen", "tg")
UNKNOWN = "?"


@dataclass(frozen=True)
class BenchRun:
    name: str
    cmd: list[str]


def build_commands(src_dir, build_dir, jobs: int, exists: bool | None = None) -> list[list[str]]:
    src, build = str(src_dir), str(build_dir)
    if exists is None:
        exists = (Path(src) / ".git").exists()
    fetch = ["git", "-C", src, "pull", "--ff-only"] if exists else ["git", "clone", "--depth", "1", REPO_URL, src]
    configure = ["cmake", "-S", src, "-B", build,
                 "-DGGML_NATIVE=OFF", "-DGGML_AVX2=ON", "-DGGML_FMA=ON", "-DGGML_F16C=ON", "-DGGML_AVX512=OFF",
                 "-DCMAKE_C_FLAGS=-march=znver2", "-DCMAKE_CXX_FLAGS=-march=znver2",
                 "-DCMAKE_BUILD_TYPE=Release"]
    compile_ = ["cmake", "--build", build, "--config", "Release", "-j", str(max(1, int(jobs))),
                "--target", "llama-bench", "llama-server"]
    return [fetch, configure, compile_]


def numa_wrap(numa: str) -> list[str]:
    if numa == "node0":
        return ["numactl", "--cpunodebind=0", "--membind=0"]
    if numa == "interleave":
        return ["numactl", "--interleave=all"]
    return []


def numa_args(numa: str) -> list[str]:
    if numa == "node0":
        return ["--numa", "numactl"]
    if numa == "interleave":
        return ["--numa", "distribute"]
    return []


def bench_matrix(model, threads=THREADS, numa=NUMA_MODES, binary: str = "llama-bench") -> list[BenchRun]:
    runs = []
    for mode in numa:
        for n in threads:
            cmd = numa_wrap(mode) + [binary, "-m", str(model), "-t", str(n), "-p", "128", "-n", "64",
                                     "-r", "3", "-o", "json"] + numa_args(mode)
            runs.append(BenchRun(f"{mode}-t{n}", cmd))
    return runs


def _num(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) and x >= 0 else None


def _first(data: dict, keys):
    for key in keys:
        if key in data:
            return data[key]
    return None


def _scan(text: str) -> list[tuple[str | None, object]]:
    decoder = json.JSONDecoder()
    found, name, pos, size = [], None, 0, len(text)
    while pos < size:
        ch = text[pos]
        if ch == "#":
            end = text.find("\n", pos)
            end = size if end < 0 else end
            label = text[pos:end].lstrip("# ").strip()
            if label:
                name = label
            pos = end
        elif ch in "[{":
            try:
                obj, end = decoder.raw_decode(text, pos)
            except ValueError:
                pos += 1
                continue
            found.append((name, obj))
            pos = end
        else:
            pos += 1
    return found


def _flatten(obj) -> list[dict]:
    if isinstance(obj, list):
        return [d for item in obj for d in _flatten(item)]
    if isinstance(obj, dict):
        nested = [d for v in obj.values() if isinstance(v, list) for d in _flatten(v)]
        return nested if nested and _first(obj, THREAD_KEYS) is None else [obj]
    return []


def _kind(item: dict) -> str | None:
    test = item.get("test")
    if isinstance(test, str):
        low = test.strip().lower()
        if low.startswith("pp"):
            return "pp"
        if low.startswith("tg"):
            return "tg"
    prompt, gen = _num(_first(item, PROMPT_KEYS)), _num(_first(item, GEN_KEYS))
    if prompt and not gen:
        return "pp"
    if gen and not prompt:
        return "tg"
    return None


def _numa_of(item: dict, name: str | None) -> str:
    declared = item.get("numa")
    if isinstance(declared, str) and declared in NUMA_MODES:
        return declared
    head = (name or "").split("-t")[0]
    return head if head in NUMA_MODES else UNKNOWN


def parse_bench_json(text) -> list[dict]:
    if not isinstance(text, str):
        return []
    merged: dict[tuple[str, int], dict] = {}
    for header, obj in _scan(text):
        for item in _flatten(obj):
            threads = _num(_first(item, THREAD_KEYS))
            tps = _num(_first(item, TPS_KEYS))
            kind = _kind(item)
            if not threads or tps is None or kind is None:
                continue
            threads = int(threads)
            label = header or (item.get("name") if isinstance(item.get("name"), str) else None)
            name = label or f"t{threads}"
            row = merged.setdefault((name, threads), {"name": name, "threads": threads, "pp_tps": 0.0,
                                                      "tg_tps": 0.0, "numa": _numa_of(item, label)})
            row[f"{kind}_tps"] = tps
    return list(merged.values())


def best_config(rows, gain: float = 0.1) -> dict:
    usable = [r for r in rows if isinstance(r, dict) and r.get("tg_tps", 0) > 0 and r.get("threads", 0) > 0]
    if not usable:
        return {"ok": False, "recommendation": "Нет пригодных замеров: в выводе не найдено скоростей генерации."}
    best = max(usable, key=lambda r: r["tg_tps"])
    numa = best.get("numa", UNKNOWN)
    per_numa: dict[str, float] = {}
    by_threads: dict[int, float] = {}
    for r in usable:
        mode = r.get("numa", UNKNOWN)
        per_numa[mode] = max(per_numa.get(mode, 0.0), r["tg_tps"])
        if mode == numa:
            by_threads[r["threads"]] = max(by_threads.get(r["threads"], 0.0), r["tg_tps"])
    points = sorted(by_threads.items())
    knee_threads = knee(points, gain)
    knee_tps = by_threads[knee_threads]
    mode_text = "режим NUMA в замерах не указан" if numa == UNKNOWN else f"NUMA-режим {numa}"
    lines = [f"Максимум генерации: {best['tg_tps']:.2f} ток/с на {best['threads']} потоках, {mode_text}."]
    if knee_threads == best["threads"]:
        if points[-1][0] == knee_threads and len(points) > 1:
            lines.append("Кривая не насыщена: расширить сетку потоков вверх.")
        lines.append(f"Рекомендация: {knee_threads} потоков.")
    else:
        lines.append(f"Излом кривой на {knee_threads} потоках ({knee_tps:.2f} ток/с): дальнейший рост "
                     f"меньше {gain:.0%} на шаг. Рекомендация: {knee_threads} потоков, остальные ядра "
                     f"оставить фону и второму инстансу.")
    if len(per_numa) > 1:
        ranked = ", ".join(f"{m} {v:.2f}" for m, v in sorted(per_numa.items(), key=lambda kv: -kv[1]))
        lines.append(f"Лучшее по режимам NUMA (ток/с): {ranked}.")
    return {"ok": True, "best": dict(best), "numa": numa, "threads": knee_threads, "knee_threads": knee_threads,
            "knee_tps": knee_tps, "curve": points, "per_numa": per_numa, "recommendation": " ".join(lines)}


def server_cmds(model, threads: int, slots: int, numa: str = "node0", bg_threads: int | None = None,
                bg_slots: int = 1, hot_port: int = 8080, bg_port: int = 8081,
                binary: str = "llama-server") -> list[list[str]]:
    hot_mode = numa if numa in NUMA_MODES and numa != "none" else "node0"
    bg_threads = max(1, int(threads) // 4) if bg_threads is None else int(bg_threads)
    hot_terr = Territory(max_llm_slots=max(1, int(slots)), llm_url=f"http://127.0.0.1:{hot_port}")
    bg_terr = Territory(max_llm_slots=max(1, int(bg_slots)), llm_url=f"http://127.0.0.1:{bg_port}")
    hot = numa_wrap(hot_mode) + build_cmd(model, hot_terr, binary=binary) + ["-t", str(int(threads))] + numa_args(hot_mode)
    bg = ["numactl", "--cpunodebind=1"] + build_cmd(model, bg_terr, binary=binary) + ["-t", str(bg_threads)]
    return [hot, bg]


def _line(cmd: list[str]) -> str:
    return shlex.join(cmd)


def plan_text(args) -> str:
    model = args.model or "MODEL.gguf"
    src, build = Path(args.src), Path(args.build)
    out = ["# План (ничего не запускается; --run build или --run bench --model FILE)", "", "## Сборка"]
    out += [_line(c) for c in build_commands(src, build, args.jobs)]
    out += ["", "## Замеры (--run bench пишет вывод в --out с заголовками '### имя')"]
    out += [f"### {r.name}\n{_line(r.cmd)}" for r in bench_matrix(model, args.threads, args.numa,
                                                                  str(build / "bin" / "llama-bench"))]
    out += ["", f"## Разбор результата: python scripts/hw_llama.py --parse {args.out}", "",
            "## Запуск серверов (потоки и режим подставить по итогам разбора)"]
    out += [_line(c) for c in server_cmds(model, args.server_threads, args.slots, args.server_numa,
                                          binary=str(build / "bin" / "llama-server"))]
    return "\n".join(out)


def run_build(args, runner=subprocess.call) -> int:
    for cmd in build_commands(args.src, args.build, args.jobs):
        print("$ " + _line(cmd), flush=True)
        code = runner(cmd)
        if code:
            return code
    return 0


def run_bench(args, runner=subprocess.run) -> int:
    runs = bench_matrix(args.model, args.threads, args.numa, str(Path(args.build) / "bin" / "llama-bench"))
    with open(args.out, "w", encoding="utf-8") as fh:
        for run in runs:
            print("$ " + _line(run.cmd), flush=True)
            res = runner(run.cmd, capture_output=True, text=True)
            fh.write(f"### {run.name}\n{res.stdout}\n")
            fh.flush()
            if res.returncode:
                print(f"Запуск {run.name} завершился кодом {res.returncode}", file=sys.stderr)
    return report(Path(args.out).read_text(encoding="utf-8"))


def report(text: str) -> int:
    rows = parse_bench_json(text)
    result = best_config(rows)
    print(f"Разобрано замеров: {len(rows)}")
    print(result["recommendation"])
    return 0 if result["ok"] else 1


def _ints(value: str) -> tuple[int, ...]:
    return tuple(int(x) for x in value.split(",") if x.strip())


def _modes(value: str) -> tuple[str, ...]:
    modes = tuple(x.strip() for x in value.split(",") if x.strip())
    bad = [m for m in modes if m not in NUMA_MODES]
    if bad:
        raise argparse.ArgumentTypeError(f"неизвестный режим NUMA: {', '.join(bad)}")
    return modes


def make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Сборка llama.cpp под Zen 2 и замеры скорости: по умолчанию печатает план")
    ap.add_argument("--run", choices=("build", "bench"))
    ap.add_argument("--parse", metavar="FILE")
    ap.add_argument("--model")
    ap.add_argument("--src", default="llama.cpp")
    ap.add_argument("--build", default="llama.cpp/build")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 16)
    ap.add_argument("--threads", type=_ints, default=THREADS)
    ap.add_argument("--numa", type=_modes, default=NUMA_MODES)
    ap.add_argument("--out", default="hw_bench.txt")
    ap.add_argument("--server-threads", type=int, default=16)
    ap.add_argument("--server-numa", default="node0", choices=NUMA_MODES)
    ap.add_argument("--slots", type=int, default=4)
    return ap


def main(argv=None) -> int:
    args = make_parser().parse_args(argv)
    if args.parse:
        path = Path(args.parse)
        if not path.is_file():
            print(f"Файл не найден: {path}", file=sys.stderr)
            return 2
        return report(path.read_text(encoding="utf-8", errors="replace"))
    if args.run == "build":
        return run_build(args)
    if args.run == "bench":
        if not args.model or not Path(args.model).is_file():
            print("Для --run bench нужен существующий файл модели: --model FILE", file=sys.stderr)
            return 2
        return run_bench(args)
    print(plan_text(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
