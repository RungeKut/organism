import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts import hw_llama

TG = {
    "node0": {8: 8.0, 16: 14.0, 24: 15.0, 32: 15.2, 48: 14.0},
    "none": {8: 7.0, 16: 11.0, 24: 12.0, 32: 12.5, 48: 12.0},
    "interleave": {8: 7.5, 16: 12.0, 24: 13.5, 32: 14.0, 48: 13.8},
}


def _entries(threads, pp, tg):
    base = {"build_commit": "synthetic", "model_type": "synthetic", "n_threads": threads}
    return [dict(base, n_prompt=128, n_gen=0, test="pp128", avg_ts=pp, stddev_ts=0.5, samples_ts=[pp, pp, pp]),
            dict(base, n_prompt=0, n_gen=64, test="tg64", avg_ts=tg, stddev_ts=0.1, samples_ts=[tg, tg, tg])]


def _sample():
    parts = []
    for mode, table in TG.items():
        for n, tg in table.items():
            parts.append(f"### {mode}-t{n}\nload: warming up\n{json.dumps(_entries(n, tg * 4, tg), indent=2)}\n")
    return "\n".join(parts)


SAMPLE_SYNTHETIC_BENCH = _sample()


class BuildTest(unittest.TestCase):
    def test_flags_and_jobs(self):
        cmds = hw_llama.build_commands("src", "bld", 24, exists=False)
        self.assertEqual(cmds[0][:3], ["git", "clone", "--depth"])
        self.assertIn(hw_llama.REPO_URL, cmds[0])
        flat = cmds[1]
        for flag in ("-DGGML_NATIVE=OFF", "-DGGML_AVX2=ON", "-DGGML_FMA=ON", "-DGGML_F16C=ON",
                     "-DGGML_AVX512=OFF", "-DCMAKE_C_FLAGS=-march=znver2", "-DCMAKE_CXX_FLAGS=-march=znver2",
                     "-DCMAKE_BUILD_TYPE=Release"):
            self.assertIn(flag, flat)
        build = cmds[2]
        self.assertEqual(build[build.index("-j") + 1], "24")

    def test_existing_source_is_pulled(self):
        cmds = hw_llama.build_commands("src", "bld", 4, exists=True)
        self.assertEqual(cmds[0], ["git", "-C", "src", "pull", "--ff-only"])


class MatrixTest(unittest.TestCase):
    def setUp(self):
        self.runs = hw_llama.bench_matrix("m.gguf")

    def test_fifteen_unique_runs(self):
        self.assertEqual(len(self.runs), 15)
        self.assertEqual(len({r.name for r in self.runs}), 15)

    def test_common_args(self):
        for run in self.runs:
            cmd = run.cmd
            self.assertIn("llama-bench", cmd)
            self.assertEqual(cmd[cmd.index("-m") + 1], "m.gguf")
            for flag, value in (("-p", "128"), ("-n", "64"), ("-r", "3"), ("-o", "json")):
                self.assertEqual(cmd[cmd.index(flag) + 1], value)

    def test_numa_wrappers(self):
        by = {r.name: r.cmd for r in self.runs}
        self.assertEqual(by["none-t8"][0], "llama-bench")
        self.assertNotIn("--numa", by["none-t8"])
        self.assertEqual(by["node0-t16"][:3], ["numactl", "--cpunodebind=0", "--membind=0"])
        self.assertEqual(by["interleave-t48"][:2], ["numactl", "--interleave=all"])
        self.assertEqual(by["interleave-t48"][by["interleave-t48"].index("--numa") + 1], "distribute")
        self.assertEqual(by["node0-t16"][by["node0-t16"].index("-t") + 1], "16")


class ParseTest(unittest.TestCase):
    def test_sample_rows(self):
        rows = hw_llama.parse_bench_json(SAMPLE_SYNTHETIC_BENCH)
        self.assertEqual(len(rows), 15)
        row = next(r for r in rows if r["name"] == "node0-t32")
        self.assertEqual((row["threads"], row["numa"]), (32, "node0"))
        self.assertAlmostEqual(row["tg_tps"], 15.2)
        self.assertAlmostEqual(row["pp_tps"], 60.8)

    def test_plain_array_without_header(self):
        rows = hw_llama.parse_bench_json(json.dumps(_entries(16, 50.0, 12.5)))
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["name"], rows[0]["threads"]), ("t16", 16))
        self.assertEqual((rows[0]["pp_tps"], rows[0]["tg_tps"]), (50.0, 12.5))

    def test_alternative_field_names(self):
        text = json.dumps([{"threads": "8", "test": "pp512", "tps": "33.5"},
                           {"threads": 8, "n_prompt": 0, "n_gen": 32, "t/s": 9.5}])
        rows = hw_llama.parse_bench_json(text)
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["pp_tps"], rows[0]["tg_tps"]), (33.5, 9.5))

    def test_wrapped_object(self):
        text = json.dumps({"results": _entries(4, 20.0, 5.0)})
        self.assertEqual(len(hw_llama.parse_bench_json(text)), 1)

    def test_garbage(self):
        for text in ("", "   ", "не json", "[", "[{", "{\"a\": ", None, 42, "[1, 2, 3]", "[{\"n_threads\": \"x\"}]",
                     "[{\"n_threads\": 4, \"test\": \"tg8\", \"avg_ts\": \"nan\"}]"):
            self.assertEqual(hw_llama.parse_bench_json(text), [], text)

    def test_noise_around_json(self):
        text = "warn [info] start\n" + json.dumps(_entries(8, 10.0, 3.0)) + "\n}{ trailing ["
        rows = hw_llama.parse_bench_json(text)
        self.assertEqual([r["tg_tps"] for r in rows], [3.0])


class BestTest(unittest.TestCase):
    def test_max_and_knee(self):
        result = hw_llama.best_config(hw_llama.parse_bench_json(SAMPLE_SYNTHETIC_BENCH))
        self.assertTrue(result["ok"])
        self.assertEqual(result["best"]["name"], "node0-t32")
        self.assertEqual(result["numa"], "node0")
        self.assertEqual(result["knee_threads"], 16)
        self.assertEqual(result["threads"], 16)
        self.assertEqual(result["curve"][0], (8, 8.0))
        self.assertIn("16", result["recommendation"])
        self.assertIn("node0", result["recommendation"])
        self.assertEqual(set(result["per_numa"]), {"node0", "none", "interleave"})

    def test_knee_equals_max(self):
        rows = [{"name": "a", "threads": n, "pp_tps": 1.0, "tg_tps": float(n), "numa": "node0"} for n in (4, 8, 16)]
        result = hw_llama.best_config(rows)
        self.assertEqual(result["threads"], 16)
        self.assertIn("не насыщена", result["recommendation"])

    def test_empty(self):
        result = hw_llama.best_config([])
        self.assertFalse(result["ok"])
        self.assertTrue(result["recommendation"])
        self.assertFalse(hw_llama.best_config([{"threads": 4, "tg_tps": 0.0}])["ok"])


class ServerTest(unittest.TestCase):
    def test_two_instances(self):
        hot, bg = hw_llama.server_cmds("m.gguf", 16, 4, "node0")
        self.assertEqual(hot[:3], ["numactl", "--cpunodebind=0", "--membind=0"])
        self.assertEqual(hot[hot.index("-np") + 1], "4")
        self.assertEqual(hot[hot.index("-t") + 1], "16")
        self.assertEqual(hot[hot.index("--port") + 1], "8080")
        self.assertEqual(bg[:2], ["numactl", "--cpunodebind=1"])
        self.assertEqual(bg[bg.index("--port") + 1], "8081")
        self.assertEqual(bg[bg.index("-t") + 1], "4")

    def test_none_falls_back_to_node0(self):
        hot, _ = hw_llama.server_cmds("m.gguf", 8, 2, "none")
        self.assertEqual(hot[:2], ["numactl", "--cpunodebind=0"])


class MainTest(unittest.TestCase):
    def run_main(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = hw_llama.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_default_prints_plan(self):
        code, out, _ = self.run_main([])
        self.assertEqual(code, 0)
        self.assertIn("-march=znver2", out)
        self.assertIn("--interleave=all", out)
        self.assertIn("llama-server", out)
        self.assertEqual(sum(1 for line in out.splitlines() if line.startswith("### ")), 15)

    def test_bench_without_model_fails(self):
        code, _, err = self.run_main(["--run", "bench"])
        self.assertEqual(code, 2)
        self.assertIn("--model", err)

    def test_bench_with_missing_model_fails(self):
        code, _, _ = self.run_main(["--run", "bench", "--model", "no_such_model.gguf"])
        self.assertEqual(code, 2)

    def test_parse_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bench.txt"
            path.write_text(SAMPLE_SYNTHETIC_BENCH, encoding="utf-8")
            code, out, _ = self.run_main(["--parse", str(path)])
            self.assertEqual(code, 0)
            self.assertIn("16", out)
            self.assertIn("Разобрано замеров: 15", out)
            self.assertEqual(self.run_main(["--parse", str(Path(tmp) / "none.txt")])[0], 2)

    def test_bad_numa_mode_is_rejected(self):
        with self.assertRaises(SystemExit):
            self.run_main(["--numa", "bogus"])

    def test_run_bench_with_fake_runner(self):
        calls = []

        def runner(cmd, **kwargs):
            calls.append(cmd)
            n = int(cmd[cmd.index("-t") + 1])
            mode = "node0" if cmd[0] == "numactl" and "--membind=0" in cmd else "none"
            tg = TG[mode].get(n, 1.0)
            return SimpleNamespace(stdout=json.dumps(_entries(n, tg * 4, tg)), returncode=0)

        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(model="m.gguf", threads=(8, 16), numa=("none", "node0"), build=tmp,
                                   out=str(Path(tmp) / "out.txt"))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = hw_llama.run_bench(args, runner)
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 4)
        self.assertIn("Разобрано замеров: 4", out.getvalue())

    def test_run_build_stops_on_failure(self):
        calls = []

        def runner(cmd):
            calls.append(cmd)
            return 1

        args = SimpleNamespace(src="s", build="b", jobs=2)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(hw_llama.run_build(args, runner), 1)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
