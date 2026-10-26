import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import hw_probe

_FLAGS = ("fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht "
          "syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl nonstop_tsc cpuid extd_apicid "
          "aperfmperf pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand "
          "lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce "
          "topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba ibrs ibpb "
          "stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 cqm rdt_a rdseed adx smap clflushopt clwb sha_ni xsaveopt "
          "xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local clzero irperf xsaveerptr rdpru "
          "wbnoinvd arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter "
          "pfthreshold avic v_vmsave_vmload vgif v_spec_ctrl umip rdpid overflow_recov succor smca")

SAMPLE_SYNTHETIC_LSCPU = f"""Architecture:                    x86_64
CPU op-mode(s):                  32-bit, 64-bit
Address sizes:                   43 bits physical, 48 bits virtual
Byte Order:                      Little Endian
CPU(s):                          192
On-line CPU(s) list:             0-191
Vendor ID:                       AuthenticAMD
Model name:                      AMD EPYC 7K62 48-Core Processor
CPU family:                      23
Model:                           49
Thread(s) per core:              2
Core(s) per socket:              48
Socket(s):                       2
Stepping:                        0
Frequency boost:                 enabled
CPU max MHz:                     2600.0000
CPU min MHz:                     1500.0000
L1d cache:                       3 MiB (96 instances)
L2 cache:                        48 MiB (96 instances)
L3 cache:                        384 MiB (24 instances)
NUMA node(s):                    2
NUMA node0 CPU(s):               0-47,96-143
NUMA node1 CPU(s):               48-95,144-191
Flags:                           {_FLAGS}
"""

_CPUS0 = " ".join(str(c) for c in [*range(0, 48), *range(96, 144)])
_CPUS1 = " ".join(str(c) for c in [*range(48, 96), *range(144, 192)])

SAMPLE_SYNTHETIC_NUMACTL = f"""available: 2 nodes (0-1)
node 0 cpus: {_CPUS0}
node 0 size: 257570 MB
node 0 free: 251204 MB
node 1 cpus: {_CPUS1}
node 1 size: 0 MB
node 1 free: 0 MB
node distances:
node     0    1
   0:   10   32
   1:   32   10
"""


def _dimm(locator, bank, size, speed):
    if size is None:
        body = "\tSize: No Module Installed\n\tForm Factor: DIMM\n\tSet: None\n"
        body += f"\tLocator: {locator}\n\tBank Locator: {bank}\n\tType: Unknown\n\tSpeed: Unknown\n"
        body += "\tManufacturer: NO DIMM\n\tConfigured Memory Speed: Unknown\n"
    else:
        body = f"\tTotal Width: 72 bits\n\tData Width: 64 bits\n\tSize: {size} GB\n\tForm Factor: DIMM\n\tSet: None\n"
        body += f"\tLocator: {locator}\n\tBank Locator: {bank}\n\tType: DDR4\n"
        body += f"\tType Detail: Synchronous Registered (Buffered)\n\tSpeed: {speed} MT/s\n"
        body += f"\tManufacturer: Samsung\n\tConfigured Memory Speed: {speed} MT/s\n"
    return f"Handle 0x00{len(locator):02d}, DMI type 17, 92 bytes\nMemory Device\n\tArray Handle: 0x0040\n{body}\n"


def _dmidecode(sizes):
    head = ("# dmidecode 3.3\nGetting SMBIOS data from sysfs.\nSMBIOS 3.2.0 present.\n\n"
            "Handle 0x0040, DMI type 16, 23 bytes\nPhysical Memory Array\n\tLocation: System Board Or Motherboard\n"
            "\tNumber Of Devices: 16\n\n")
    names = "ABCDEFGHIJKLMNOP"
    return head + "".join(
        _dimm(f"DIMM_{names[i]}1", f"P{i // 8}_Node{i // 8}_Channel{i % 8}_Dimm0", s[0], s[1] if s else 0)
        if s else _dimm(f"DIMM_{names[i]}1", f"P{i // 8}_Node{i // 8}_Channel{i % 8}_Dimm0", None, 0)
        for i, s in enumerate(sizes))


SAMPLE_SYNTHETIC_DMIDECODE_4OF16 = _dmidecode([(64, 3200)] * 4 + [None] * 12)
SAMPLE_SYNTHETIC_DMIDECODE_MIXED = _dmidecode([(64, 3200)] * 2 + [(32, 2933)] * 2 + [None] * 12)
SAMPLE_SYNTHETIC_DMIDECODE_FULL = _dmidecode([(64, 3200)] * 8 + [None] * 8)

SAMPLE_SYNTHETIC_MEMINFO = """MemTotal:       263752704 kB
MemFree:        257233920 kB
MemAvailable:   259000000 kB
Buffers:           100000 kB
"""


class ParseLscpuTest(unittest.TestCase):
    def test_fields(self):
        r = hw_probe.parse_lscpu(SAMPLE_SYNTHETIC_LSCPU)
        self.assertEqual(r["model_name"], "AMD EPYC 7K62 48-Core Processor")
        self.assertEqual((r["sockets"], r["cores_per_socket"], r["threads_per_core"], r["cpus"]), (2, 48, 2, 192))
        self.assertEqual(r["l3"], "384 MiB (24 instances)")
        self.assertEqual(r["l3_mb"], 384.0)
        self.assertEqual(r["flags"], {"avx2": True, "avx512f": False, "fma": True})
        self.assertEqual(r["numa_nodes"], [{"id": 0, "cpus": "0-47,96-143"}, {"id": 1, "cpus": "48-95,144-191"}])

    def test_empty(self):
        r = hw_probe.parse_lscpu("")
        self.assertIsNone(r["sockets"])
        self.assertEqual(r["numa_nodes"], [])
        self.assertFalse(r["flags"]["avx2"])

    def test_old_l3_format(self):
        r = hw_probe.parse_lscpu("L3 cache:  16384K\r\nFlags: avx avx512f\r\n")
        self.assertEqual(r["l3_mb"], 16.0)
        self.assertTrue(r["flags"]["avx512f"])
        self.assertFalse(r["flags"]["avx2"])


class ParseNumactlTest(unittest.TestCase):
    def test_nodes_and_distances(self):
        r = hw_probe.parse_numactl_hardware(SAMPLE_SYNTHETIC_NUMACTL)
        self.assertEqual([n["id"] for n in r["nodes"]], [0, 1])
        self.assertEqual(r["nodes"][0]["size_mb"], 257570)
        self.assertEqual(r["nodes"][0]["free_mb"], 251204)
        self.assertEqual(r["nodes"][0]["cpus"], "0-47,96-143")
        self.assertEqual(r["nodes"][1]["size_mb"], 0)
        self.assertEqual(r["distances"], [[10, 32], [32, 10]])

    def test_node_without_cpus(self):
        r = hw_probe.parse_numactl_hardware("node 1 cpus:\nnode 1 size: 100 MB\nnode 1 free: 90 MB\n")
        self.assertEqual(r["nodes"], [{"id": 1, "cpus": "", "size_mb": 100, "free_mb": 90}])

    def test_empty(self):
        self.assertEqual(hw_probe.parse_numactl_hardware(""), {"nodes": [], "distances": []})


class ParseDmidecodeTest(unittest.TestCase):
    def test_occupied_only(self):
        slots = hw_probe.parse_dmidecode_memory(SAMPLE_SYNTHETIC_DMIDECODE_4OF16)
        self.assertEqual(len(slots), 4)
        self.assertEqual([s["locator"] for s in slots], ["DIMM_A1", "DIMM_B1", "DIMM_C1", "DIMM_D1"])
        self.assertTrue(all(s["size_gb"] == 64 and s["speed_mts"] == 3200 and s["populated"] for s in slots))
        self.assertEqual(hw_probe.count_dmidecode_empty(SAMPLE_SYNTHETIC_DMIDECODE_4OF16), 12)

    def test_mb_size_and_mhz_speed(self):
        text = "Memory Device\n\tSize: 32768 MB\n\tLocator: A\n\tConfigured Clock Speed: 2933 MHz\n"
        slots = hw_probe.parse_dmidecode_memory(text)
        self.assertEqual((slots[0]["size_gb"], slots[0]["speed_mts"]), (32.0, 2933))

    def test_empty(self):
        self.assertEqual(hw_probe.parse_dmidecode_memory(""), [])
        self.assertEqual(hw_probe.count_dmidecode_empty(""), 0)


class ChannelsReportTest(unittest.TestCase):
    def setUp(self):
        self.nodes = hw_probe.parse_numactl_hardware(SAMPLE_SYNTHETIC_NUMACTL)["nodes"]

    def report(self, text, nodes=None):
        return hw_probe.channels_report(hw_probe.parse_dmidecode_memory(text), nodes or self.nodes,
                                        empty_slots=hw_probe.count_dmidecode_empty(text))

    def test_catches_node_without_memory_and_incomplete_channels(self):
        r = self.report(SAMPLE_SYNTHETIC_DMIDECODE_4OF16)
        text = "\n".join(r["warnings"])
        self.assertIn("узел 1 без своей памяти", text)
        self.assertIn("заняты только 4 слотов", text)
        self.assertIn("из 16", text)
        self.assertNotIn("разного размера", text)
        self.assertEqual((r["occupied"], r["empty"], r["total_slots"], r["total_gb"]), (4, 12, 16, 256.0))
        self.assertEqual(r["nodes_without_memory"], [1])

    def test_mixed_sizes(self):
        r = self.report(SAMPLE_SYNTHETIC_DMIDECODE_MIXED)
        text = "\n".join(r["warnings"])
        self.assertIn("планки разного размера: 32, 64 ГБ", text)
        self.assertIn("разной скорости", text)

    def test_clean(self):
        nodes = [{"id": 0, "cpus": "0-47", "size_mb": 524288, "free_mb": 500000}]
        r = self.report(SAMPLE_SYNTHETIC_DMIDECODE_FULL, nodes)
        self.assertEqual(r["warnings"], [])

    def test_slots_unknown(self):
        r = hw_probe.channels_report(None, self.nodes)
        self.assertEqual(r["warnings"], ["узел 1 без своей памяти: его потоки читают чужие каналы"])
        self.assertIsNone(r["occupied"])

    def test_memory_unknown_gives_no_node_warning(self):
        r = hw_probe.channels_report(None, [{"id": 0, "cpus": "0", "size_mb": None}])
        self.assertEqual(r["warnings"], [])


class HelpersTest(unittest.TestCase):
    def test_cpulist_roundtrip(self):
        cpus = hw_probe.parse_cpulist("0-3,8,10-11")
        self.assertEqual(cpus, [0, 1, 2, 3, 8, 10, 11])
        self.assertEqual(hw_probe.format_cpulist(cpus), "0-3,8,10-11")
        self.assertEqual(hw_probe.parse_cpulist(""), [])

    def test_meminfo(self):
        r = hw_probe.parse_meminfo(SAMPLE_SYNTHETIC_MEMINFO)
        self.assertEqual(r["total_mb"], 263752704 // 1024)
        node = hw_probe.parse_meminfo("Node 1 MemTotal:  1048576 kB\nNode 1 MemFree:  524288 kB\n")
        self.assertEqual((node["total_mb"], node["free_mb"]), (1024, 512))

    def test_sysfs_nodes(self):
        with tempfile.TemporaryDirectory() as d:
            for i, (cpus, kb) in enumerate((("0-3", 4194304), ("4-7", 0))):
                n = Path(d) / f"node{i}"
                n.mkdir()
                (n / "cpulist").write_text(cpus + "\n")
                (n / "meminfo").write_text(f"Node {i} MemTotal:  {kb} kB\nNode {i} MemFree:  {kb // 2} kB\n")
            nodes = hw_probe.read_sysfs_nodes(Path(d))
        self.assertEqual([(n["id"], n["cpus"], n["size_mb"]) for n in nodes], [(0, "0-3", 4096), (1, "4-7", 0)])
        self.assertEqual(hw_probe.read_sysfs_nodes(Path("/nonexistent/hw_probe")), [])

    def test_bandwidth_cmd(self):
        cmd = hw_probe.bandwidth_cmd("/usr/bin/numactl", 1, 0, 64, 5, python="py")
        self.assertEqual(cmd[:5], ["/usr/bin/numactl", "--cpunodebind=1", "--membind=0", "py", "-c"])
        self.assertIn("from organism.calibrate import membw_gbps", cmd[5])
        self.assertIn("membw_gbps(64, 5)", cmd[5])


def _fake_run(cmd, timeout=30, cwd=None):
    tool = Path(cmd[0]).name
    if tool == "lscpu":
        return 0, SAMPLE_SYNTHETIC_LSCPU
    if tool == "dmidecode":
        return 0, SAMPLE_SYNTHETIC_DMIDECODE_4OF16
    if tool == "numactl" and "--hardware" in cmd:
        return 0, SAMPLE_SYNTHETIC_NUMACTL
    if tool == "numactl":
        return 0, "warming\n" + ("21.5" if "--cpunodebind=0" in cmd else "9.25") + "\n"
    return None


class CollectTest(unittest.TestCase):
    def test_collect_real_machine_does_not_fail(self):
        r = hw_probe.collect(bw_mb=4, bw_runs=1)
        for key in ("host", "logical_cpus", "nodes", "channels", "bandwidth", "curve", "knee", "warnings", "skipped"):
            self.assertIn(key, r)
        self.assertGreaterEqual(len(r["nodes"]), 1)
        self.assertGreaterEqual(r["knee"], 1)
        json.dumps(r, ensure_ascii=False)

    def test_collect_without_tools(self):
        with mock.patch.object(hw_probe, "_which", lambda name: None), \
                mock.patch.object(hw_probe, "_run", lambda *a, **k: None), \
                mock.patch.object(hw_probe, "read_sysfs_nodes", lambda *a, **k: []):
            r = hw_probe.collect(bw_mb=4, bw_runs=1)
        self.assertEqual(r["bandwidth"]["mode"], "общий")
        self.assertIsNone(r["dmidecode"]["slots"])
        self.assertTrue(any("dmidecode" in s for s in r["skipped"]))
        self.assertTrue(any("lscpu" in s for s in r["skipped"]))

    def test_collect_with_server_samples(self):
        with mock.patch.object(hw_probe, "_which", lambda name: "/usr/bin/" + name), \
                mock.patch.object(hw_probe, "_run", _fake_run):
            r = hw_probe.collect(bw_mb=4, bw_runs=1)
        self.assertEqual(r["lscpu"]["sockets"], 2)
        self.assertEqual(r["bandwidth"]["mode"], "numactl")
        self.assertEqual(r["bandwidth"]["local_gbps"], {"0": 21.5})
        self.assertEqual(r["bandwidth"]["remote_gbps"], {"1": 9.25})
        self.assertEqual(r["nodes"][0]["cpu_count"], 96)
        self.assertEqual(r["channels"]["occupied"], 4)
        self.assertTrue(any("узел 1 без своей памяти" in w for w in r["warnings"]))
        self.assertTrue(any("заняты только 4 слотов" in w for w in r["warnings"]))
        text = hw_probe.format_summary(r)
        self.assertIn("Предупреждения", text)
        self.assertIn("EPYC 7K62", text)


class MainTest(unittest.TestCase):
    def test_writes_valid_json(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "sub" / "probe.json"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = hw_probe.main(["--out", str(out), "--bw-mb", "4", "--bw-runs", "1"])
            data = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertIn("nodes", data)
        self.assertIn("Узлы NUMA", buf.getvalue())
        self.assertIn(str(out), buf.getvalue())


if __name__ == "__main__":
    unittest.main()
