from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
from ctypes import wintypes
from pathlib import Path

ALL_PROCESSOR_GROUPS = 0xFFFF
CREATE_SUSPENDED = 0x00000004
JOB_OBJECT_GROUP_INFORMATION_EX = 14
STARTF_USESTDHANDLES = 0x00000100
INFINITE = 0xFFFFFFFF


class GROUP_AFFINITY(ctypes.Structure):
    _fields_ = [("Mask", ctypes.c_size_t), ("Group", wintypes.WORD), ("Reserved", wintypes.WORD * 3)]


class PROCESSOR_NUMBER(ctypes.Structure):
    _fields_ = [("Group", wintypes.WORD), ("Number", ctypes.c_ubyte), ("Reserved", ctypes.c_ubyte)]


class STARTUPINFOW(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("lpReserved", wintypes.LPWSTR), ("lpDesktop", wintypes.LPWSTR),
                ("lpTitle", wintypes.LPWSTR), ("dwX", wintypes.DWORD), ("dwY", wintypes.DWORD),
                ("dwXSize", wintypes.DWORD), ("dwYSize", wintypes.DWORD), ("dwXCountChars", wintypes.DWORD),
                ("dwYCountChars", wintypes.DWORD), ("dwFillAttribute", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("wShowWindow", wintypes.WORD), ("cbReserved2", wintypes.WORD), ("lpReserved2", ctypes.c_void_p),
                ("hStdInput", wintypes.HANDLE), ("hStdOutput", wintypes.HANDLE), ("hStdError", wintypes.HANDLE)]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [("hProcess", wintypes.HANDLE), ("hThread", wintypes.HANDLE),
                ("dwProcessId", wintypes.DWORD), ("dwThreadId", wintypes.DWORD)]


def mask_bits(mask: int) -> list[int]:
    return [i for i in range(mask.bit_length()) if mask >> i & 1]


def _k32():
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetActiveProcessorGroupCount.restype = wintypes.WORD
    k.GetActiveProcessorCount.argtypes = [wintypes.WORD]
    k.GetActiveProcessorCount.restype = wintypes.DWORD
    k.GetNumaHighestNodeNumber.argtypes = [ctypes.POINTER(wintypes.ULONG)]
    k.GetNumaNodeProcessorMaskEx.argtypes = [wintypes.USHORT, ctypes.POINTER(GROUP_AFFINITY)]
    k.GetNumaAvailableMemoryNodeEx.argtypes = [wintypes.USHORT, ctypes.POINTER(ctypes.c_ulonglong)]
    k.GetCurrentProcessorNumberEx.argtypes = [ctypes.POINTER(PROCESSOR_NUMBER)]
    k.GetCurrentProcessorNumberEx.restype = None
    k.GetNumaProcessorNodeEx.argtypes = [ctypes.POINTER(PROCESSOR_NUMBER), ctypes.POINTER(wintypes.USHORT)]
    k.SetThreadGroupAffinity.argtypes = [wintypes.HANDLE, ctypes.POINTER(GROUP_AFFINITY), ctypes.c_void_p]
    k.ResumeThread.argtypes = [wintypes.HANDLE]
    k.ResumeThread.restype = wintypes.DWORD
    k.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    k.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k.CreateJobObjectW.restype = wintypes.HANDLE
    k.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k.CreateProcessW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.c_void_p, ctypes.c_void_p,
                                 wintypes.BOOL, wintypes.DWORD, ctypes.c_void_p, wintypes.LPCWSTR,
                                 ctypes.c_void_p, ctypes.POINTER(PROCESS_INFORMATION)]
    k.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    return k


def topology() -> dict:
    k = _k32()
    groups = [{"group": g, "processors": k.GetActiveProcessorCount(g)}
              for g in range(k.GetActiveProcessorGroupCount())]
    highest = wintypes.ULONG(0)
    k.GetNumaHighestNodeNumber(ctypes.byref(highest))
    nodes = []
    for n in range(highest.value + 1):
        ga = GROUP_AFFINITY()
        if not k.GetNumaNodeProcessorMaskEx(n, ctypes.byref(ga)):
            continue
        free = ctypes.c_ulonglong(0)
        has_mem = bool(k.GetNumaAvailableMemoryNodeEx(n, ctypes.byref(free)))
        nodes.append({"node": n, "group": ga.Group, "mask": hex(ga.Mask), "processors": len(mask_bits(ga.Mask)),
                      "available_mb": free.value // 2**20 if has_mem else None})
    return {"groups": groups, "all_processors": k.GetActiveProcessorCount(ALL_PROCESSOR_GROUPS), "nodes": nodes}


def node_affinity(node: int) -> tuple[int, int]:
    for n in topology()["nodes"]:
        if n["node"] == node:
            return n["group"], int(n["mask"], 16)
    raise ValueError(f"узла {node} нет")


def whereami() -> dict:
    k = _k32()
    pn = PROCESSOR_NUMBER()
    k.GetCurrentProcessorNumberEx(ctypes.byref(pn))
    node = wintypes.USHORT(0)
    ok = k.GetNumaProcessorNodeEx(ctypes.byref(pn), ctypes.byref(node))
    return {"pid": os.getpid(), "group": pn.Group, "processor": pn.Number, "node": node.value if ok else None,
            "os_cpu_count": os.cpu_count()}


def _inheritable(path: str | None, mode: str):
    if not path:
        return None, None
    import msvcrt
    f = open(path, mode)
    handle = msvcrt.get_osfhandle(f.fileno())
    os.set_handle_inheritable(handle, True)
    return f, handle


def launch(cmd: list[str], group: int, mask: int, out: str | None = None, err: str | None = None) -> int:
    import msvcrt
    k = _k32()
    fo, ho = _inheritable(out, "wb")
    fe, he = _inheritable(err, "wb")
    try:
        si = STARTUPINFOW()
        si.cb = ctypes.sizeof(si)
        if ho or he:
            si.dwFlags |= STARTF_USESTDHANDLES
            si.hStdInput = msvcrt.get_osfhandle(sys.stdin.fileno()) if sys.stdin else None
            si.hStdOutput = ho or msvcrt.get_osfhandle(sys.stdout.fileno())
            si.hStdError = he or msvcrt.get_osfhandle(sys.stderr.fileno())
        pi = PROCESS_INFORMATION()
        line = ctypes.create_unicode_buffer(subprocess.list2cmdline(cmd))
        if not k.CreateProcessW(None, line, None, None, bool(ho or he), CREATE_SUSPENDED, None, None,
                                ctypes.byref(si), ctypes.byref(pi)):
            raise ctypes.WinError(ctypes.get_last_error())
        job = None
        try:
            ga = GROUP_AFFINITY(Mask=mask, Group=group)
            job = k.CreateJobObjectW(None, None)
            steps = (job,
                     job and k.SetInformationJobObject(job, JOB_OBJECT_GROUP_INFORMATION_EX, ctypes.byref(ga),
                                                       ctypes.sizeof(ga)),
                     job and k.AssignProcessToJobObject(job, pi.hProcess),
                     k.SetThreadGroupAffinity(pi.hThread, ctypes.byref(ga), None))
            if not all(steps):
                error = ctypes.WinError(ctypes.get_last_error())
                k.TerminateProcess(pi.hProcess, 1)
                raise error
            k.ResumeThread(pi.hThread)
            k.WaitForSingleObject(pi.hProcess, INFINITE)
            code = wintypes.DWORD(0)
            k.GetExitCodeProcess(pi.hProcess, ctypes.byref(code))
            return code.value
        finally:
            k.CloseHandle(pi.hThread)
            k.CloseHandle(pi.hProcess)
            if job:
                k.CloseHandle(job)
    finally:
        for f in (fo, fe):
            if f:
                f.close()


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[argv.index("--") + 1:] if "--" in argv else []
    own = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description="Группы процессоров Windows: карта и запуск на узле или группе")
    ap.add_argument("--topology", action="store_true")
    ap.add_argument("--whereami", action="store_true")
    ap.add_argument("--node", type=int)
    ap.add_argument("--group", type=int)
    ap.add_argument("--mask", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--err", default=None)
    args = ap.parse_args(own)
    if os.name != "nt":
        print("только для Windows", file=sys.stderr)
        return 2
    if args.topology or args.whereami:
        report = topology() if args.topology else whereami()
        text = json.dumps(report, ensure_ascii=False, indent=2)
        if args.out:
            Path(args.out).write_text(text, encoding="utf-8")
        print(text)
        return 0
    if not cmd or (args.node is None and args.group is None):
        ap.error("нужны --node или --group и команда после --")
    if args.node is not None:
        group, mask = node_affinity(args.node)
    else:
        group, mask = args.group, 0
    if args.mask:
        mask = int(args.mask, 16)
    if not mask:
        mask = (1 << _k32().GetActiveProcessorCount(group)) - 1
    return launch(cmd, group, mask, args.out, args.err)


if __name__ == "__main__":
    raise SystemExit(main())
