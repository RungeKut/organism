from __future__ import annotations

import asyncio
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor


def membw_gbps(mb: int = 64, runs: int = 5) -> float:
    size = max(1, int(mb)) * 2**20
    src, dst = bytearray(size), bytearray(size)
    best = float("inf")
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        dst[:] = src
        best = min(best, time.perf_counter() - t0)
    return 2 * size / max(best, 1e-9) / 1e9


def curve(threads: tuple[int, ...] = (1, 2, 4, 6), tasks: int = 24, buf_mb: int = 2,
          repeats: int = 2) -> list[tuple[int, float]]:
    buf = bytes(max(1, int(buf_mb)) * 2**20)
    result = []
    for n in sorted({max(1, int(t)) for t in threads}):
        best = float("inf")
        with ThreadPoolExecutor(max_workers=n) as pool:
            list(pool.map(lambda _: hashlib.sha256(buf).digest(), range(n)))
            for _ in range(max(1, repeats)):
                t0 = time.perf_counter()
                list(pool.map(lambda _: hashlib.sha256(buf).digest(), range(tasks)))
                best = min(best, time.perf_counter() - t0)
        result.append((n, tasks / max(best, 1e-9)))
    return result


def knee(points: list[tuple[int, float]], gain: float = 0.1) -> int:
    if not points:
        return 1
    for (n, rate), (_, nxt) in zip(points, points[1:]):
        if nxt < rate * (1 + gain):
            return n
    return points[-1][0]


async def llm_curve(llm, slots: tuple[int, ...] = (1, 2, 4), per_slot: int = 4) -> list[tuple[int, float]]:
    messages = [{"role": "user", "content": "калибровка"}]
    result = []
    for s in sorted({max(1, int(x)) for x in slots}):
        gate = asyncio.Semaphore(s)

        async def one() -> None:
            async with gate:
                await llm.chat(messages, max_tokens=8)

        total = s * max(1, per_slot)
        t0 = time.perf_counter()
        await asyncio.gather(*(one() for _ in range(total)))
        result.append((s, total / max(time.perf_counter() - t0, 1e-9)))
    return result
