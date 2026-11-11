from __future__ import annotations

import json
import re

from .llm import LLM, LLMError
from .mother import Mother

MAX_PROFESSIONS = 8
MAX_PROMPT = 600
NAME = re.compile(r"^[\w]+$")
SYSTEM = ("Ты архитектор организма. Тебе дан список профессий (JSON) и сводка работы. "
          "Предложи ровно одно изменение: добавь одну профессию или перепиши промпт одной существующей. "
          "Профессия нужна для задач, с которыми организм справляется плохо (см. сводку); имя по-русски, одно слово, не «архитектор». "
          "Поля профессии: name (одно слово), prompt (до 600 знаков), lora_id: null, tools: []. "
          "Ответь только полным JSON-списком профессий.")
BASELINE = "Ты помощник."
BREVITY = "Отвечай коротко и по делу, не больше пяти предложений."
QUESTIONS = 8
WIN = 0.75


def _load(text: str) -> list[dict] | None:
    try:
        data = json.loads(text)
    except ValueError:
        return None
    return data if isinstance(data, list) and all(isinstance(r, dict) for r in data) else None


def extract(raw: str) -> str | None:
    start, end = raw.find("["), raw.rfind("]")
    if start < 0 or end < start:
        return None
    rows = _load(raw[start:end + 1])
    return None if rows is None else json.dumps(rows, ensure_ascii=False, indent=2)


def changed(before: str, after: str, reserved: frozenset[str]) -> dict | None:
    old, new = _load(before), _load(after)
    if old is None or new is None or len(new) > MAX_PROFESSIONS:
        return None
    for row in new:
        name, prompt = row.get("name"), row.get("prompt")
        if not isinstance(name, str) or not NAME.match(name):
            return None
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT:
            return None
        if row.get("lora_id") is not None or row.get("tools", []) != []:
            return None
    if len({r["name"] for r in new}) != len(new):
        return None
    old_by = {r.get("name"): r for r in old}
    diff = [r for r in new if old_by.get(r["name"]) != r]
    removed = set(old_by) - {r["name"] for r in new}
    if len(diff) != 1 or removed:
        return None
    row = diff[0]
    if row["name"] in reserved and row["name"] not in old_by:
        return None
    if row["name"] not in old_by:
        if any(other.get("prompt") == row["prompt"] for other in old):
            return None
    return row


def summary(memories: list[str], stats: dict, inputs: list[str], failures: dict) -> str:
    asked = "; ".join(t[:80] for t in inputs[-5:]) or "нет"
    dreams = " | ".join(memories) or "нет"
    return "\n".join([f"Последние сны: {dreams}",
                      f"Счётчики: {json.dumps(stats, ensure_ascii=False)}",
                      f"Сбои по видам: {json.dumps(failures, ensure_ascii=False)}",
                      f"Последние вопросы людей: {asked}"])


async def propose_text(llm: LLM, before: str, summary: str) -> str | None:
    try:
        raw = await llm.chat([{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": f"Профессии:\n{before}\n\nСводка:\n{summary}"}],
                             max_tokens=700)
    except (LLMError, PermissionError):
        return None
    return extract(raw)


async def questions(llm: LLM, row: dict) -> list[str]:
    ask = (f"Профессия «{row['name']}»: {row['prompt']}\nПридумай {QUESTIONS} коротких вопроса из её области, "
           "по одному в строке, без нумерации.")
    try:
        raw = await llm.chat([{"role": "system", "content": "Составляй вопросы для проверки."},
                              {"role": "user", "content": ask}], max_tokens=150)
    except (LLMError, PermissionError):
        return []
    lines = [ln.strip(" -*\t0123456789.)") for ln in raw.splitlines()]
    return [ln for ln in lines if len(ln) > 3][:QUESTIONS]


async def _answer(llm: LLM, prompt: str, text: str) -> str | None:
    try:
        raw = await llm.chat([{"role": "system", "content": f"{prompt} {BREVITY}"}, {"role": "user", "content": text}],
                             max_tokens=200)
    except (LLMError, PermissionError):
        return None
    return raw.strip()


async def duel(llm: LLM, mother: Mother, base_prompt: str, new_prompt: str, texts: list[str], log=None) -> float:
    wins, parsed = 0.0, 0
    for text in texts:
        base = await _answer(llm, base_prompt, text)
        new = await _answer(llm, new_prompt, text)
        if base is None or new is None:
            continue
        if log is None:
            share = await mother.compare(text, base, new)
        else:
            calls: list[dict] = []
            share = await mother.compare(text, base, new, log=calls.append)
            log({"вопрос": text, "база": base, "новая": new, "доля_нового": share, "судья": calls})
        if share is not None:
            wins += share
            parsed += 1
    return wins / parsed if parsed else 0.5
