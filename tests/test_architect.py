import json
import tempfile
import unittest
from pathlib import Path

from organism import architect, holdout, proposals
from organism.config import Genome, Territory
from organism.llm import LLM
from organism.organism import Organism

RES = frozenset({"речь", "моторика"})
GOOD = {"name": "аналитик", "prompt": "Ты аналитик.", "lora_id": None, "tools": []}


def dump(rows):
    return json.dumps(rows, ensure_ascii=False)


class ChangedTest(unittest.TestCase):
    def test_new_profession_ok(self):
        self.assertEqual(architect.changed("[]", dump([GOOD]), RES), GOOD)

    def test_rejects_reserved_dash_tools_lora_long_prompt(self):
        bad = [dict(GOOD, name="речь"), dict(GOOD, name="а-б"), dict(GOOD, tools=["git status"]),
               dict(GOOD, lora_id=3), dict(GOOD, prompt="я" * 601), dict(GOOD, prompt=" ")]
        for row in bad:
            self.assertIsNone(architect.changed("[]", dump([row]), RES), row)

    def test_only_one_change_and_no_removal(self):
        other = dict(GOOD, name="юрист")
        self.assertIsNone(architect.changed("[]", dump([GOOD, other]), RES))
        self.assertIsNone(architect.changed(dump([GOOD, other]), dump([GOOD]), RES))
        edit = dict(GOOD, prompt="Новый промпт")
        self.assertEqual(architect.changed(dump([GOOD, other]), dump([edit, other]), RES), edit)
        self.assertIsNone(architect.changed(dump([GOOD]), dump([GOOD]), RES))

    def test_extract_from_noisy_answer(self):
        self.assertEqual(json.loads(architect.extract("вот: " + dump([GOOD]) + " готово")), [GOOD])
        self.assertIsNone(architect.extract("нет списка"))
        self.assertIsNone(architect.extract("[не json]"))


class QuestionsTest(unittest.IsolatedAsyncioTestCase):
    async def test_questions_parsed_from_lines(self):
        class Q(LLM):
            async def chat(self, messages, max_tokens=256, lora=None, deadline_s=None):
                return "\n".join(["1. Что такое рынок?", "- Как считать маржу?", "", "а", "3) Зачем нужен бюджет?", "лишний вопрос здесь"])

        got = await architect.questions(Q(), GOOD)
        self.assertEqual(got, ["Что такое рынок?", "Как считать маржу?", "Зачем нужен бюджет?"])


class Scripted(LLM):
    def __init__(self, answer: str, good_for_profession: bool):
        self.answer = answer
        self.good = good_for_profession

    async def chat(self, messages, max_tokens=256, lora=None, deadline_s=None) -> str:
        system, last = messages[0]["content"], messages[-1]["content"]
        if system == architect.SYSTEM:
            return self.answer
        if system.startswith("Сравни"):
            tail = last.rsplit("Ответ 2:", 1)[1] if "Ответ 2:" in last else ""
            return "2" if "хороший" in tail else "1"
        if system.startswith(GOOD["prompt"]):
            return "хороший ответ" if self.good else "плохой ответ"
        return "плохой ответ"


class ArchitectOrganismTest(unittest.IsolatedAsyncioTestCase):
    async def run_architect(self, llm):
        with tempfile.TemporaryDirectory() as tmp:
            org = Organism(tmp, llm, Territory(exec_allowlist=()), Genome())
            holdout.add(Path(tmp), "вопрос один")
            holdout.add(Path(tmp), "вопрос два")
            await org._architect()
            topics = [t for _, t, _, _ in org.journal.all_events_since(0)]
            pending = proposals.pending(Path(tmp))
            org.journal.close()
            return topics, pending

    async def test_better_profession_is_proposed(self):
        topics, pending = await self.run_architect(Scripted(dump([GOOD]), True))
        self.assertIn("архитектор_предложил", topics)
        self.assertEqual(len(pending), 1)
        self.assertEqual((pending[0]["kind"], pending[0]["target"]), ("профессия", "professions.json"))
        self.assertGreater(pending[0]["child_score"], pending[0]["parent_score"])

    async def test_not_better_is_not_proposed(self):
        topics, pending = await self.run_architect(Scripted(dump([GOOD]), False))
        self.assertIn("архитектор_не_лучше", topics)
        self.assertEqual(pending, [])

    async def test_garbage_answer_gives_no_proposal(self):
        topics, pending = await self.run_architect(Scripted("не знаю", True))
        self.assertIn("архитектор_без_предложения", topics)
        self.assertEqual(pending, [])


class SummaryTest(unittest.TestCase):
    def test_summary_has_failures_and_questions(self):
        text = architect.summary(["сон 1"], {"сбой_llm": 2}, ["а", "б" * 200], {"отклонено": 3})
        for part in ("сон 1", "сбой_llm", "отклонено", "а;"):
            self.assertIn(part, text)
        self.assertNotIn("б" * 100, text)
