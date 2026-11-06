import tempfile
import unittest
from pathlib import Path

from organism.config import Genome
from organism.llm import FakeLLM, LLM, LLMError
from organism.mother import Mother


class Always(LLM):
    def __init__(self, answer: str):
        self.answer = answer

    async def chat(self, messages, max_tokens=256, lora=None, deadline_s=None) -> str:
        return self.answer


class ByPosition(LLM):
    async def chat(self, messages, max_tokens=256, lora=None, deadline_s=None) -> str:
        content = messages[-1]["content"]
        tail = content.rsplit("Ответ 2:", 1)[1] if "Ответ 2:" in content else ""
        return "2" if "хороший" in tail else "1"


class Queue(LLM):
    def __init__(self, answers: list[str]):
        self.answers = list(answers)

    async def chat(self, messages, max_tokens=256, lora=None, deadline_s=None) -> str:
        return self.answers.pop(0)


class Boom(LLM):
    async def chat(self, messages, max_tokens=256, lora=None, deadline_s=None) -> str:
        raise LLMError("упс")


class CompareTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def mother(self, llm) -> Mother:
        return Mother(llm, Genome(), Path(self.tmp.name) / "birth.txt")

    async def test_positional_bias_cancels_to_half(self):
        for digit in ("1", "2"):
            share = await self.mother(Always(digit)).compare("вопрос", "ответ а", "ответ б")
            self.assertEqual(share, 0.5, digit)

    async def test_fakellm_compare_gives_half(self):
        self.assertEqual(await self.mother(FakeLLM()).compare("вопрос", "ответ а", "ответ б"), 0.5)

    async def test_good_in_second_argument_wins(self):
        share = await self.mother(ByPosition()).compare("вопрос", "плохой ответ", "хороший ответ")
        self.assertEqual(share, 1.0)

    async def test_good_in_first_argument_loses(self):
        share = await self.mother(ByPosition()).compare("вопрос", "хороший ответ", "плохой ответ")
        self.assertEqual(share, 0.0)

    async def test_garbage_in_both_calls_gives_none(self):
        self.assertIsNone(await self.mother(Always("два")).compare("вопрос", "а", "б"))

    async def test_one_parsed_call_is_enough(self):
        self.assertEqual(await self.mother(Queue(["2", "мусор"])).compare("вопрос", "а", "б"), 1.0)
        self.assertEqual(await self.mother(Queue(["мусор", "1"])).compare("вопрос", "а", "б"), 1.0)

    async def test_llm_error_gives_none(self):
        self.assertIsNone(await self.mother(Boom()).compare("вопрос", "а", "б"))


if __name__ == "__main__":
    unittest.main()
