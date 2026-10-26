import tempfile
import unittest
from pathlib import Path

from organism.config import Territory
from organism.events import Event
from organism.journal import Journal
from organism.skills import Chain, ReflexCache, mine

ALLOWED = ("git status", "git --version")
TERRITORY = Territory(exec_allowlist=ALLOWED)


class SkillsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.j = Journal(self.dir / "j.db")
        self.n = 0

    def tearDown(self):
        self.j.close()
        self.tmp.cleanup()

    def episode(self, text, cmds, oks=None):
        self.n += 1
        ep = f"ep{self.n}"
        self.j.event(Event("вход", "человек", {"text": text, "episode": ep}))
        for i, cmd in enumerate(cmds):
            ok = True if oks is None else oks[i]
            self.j.action("выполнить", ok, {"cmd": cmd, "episode": ep})

    def repeat(self, n, text, cmds, oks=None):
        for _ in range(n):
            self.episode(text, cmds, oks)

    def cache(self):
        return ReflexCache(self.dir / "reflex.json")

    def test_three_same_episodes_become_reflex(self):
        self.repeat(3, "Покажи  Статус", ["git status"])
        chains = mine(self.j, 0)
        self.assertEqual(chains, [Chain("покажи статус", ("git status",), 3, 1.0)])
        cache = self.cache()
        self.assertEqual(cache.compile(chains, TERRITORY, 2), 1)
        self.assertEqual(cache.match("покажи статус"), ("git status",))
        self.assertEqual(cache.match("  ПОКАЖИ   статус "), ("git status",))

    def test_below_min_count_not_mined(self):
        self.repeat(2, "статус", ["git status"])
        self.assertEqual(mine(self.j, 0), [])
        self.assertEqual(len(mine(self.j, 0, min_count=2)), 1)

    def test_failures_rejected(self):
        self.repeat(2, "статус", ["git status"])
        self.repeat(2, "статус", ["git status"], [False])
        chains = mine(self.j, 0)
        self.assertEqual(chains[0].ok_rate, 0.5)
        cache = self.cache()
        self.assertEqual(cache.compile(chains, TERRITORY, 2), 0)
        self.assertIsNone(cache.match("статус"))

    def test_command_outside_allowlist_rejected(self):
        self.repeat(3, "удали", ["rm -rf x"])
        self.repeat(3, "смесь", ["git status", "rm x"])
        cache = self.cache()
        self.assertEqual(cache.compile(mine(self.j, 0), TERRITORY, 2), 0)
        self.assertIsNone(cache.match("удали"))
        self.assertIsNone(cache.match("смесь"))
        self.assertEqual(TERRITORY.exec_allowlist, ALLOWED)

    def test_chain_longer_than_max_chain_rejected(self):
        self.repeat(3, "всё", ["git status", "git --version", "git status"])
        self.repeat(3, "два", ["git status", "git --version"])
        cache = self.cache()
        self.assertEqual(cache.compile(mine(self.j, 0), TERRITORY, 2), 1)
        self.assertIsNone(cache.match("всё"))
        self.assertEqual(cache.match("два"), ("git status", "git --version"))

    def test_ambiguous_key_rejected(self):
        self.repeat(3, "версия", ["git --version"])
        self.repeat(3, "версия", ["git status"])
        cache = self.cache()
        self.assertEqual(cache.compile(mine(self.j, 0), TERRITORY, 2), 0)

    def test_save_load_roundtrip(self):
        self.repeat(3, "статус", ["git status"])
        cache = self.cache()
        cache.compile(mine(self.j, 0), TERRITORY, 2)
        cache.save()
        loaded = self.cache()
        self.assertEqual(loaded.reflexes, cache.reflexes)
        self.assertEqual(loaded.match("статус"), ("git status",))

    def test_missing_or_broken_file_is_empty(self):
        self.assertIsNone(self.cache().match("статус"))
        (self.dir / "reflex.json").write_text("{не json", encoding="utf-8")
        self.assertEqual(self.cache().reflexes, {})

    def test_near_text_does_not_match(self):
        self.repeat(3, "покажи статус", ["git status"])
        cache = self.cache()
        cache.compile(mine(self.j, 0), TERRITORY, 2)
        for text in ("покажи статус репозитория", "покажи стату", "статус", "", "покажи статус."):
            self.assertIsNone(cache.match(text))

    def test_episodes_before_t0_ignored(self):
        self.repeat(3, "статус", ["git status"])
        self.assertEqual(mine(self.j, 1e12), [])


if __name__ == "__main__":
    unittest.main()
