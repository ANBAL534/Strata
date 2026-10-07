"""The Latin draft subset keeps accented words and removes direct tokens for other scripts."""
from __future__ import annotations

import unittest

from tools import draft_vocab as DV
from tools.draft_vocab import latin_subset


class FakeTokenizer:
    def __init__(self, tokens: list[bytes]):
        self.tokens = tokens

    def token_bytes(self, i: int) -> bytes:
        return self.tokens[i]


class LatinSubset(unittest.TestCase):
    def test_latin_accents_without_cjk_cyrillic_or_other_scripts(self):
        tok = FakeTokenizer([b" hello", "привет".encode(), "漢".encode(), "ー".encode(),
                             b"\xe9", b"!", "café".encode(), "Straße".encode(),
                             "façade".encode(), "żółć".encode(), "áпривет".encode(),
                             b"...", "γειά".encode(), "😄".encode(), "、".encode(),
                             "niño".encode(), "jour".encode(), "e\u0301".encode()])
        base = [0, 1, 2, 3, 4, 5, 14]
        self.assertEqual(latin_subset(tok, base), [0, 4, 5, 6, 7, 8, 9, 15, 17])

    def test_latin_is_a_shared_script_selector(self):
        self.assertIn("latin", DV.SCRIPTS)
        tok = FakeTokenizer([b" hello", "привет".encode(), "漢".encode(), "café".encode(),
                             b"\xe9", "niño".encode()])
        self.assertEqual(DV.select_ids(tok, [0, 1, 2, 4], {"latin"}), [0, 4, 3, 5])

    def test_latin_selector_rejects_other_additions_and_corpus(self):
        tok = FakeTokenizer([b"hello"])
        with self.assertRaisesRegex(ValueError, "--add latin needs --base"):
            DV.select_ids(tok, [], {"latin"})
        with self.assertRaisesRegex(ValueError, "--add latin cannot be combined"):
            DV.select_ids(tok, [], {"latin", "cyrillic"})
        with self.assertRaisesRegex(ValueError, "--add latin cannot be combined"):
            DV.select_ids(tok, [], {"latin"}, {0})


if __name__ == "__main__":
    unittest.main()
