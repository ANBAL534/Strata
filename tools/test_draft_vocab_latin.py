"""The Latin draft subset keeps accented words and removes direct tokens for other scripts."""
from __future__ import annotations

import unittest

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
                             "niño".encode(), "jour".encode()])
        base = [0, 1, 2, 3, 4, 5, 14]
        self.assertEqual(latin_subset(tok, base), [0, 4, 5, 6, 7, 8, 9, 15])


if __name__ == "__main__":
    unittest.main()
