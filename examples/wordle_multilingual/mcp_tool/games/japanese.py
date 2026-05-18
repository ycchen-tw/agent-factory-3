"""Japanese Wordle: 5 katakana characters → one or more kanji words.

Hiragana input is accepted and auto-converted to katakana before any
comparison, mirroring the original ことのはたんご UI which lets players
type either script (see kotonoha.js: switch_lang katakana/hiragana toggle).
"""
from __future__ import annotations

from .base import BaseWordleGame


def _hira_to_kata(word: str) -> str:
    """Map hiragana → katakana via the +0x60 codepoint offset.

    Covers all standard hiragana (U+3041..U+3096) including small kana
    (ぁぃぅぇぉっゃゅょゎ), voiced/semi-voiced variants (がぱ etc.), and
    rare forms (ゔゕゖ). Long mark ー and any other code points pass through
    unchanged.
    """
    return "".join(
        chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c
        for c in word
    )


class JapaneseWordleGame(BaseWordleGame):
    word_length = 5
    mode = "japanese"

    def __init__(
        self,
        answer: str,
        valid_guesses: set[str],
        display_map: dict[str, list[str]],
        max_attempts: int = 6,
    ):
        self.valid_guesses = set(valid_guesses)
        self.display_map = display_map
        super().__init__(answer, max_attempts)

    def normalize(self, word: str) -> str:
        # Accept either script — the data set + scoring are katakana-canonical.
        return _hira_to_kata(word)

    def is_valid_guess(self, word: str) -> bool:
        return word in self.valid_guesses

    def display(self, word: str) -> str:
        candidates = self.display_map.get(word, [])
        if not candidates:
            return word
        return f"{word} ({'/'.join(candidates[:2])})"
