"""Japanese Wordle: 5 katakana characters → one or more kanji words."""
from __future__ import annotations

from .base import BaseWordleGame


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

    def is_valid_guess(self, word: str) -> bool:
        return word in self.valid_guesses

    def display(self, word: str) -> str:
        candidates = self.display_map.get(word, [])
        if not candidates:
            return word
        return f"{word} ({'/'.join(candidates[:2])})"
