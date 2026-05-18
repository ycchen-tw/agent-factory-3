"""English Wordle: 5-letter words, case-insensitive."""
from __future__ import annotations

from .base import BaseWordleGame


class EnglishWordleGame(BaseWordleGame):
    word_length = 5
    mode = "english"

    def __init__(self, answer: str, valid_guesses: set[str], max_attempts: int = 6):
        self.valid_guesses = {w.lower() for w in valid_guesses}
        super().__init__(answer, max_attempts)

    def normalize(self, word: str) -> str:
        return word.lower()

    def is_valid_guess(self, word: str) -> bool:
        return word in self.valid_guesses
