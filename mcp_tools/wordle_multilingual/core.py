"""
Wordle Game Core - Shared logic for multilingual Wordle implementations
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


@dataclass
class GuessResult:
    """Result for a single character in a guess"""
    char: str
    status: Literal["correct", "present", "absent"]


class BaseWordleGame(ABC):
    """Base class for Wordle games in different languages"""

    def __init__(self, answer: str, max_attempts: int = 6):
        self.answer = self.normalize_word(answer)
        self.max_attempts = max_attempts
        self.attempts = 0
        self.guesses = []
        self.won = False
        self.lost = False

    @abstractmethod
    def is_valid_guess(self, word: str) -> bool:
        """Check if the word is valid for guessing"""
        pass

    @abstractmethod
    def get_display_text(self, word: str) -> str:
        """Get display text for the word (e.g., with kanji/hanzi)"""
        pass

    @abstractmethod
    def normalize_word(self, word: str) -> str:
        """Normalize word (e.g., lowercase for English, no-op for others)"""
        pass

    def check_guess(self, guess: str) -> list[GuessResult]:
        """
        Check a guess against the answer using two-pass algorithm.

        First pass: mark correct positions
        Second pass: mark present/absent
        """
        guess = self.normalize_word(guess)
        results = []
        answer_chars = list(self.answer)

        # First pass: mark correct positions
        for i, char in enumerate(guess):
            if char == self.answer[i]:
                results.append(GuessResult(char, "correct"))
                answer_chars[i] = None
            else:
                results.append(GuessResult(char, ""))

        # Second pass: mark present/absent
        for i, result in enumerate(results):
            if result.status == "":
                char = guess[i]
                if char in answer_chars:
                    result.status = "present"
                    answer_chars[answer_chars.index(char)] = None
                else:
                    result.status = "absent"

        return results

    def _build_response(self, error_msg: str | None = None, guess: str | None = None,
                       results: list[GuessResult] | None = None) -> dict:
        """Build unified response format"""
        response = {
            "attempts": self.attempts,
            "attempts_remaining": self.max_attempts - self.attempts,
            "won": self.won,
            "lost": self.lost,
            "game_over": self.won or self.lost
        }

        if error_msg:
            response["error"] = error_msg

        if guess:
            response["guess"] = guess

        if results:
            response["results"] = [{"char": r.char, "status": r.status} for r in results]

        if self.won or self.lost:
            answer_display = self.get_display_text(self.answer)
            response["answer"] = answer_display

        return response

    def make_guess(self, word: str) -> dict:
        """
        Make a guess. Always consumes an attempt, even if invalid.

        Returns dict with game state and results.
        """
        if self.won or self.lost:
            return self._build_response(error_msg="Game is over. Use reset_game to start a new game.")

        # Consume attempt first
        self.attempts += 1

        # Validate input
        normalized = self.normalize_word(word)
        validation_error = None

        if len(normalized) != len(self.answer):
            validation_error = f"Word must be {len(self.answer)} characters"
        elif not self.is_valid_guess(normalized):
            validation_error = f"'{word}' is not in the word list"

        # If validation error, check if game is over
        if validation_error:
            if self.attempts >= self.max_attempts:
                self.lost = True
            return self._build_response(error_msg=validation_error)

        # Valid guess: perform normal game logic
        results = self.check_guess(normalized)
        guess_display = self.get_display_text(normalized)

        self.guesses.append({
            "word": normalized,
            "display": guess_display,
            "results": results
        })

        if normalized == self.answer:
            self.won = True
        elif self.attempts >= self.max_attempts:
            self.lost = True

        return self._build_response(
            guess=guess_display,
            results=results
        )
