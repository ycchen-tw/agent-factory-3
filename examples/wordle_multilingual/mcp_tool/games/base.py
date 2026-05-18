"""Shared base for character-grid Wordle variants (english / chewing / japanese).

Handle does NOT inherit from this — its scoring is multi-axis (char + pinyin
initial/final/tone) and lives in handle.py with its own logic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Literal


Status = Literal["correct", "present", "absent"]


@dataclass
class GuessResult:
    char: str
    status: Status


class BaseWordleGame(ABC):
    """Single-axis Wordle: per-position symbol comparison with two-pass scoring.

    Behavior (matches the reference react-wordle template):
      - `attempts` is incremented on every call to make_guess, including
        invalid guesses (wrong length / not in valid list). This is the same
        contract examples/wordle/mcp_tool/server.py uses.
      - The game ends as soon as the answer is matched OR attempts == max.
      - The full answer is included in the response once the game is over.
    """

    word_length: int
    mode: str

    def __init__(self, answer: str, max_attempts: int = 6):
        self.answer = self.normalize(answer)
        if len(self.answer) != self.word_length:
            raise ValueError(
                f"{self.mode}: answer length {len(self.answer)} != expected {self.word_length}"
            )
        self.max_attempts = max_attempts
        self.attempts = 0
        self.guesses: list[dict] = []
        self.won = False
        self.lost = False

    # ── overrides ────────────────────────────────────────────────────────

    def normalize(self, word: str) -> str:
        return word

    @abstractmethod
    def is_valid_guess(self, word: str) -> bool: ...

    def display(self, word: str) -> str:
        """Human-readable rendering (e.g. add kanji/hanzi annotation)."""
        return word

    # ── core scoring (two-pass) ──────────────────────────────────────────

    def _split_chars(self, word: str) -> list[str]:
        """Mode-specific char iteration (default: per-unicode-codepoint)."""
        return list(word)

    def check(self, guess: str) -> list[GuessResult]:
        guess_chars = self._split_chars(guess)
        answer_chars = self._split_chars(self.answer)
        results: list[GuessResult] = []
        remaining: list[str | None] = list(answer_chars)

        # Pass 1: correct positions
        for i, ch in enumerate(guess_chars):
            if ch == answer_chars[i]:
                results.append(GuessResult(ch, "correct"))
                remaining[i] = None
            else:
                results.append(GuessResult(ch, "absent"))

        # Pass 2: present (found elsewhere) vs absent
        for i, r in enumerate(results):
            if r.status == "absent":
                ch = guess_chars[i]
                if ch in remaining:
                    r.status = "present"
                    remaining[remaining.index(ch)] = None
        return results

    # ── public API ───────────────────────────────────────────────────────

    def make_guess(self, word: str) -> dict:
        if self.won or self.lost:
            return self._response(error="Game is over.")

        self.attempts += 1
        guess = self.normalize(word.strip())

        guess_chars = self._split_chars(guess)
        if len(guess_chars) != self.word_length:
            return self._maybe_finalize(
                self._response(error=f"Word must be exactly {self.word_length} characters.")
            )

        if not self.is_valid_guess(guess):
            return self._maybe_finalize(
                self._response(error=f"'{word}' is not in the valid word list.")
            )

        results = self.check(guess)
        self.guesses.append({"guess": guess, "results": results})

        if guess == self.answer:
            self.won = True
        elif self.attempts >= self.max_attempts:
            self.lost = True

        return self._response(guess=guess, results=results)

    # ── response building ────────────────────────────────────────────────

    def _maybe_finalize(self, partial: dict) -> dict:
        """If we just consumed the last attempt on an invalid guess, mark lost."""
        if not self.won and not self.lost and self.attempts >= self.max_attempts:
            self.lost = True
            partial["lost"] = True
            partial["game_over"] = True
            partial["answer"] = self.display(self.answer)
        return partial

    def _response(
        self,
        *,
        guess: str | None = None,
        results: Iterable[GuessResult] | None = None,
        error: str | None = None,
    ) -> dict:
        out: dict = {
            "mode": self.mode,
            "attempts": self.attempts,
            "attempts_remaining": self.max_attempts - self.attempts,
            "won": self.won,
            "lost": self.lost,
            "game_over": self.won or self.lost,
        }
        if guess is not None:
            out["guess"] = self.display(guess)
        if results is not None:
            out["results"] = [{"char": r.char, "status": r.status} for r in results]
        if error is not None:
            out["error"] = error
        if self.won or self.lost:
            out["answer"] = self.display(self.answer)
        return out
