"""English Wordle MCP server.

A self-contained 5-letter Wordle game exposed as a single MCP tool. Each
training rollout spawns one server with a fixed target word via --target.

Usage:
    python server.py --target apple --no-banner
"""
import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Annotated, Literal

from fastmcp import FastMCP

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_ANSWERS = DATA_DIR / "answers.txt"
DEFAULT_ALLOWED = DATA_DIR / "allowed_guesses.txt"
MAX_ATTEMPTS = 6
WORD_LENGTH = 5


@dataclass
class GuessResult:
    char: str
    status: Literal["correct", "present", "absent"]


class WordleGame:
    def __init__(self, answer: str, valid_guesses: set[str]):
        self.answer = answer.lower()
        self.valid_guesses = {w.lower() for w in valid_guesses}
        self.attempts = 0
        self.won = False
        self.lost = False

    def _check(self, guess: str) -> list[GuessResult]:
        """Two-pass check: correct positions first, then present/absent."""
        results: list[GuessResult] = []
        remaining = list(self.answer)
        for i, ch in enumerate(guess):
            if ch == self.answer[i]:
                results.append(GuessResult(ch, "correct"))
                remaining[i] = ""
            else:
                results.append(GuessResult(ch, "absent"))
        for i, r in enumerate(results):
            if r.status == "absent" and guess[i] in remaining:
                r.status = "present"
                remaining[remaining.index(guess[i])] = ""
        return results

    def make_guess(self, word: str) -> dict:
        if self.won or self.lost:
            return self._response(error="Game is over.")

        self.attempts += 1
        guess = word.lower().strip()

        if len(guess) != WORD_LENGTH:
            error = f"Word must be exactly {WORD_LENGTH} letters."
            if self.attempts >= MAX_ATTEMPTS:
                self.lost = True
            return self._response(error=error)

        if guess not in self.valid_guesses:
            error = f"'{word}' is not in the allowed word list."
            if self.attempts >= MAX_ATTEMPTS:
                self.lost = True
            return self._response(error=error)

        results = self._check(guess)
        if guess == self.answer:
            self.won = True
        elif self.attempts >= MAX_ATTEMPTS:
            self.lost = True

        return self._response(guess=guess, results=results)

    def _response(self, *, guess: str | None = None,
                  results: list[GuessResult] | None = None,
                  error: str | None = None) -> dict:
        out: dict = {
            "attempts": self.attempts,
            "attempts_remaining": MAX_ATTEMPTS - self.attempts,
            "won": self.won,
            "lost": self.lost,
            "game_over": self.won or self.lost,
        }
        if guess is not None:
            out["guess"] = guess
        if results is not None:
            out["results"] = [{"char": r.char, "status": r.status} for r in results]
        if error is not None:
            out["error"] = error
        if self.won or self.lost:
            out["answer"] = self.answer
        return out


def load_words(path: Path) -> set[str]:
    return {line.strip().lower() for line in path.read_text().splitlines() if line.strip()}


def build_server(target: str, answers_path: Path, allowed_path: Path) -> FastMCP:
    answers = load_words(answers_path)
    allowed = load_words(allowed_path) | answers

    target = target.lower()
    if target not in answers:
        raise ValueError(f"target '{target}' not in answers list")

    game = WordleGame(answer=target, valid_guesses=allowed)
    mcp = FastMCP(
        name="WordleGame",
        instructions=dedent(f"""
            Wordle: guess a 5-letter English word in {MAX_ATTEMPTS} attempts.

            Each guess returns per-letter feedback:
              - 'correct': letter is in the word AND in the right position
              - 'present': letter is in the word but in the WRONG position
              - 'absent':  letter is NOT in the word

            Use 'guess' to make a guess. Game ends on win or after {MAX_ATTEMPTS} attempts.
        """).strip(),
    )

    @mcp.tool(description="Make a Wordle guess. Returns per-letter feedback and game state.")
    def guess(word: Annotated[str, "The 5-letter word to guess"]) -> dict:
        result = game.make_guess(word)
        if result.get("game_over"):
            result["early_exit"] = True   # tells the runner: this rollout can stop now
        return result

    return mcp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="Target word for this game")
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--allowed", type=Path, default=DEFAULT_ALLOWED)
    parser.add_argument("--no-banner", action="store_true")
    args = parser.parse_args()

    mcp = build_server(args.target, args.answers, args.allowed)
    mcp.run(show_banner=not args.no_banner)


if __name__ == "__main__":
    main()
