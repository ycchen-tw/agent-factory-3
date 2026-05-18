"""Handle (汉兜) — Chinese idiom Wordle with 4-axis scoring.

Mirrors antfu/handle's logic (src/logic/utils.ts):
  per character position, return four independent statuses
    - char_status   : the hanzi itself
    - initial_status: pinyin initial (ㄓㄔㄕ / zh/ch/sh and single consonants)
    - final_status  : pinyin final
    - tone_status   : tone number (1/2/3/4, 0 = neutral)

Each axis uses an unmatched-pool 'present' rule (same two-pass idea as classic
Wordle, but applied independently per axis instead of per char).

The hanzi axis additionally implements handle's "pinyin override": when the
guess hanzi happens to appear in the answer, the guess's pinyin for that
position is forced to the answer's pinyin — so a polyphone in a guessing
context isn't penalized for picking the "wrong" reading.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pypinyin import lazy_pinyin
from pypinyin import Style as PinyinStyle


Status = Literal["correct", "present", "absent"]

# Order matters: longer prefixes first.
PINYIN_INITIALS = (
    "zh", "ch", "sh",
    "b", "p", "m", "f", "d", "t", "n", "l",
    "g", "k", "h", "j", "q", "x",
    "r", "z", "c", "s",
    "y", "w",
)


def _split_pinyin(syllable: str) -> tuple[str, str, int]:
    """Split 'lu4' → (initial='l', final='u', tone=4).

    Returns tone=0 for neutral / unknown.
    Mirrors handle's behavior where syllables with only an initial-shaped
    body (e.g. 'n', 'm', 'ng') are treated as no-initial + final.
    """
    s = syllable.strip()
    tone = 0
    if s and s[-1].isdigit():
        tone = int(s[-1])
        s = s[:-1]
    initial = ""
    for prefix in PINYIN_INITIALS:
        if s.startswith(prefix):
            initial = prefix
            break
    final = s[len(initial):]
    # No final but an initial-shaped body → flip into final slot.
    if initial and not final:
        return ("", initial, tone)
    return (initial, final, tone)


@dataclass
class ParsedChar:
    char: str
    initial: str
    final: str
    tone: int

    @property
    def parts(self) -> tuple[str, str]:
        return (self.initial, self.final)


def _idiom_pinyin(idiom: str, polyphones: dict[str, str]) -> list[str]:
    """Get per-character pinyin syllables (TONE3 style: 'lu4', 'yi1')."""
    if idiom in polyphones:
        return polyphones[idiom].split()
    return lazy_pinyin(
        idiom,
        style=PinyinStyle.TONE3,
        neutral_tone_with_five=False,
    )


def _parse_idiom(
    idiom: str,
    polyphones: dict[str, str],
    answer: str | None = None,
    answer_parsed: list[ParsedChar] | None = None,
) -> list[ParsedChar]:
    """Parse an idiom into ParsedChars. When `answer` is provided, override
    the guess's pinyin with the answer's pinyin for any char that appears in
    the answer (handles polyphones gracefully).
    """
    pinyins = _idiom_pinyin(idiom, polyphones)
    out: list[ParsedChar] = []
    for i, ch in enumerate(idiom):
        py = pinyins[i] if i < len(pinyins) else ""
        if answer and answer_parsed and ch in answer:
            j = answer.index(ch)
            py_from_answer = f"{answer_parsed[j].initial}{answer_parsed[j].final}{answer_parsed[j].tone or ''}"
            py = py_from_answer
        ini, fin, tone = _split_pinyin(py)
        out.append(ParsedChar(char=ch, initial=ini, final=fin, tone=tone))
    return out


def _grade_axis(
    guess_vals: list,
    answer_vals: list,
    *,
    parts_per_position: list[tuple[str, ...]] | None = None,
) -> list[Status]:
    """Two-pass scoring on an arbitrary axis.

    parts_per_position: if set, position i is considered 'correct' when the
    guess value is in answer's parts tuple at position i (not just equal).
    This matches handle's _1/_2 (initial/final) rule, where an initial 'b'
    is treated 'exact' if 'b' appears anywhere in answer[i].parts.
    """
    n = len(answer_vals)
    statuses: list[Status] = ["absent"] * n
    unmatched: list = []

    def is_position_match(i: int) -> bool:
        if parts_per_position is not None:
            # exact if guess[i] is empty OR appears in answer[i]'s parts
            return (not guess_vals[i]) or (guess_vals[i] in parts_per_position[i])
        return guess_vals[i] == answer_vals[i]

    # Pass 1: position-correct
    for i in range(n):
        if is_position_match(i):
            statuses[i] = "correct"
        else:
            unmatched.append(answer_vals[i])

    # Pass 2: present
    for i in range(n):
        if statuses[i] == "correct":
            continue
        g = guess_vals[i]
        if g in unmatched:
            unmatched.remove(g)
            statuses[i] = "present"
    return statuses


class HandleWordleGame:
    """Stand-alone game class with the same `make_guess` interface as
    BaseWordleGame, but emits richer per-char results.
    """

    word_length = 4
    mode = "handle"

    def __init__(
        self,
        answer: str,
        valid_guesses: set[str],
        polyphones: dict[str, str],
        max_attempts: int = 10,
    ):
        if len(answer) != self.word_length:
            raise ValueError(f"handle: answer must be {self.word_length} hanzi, got {answer!r}")
        self.answer = answer
        self.valid_guesses = set(valid_guesses)
        self.polyphones = polyphones
        self.max_attempts = max_attempts
        self.attempts = 0
        self.guesses: list[dict] = []
        self.won = False
        self.lost = False
        # Pre-parse the answer once.
        self._answer_parsed = _parse_idiom(answer, polyphones)

    # ── public API (mirrors BaseWordleGame) ───────────────────────────────

    def is_valid_guess(self, word: str) -> bool:
        return word in self.valid_guesses

    def display(self, word: str) -> str:
        return word

    def make_guess(self, word: str) -> dict:
        if self.won or self.lost:
            return self._response(error="Game is over.")

        self.attempts += 1
        guess = word.strip()

        if len(guess) != self.word_length:
            return self._maybe_finalize(
                self._response(error=f"Idiom must be exactly {self.word_length} hanzi.")
            )

        if not self.is_valid_guess(guess):
            return self._maybe_finalize(
                self._response(error=f"'{word}' is not a recognized 4-character idiom.")
            )

        char_results = self._grade(guess)
        self.guesses.append({"guess": guess, "char_results": char_results})

        if guess == self.answer:
            self.won = True
        elif self.attempts >= self.max_attempts:
            self.lost = True

        return self._response(guess=guess, char_results=char_results)

    # ── 4-axis grading ────────────────────────────────────────────────────

    def _grade(self, guess: str) -> list[dict]:
        gp = _parse_idiom(guess, self.polyphones, answer=self.answer, answer_parsed=self._answer_parsed)
        ap = self._answer_parsed

        char_status = _grade_axis(
            [c.char for c in gp],
            [c.char for c in ap],
        )
        tone_status = _grade_axis(
            [c.tone for c in gp],
            [c.tone for c in ap],
        )

        # For initial / final: 'exact' rule references answer[i].parts pool.
        parts_per_position = [c.parts for c in ap]
        # Flat answer parts pool (used for "present" axis on initial/final;
        # we evaluate initial and final independently, each axis maintaining
        # its own unmatched pool to mirror handle's logic.)
        initial_status = _grade_axis(
            [c.initial for c in gp],
            [c.initial for c in ap],
            parts_per_position=parts_per_position,
        )
        final_status = _grade_axis(
            [c.final for c in gp],
            [c.final for c in ap],
            parts_per_position=parts_per_position,
        )

        out = []
        for i, g in enumerate(gp):
            out.append({
                "char": g.char,
                "char_status": char_status[i],
                "initial": g.initial,
                "initial_status": initial_status[i],
                "final": g.final,
                "final_status": final_status[i],
                "tone": g.tone,
                "tone_status": tone_status[i],
            })
        return out

    # ── response building ─────────────────────────────────────────────────

    def _maybe_finalize(self, partial: dict) -> dict:
        if not self.won and not self.lost and self.attempts >= self.max_attempts:
            self.lost = True
            partial["lost"] = True
            partial["game_over"] = True
            partial["answer"] = self.answer
        return partial

    def _response(
        self,
        *,
        guess: str | None = None,
        char_results: list[dict] | None = None,
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
            out["guess"] = guess
        if char_results is not None:
            out["char_results"] = char_results
        if error is not None:
            out["error"] = error
        if self.won or self.lost:
            out["answer"] = self.answer
        return out
