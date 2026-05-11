"""
English Wordle MCP Server
"""
import argparse
import random
from pathlib import Path
from typing import Annotated
from textwrap import dedent

from fastmcp import FastMCP

from core import BaseWordleGame


class EnglishWordleGame(BaseWordleGame):
    """English Wordle game implementation"""

    def __init__(self, answer: str, valid_guesses: set[str], max_attempts: int = 6):
        self.valid_guesses = {word.lower() for word in valid_guesses}
        super().__init__(answer, max_attempts)

    def is_valid_guess(self, word: str) -> bool:
        return word.lower() in self.valid_guesses

    def get_display_text(self, word: str) -> str:
        return word.lower()

    def normalize_word(self, word: str) -> str:
        return word.lower()


def load_word_list(path: Path) -> set[str]:
    """Load word list from text file"""
    if not path.exists():
        raise FileNotFoundError(f"Word list not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return {line.strip().lower() for line in f if line.strip()}


def create_wordle_server(
    answers_path: Path,
    guesses_path: Path,
    target_word: str | None = None,
    random_seed: int | None = None
) -> FastMCP:

    # Load word lists
    answers = load_word_list(answers_path)
    valid_guesses = load_word_list(guesses_path)

    # Select target word
    if target_word:
        target = target_word.lower()
        if target not in answers:
            raise ValueError(f"Target word '{target_word}' not in answers list")
    else:
        if random_seed is not None:
            random.seed(random_seed)
        target = random.choice(sorted(answers))

    # Create game instance
    game = EnglishWordleGame(answer=target, valid_guesses=valid_guesses | answers)

    # Create MCP server
    mcp = FastMCP(
        name="WordleGame",
        instructions=dedent(f"""
            This is a Wordle game server. The goal is to guess a 5-letter word in {game.max_attempts} attempts.

            Each guess returns:
            - 'correct': letter is in the word and in the correct position (green)
            - 'present': letter is in the word but in wrong position (yellow)
            - 'absent': letter is not in the word (gray)

            Use the guess tool to make your guesses.
        """).strip()
    )

    @mcp.tool(
        description=dedent("""
            Make a guess in the Wordle game.

            Returns the result of the guess including:
            - Letter-by-letter feedback (correct/present/absent)
            - Number of attempts used and remaining
            - Whether the game is won, lost, or still in progress
            - The answer if the game is over
        """).strip()
    )
    def guess(word: Annotated[str, "The 5-letter word to guess"]) -> dict:
        """Make a guess in the Wordle game."""
        tool_result = game.make_guess(word)

        if tool_result.get("game_over"):
            tool_result["early_exit"] = True

        return tool_result

    @mcp.tool(
        description="Reset the game with a new random word (using the same seed if provided).",
        annotations={"include_in_prompt": False},
    )
    def reset_game() -> dict:
        nonlocal game

        if random_seed is not None:
            random.seed(random_seed)
        new_target = random.choice(sorted(answers))
        game = EnglishWordleGame(answer=new_target, valid_guesses=valid_guesses | answers)

        return {
            "message": "Game reset successfully",
            "max_attempts": game.max_attempts
        }

    return mcp


def main():
    parser = argparse.ArgumentParser(description="Wordle Game MCP Server")
    parser.add_argument(
        "--answers",
        type=Path,
        default=Path(__file__).parent / "data/english/wordle-answers-alphabetical.txt",
        help="Path to the answers word list file"
    )
    parser.add_argument(
        "--guesses",
        type=Path,
        default=Path(__file__).parent / "data/english/wordle-allowed-guesses.txt",
        help="Path to the valid guesses word list file"
    )
    parser.add_argument(
        "--target",
        type=str,
        help="Specific target word (optional)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for target word selection (optional)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: None)"
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Disable the startup banner display"
    )

    args = parser.parse_args()

    mcp = create_wordle_server(
        answers_path=args.answers,
        guesses_path=args.guesses,
        target_word=args.target,
        random_seed=args.seed
    )

    mcp.run(
        show_banner=not args.no_banner,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()
