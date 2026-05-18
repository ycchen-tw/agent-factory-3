"""Wordle game implementations for 4 modes: english / chewing / japanese / handle."""
from .base import BaseWordleGame, GuessResult
from .english import EnglishWordleGame
from .chewing import ChewingWordleGame
from .japanese import JapaneseWordleGame
from .handle import HandleWordleGame

GAME_CLASSES = {
    "english": EnglishWordleGame,
    "chewing": ChewingWordleGame,
    "japanese": JapaneseWordleGame,
    "handle": HandleWordleGame,
}

__all__ = [
    "BaseWordleGame",
    "GuessResult",
    "EnglishWordleGame",
    "ChewingWordleGame",
    "JapaneseWordleGame",
    "HandleWordleGame",
    "GAME_CLASSES",
]
