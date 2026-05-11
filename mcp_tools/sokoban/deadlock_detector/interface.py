# interface.py
from typing import Tuple, List
from .state import SokobanState          # repo module
from .deadlock import DeadlockDetector   # repo module

# ────────────────────────────────────────────────────────────────
# Public API
# ----------------------------------------------------------------
def check_deadlock(ascii_board: str) -> Tuple[bool, List[str]]:
    """
    Convert an ASCII Sokoban board that uses the symbols

        X  : wall
        .  : empty floor
        B  : box
        @  : player
        T  : target
        *  : box on target

    into the internal grid-format expected by `SokobanState`, then run
    the repo’s `DeadlockDetector`.

    Parameters
    ----------
    ascii_board : str
        Multiline string. Each row may be separated by spaces for readability.

    Returns
    -------
    Tuple[bool, List[str]]
        * first element  : `True` if the position is provably unsolvable  
        * second element : list of detector-reported reasons (empty if solvable)
    """

    # 1) normalise the layout ────────────────────────────────────
    char_map = {
        'X': SokobanState.WALL,          # '#'
        '.': SokobanState.FLOOR,         # ' '
        'B': SokobanState.BOX,           # '$'
        '@': SokobanState.PLAYER,        # '@'
        'T': SokobanState.GOAL,          # '.'
        '*': SokobanState.BOX_ON_GOAL,   # '*'
        '+': SokobanState.PLAYER_ON_GOAL, # '+'
    }

    grid: List[str] = [
        ''.join(char_map[ch] for ch in row.split())
        for row in ascii_board.strip().splitlines()
        if row.strip()                       # skip blank rows
    ]

    # 2) build state & detector ───────────────────────────────────
    state     = SokobanState(grid)           # uses constants above :contentReference[oaicite:0]{index=0}
    detector  = DeadlockDetector(state)      # repo’s deadlock module :contentReference[oaicite:1]{index=1}

    # 3) run detection ────────────────────────────────────────────
    return detector.detect_deadlocks()


# ────────────────────────────────────────────────────────────────
# Example (remove or guard with `if __name__ == "__main__":`)
# ----------------------------------------------------------------
if __name__ == "__main__":
    TEST_LEVEL = """
    X X X X X X X
    X X X X X X X
    X X X X X X X
    X X X . . . X
    X . B . T . X
    X @ B . . T X
    X X X X X X X
    """
    deadlock, reasons = check_deadlock(TEST_LEVEL)
    print("Deadlock?", deadlock)
    for r in reasons:
        print("  •", r)