"""Minesweeper game core logic."""

from dataclasses import dataclass
from enum import Enum
import random


class GameState(Enum):
    PLAYING = "playing"
    WON = "won"
    LOST = "lost"


@dataclass
class Cell:
    has_mine: bool = False
    revealed: bool = False
    flagged: bool = False
    adjacent_mines: int = 0


class MinesweeperGame:
    """Minesweeper game implementation with first-click protection."""

    def __init__(
        self,
        width: int,
        height: int,
        num_mines: int,
        seed: int | None = None,
        text_mode: bool = False,
        no_guess: bool = True,
        no_guess_max_attempts: int = 5000,
    ):
        if width < 1 or height < 1:
            raise ValueError("Width and height must be at least 1")
        if num_mines < 0:
            raise ValueError("Number of mines cannot be negative")
        if num_mines >= width * height:
            raise ValueError("Too many mines for the board size")

        self.width = width
        self.height = height
        self.num_mines = num_mines
        self.rng = random.Random(seed)
        self.state = GameState.PLAYING
        self.first_click = True
        self.text_mode = text_mode
        self.no_guess = no_guess
        self.no_guess_max_attempts = no_guess_max_attempts
        self.no_guess_attempts_used = 0  # populated after first click when no_guess=True

        # Initialize empty board (mines placed on first click)
        self.board: list[list[Cell]] = [
            [Cell() for _ in range(width)] for _ in range(height)
        ]

    def _get_neighbors(self, x: int, y: int) -> list[tuple[int, int]]:
        """Get valid neighbor coordinates."""
        neighbors = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    neighbors.append((nx, ny))
        return neighbors

    def _place_mines(self, avoid_x: int, avoid_y: int) -> None:
        """Place mines randomly, avoiding the first click position and its neighbors."""
        avoid = {(avoid_x, avoid_y)}
        avoid.update(self._get_neighbors(avoid_x, avoid_y))

        all_positions = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in avoid
        ]
        if len(all_positions) < self.num_mines:
            all_positions = [
                (x, y)
                for y in range(self.height)
                for x in range(self.width)
                if (x, y) != (avoid_x, avoid_y)
            ]

        if self.no_guess:
            mine_positions = self._sample_no_guess_mines(avoid_x, avoid_y, all_positions)
        else:
            mine_positions = self.rng.sample(all_positions, min(self.num_mines, len(all_positions)))

        for x, y in mine_positions:
            self.board[y][x].has_mine = True
        self._count_adjacent_mines()

    def _sample_no_guess_mines(self, fx: int, fy: int, all_positions: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Rejection-sample mine placements until the resulting board is logically solvable
        from the first click using single-cell + subset deductions only."""
        k = min(self.num_mines, len(all_positions))
        for attempt in range(1, self.no_guess_max_attempts + 1):
            mines = set(self.rng.sample(all_positions, k))
            if _is_solvable_no_guess(self.width, self.height, mines, fx, fy):
                self.no_guess_attempts_used = attempt
                return list(mines)
        raise RuntimeError(
            f"no-guess generation failed after {self.no_guess_max_attempts} attempts "
            f"for {self.width}x{self.height}/{self.num_mines} (mine density too high?)"
        )

    def _count_adjacent_mines(self) -> None:
        """Calculate adjacent mine count for each cell."""
        for y in range(self.height):
            for x in range(self.width):
                if self.board[y][x].has_mine:
                    continue
                count = sum(
                    1 for nx, ny in self._get_neighbors(x, y) if self.board[ny][nx].has_mine
                )
                self.board[y][x].adjacent_mines = count

    def _reveal_cascade(self, x: int, y: int) -> int:
        """Recursively reveal cells with no adjacent mines. Returns count of revealed cells."""
        stack = [(x, y)]
        revealed_count = 0
        while stack:
            cx, cy = stack.pop()
            cell = self.board[cy][cx]

            if cell.revealed or cell.flagged:
                continue

            cell.revealed = True
            revealed_count += 1

            # If no adjacent mines, reveal neighbors
            if cell.adjacent_mines == 0 and not cell.has_mine:
                for nx, ny in self._get_neighbors(cx, cy):
                    if not self.board[ny][nx].revealed:
                        stack.append((nx, ny))
        return revealed_count

    def _check_win(self) -> bool:
        """Check if all non-mine cells are revealed."""
        for row in self.board:
            for cell in row:
                if not cell.has_mine and not cell.revealed:
                    return False
        return True

    def reveal(self, x: int, y: int) -> dict | tuple[str, dict]:
        """Reveal a cell at the given coordinates."""
        # Validate coordinates
        if not (0 <= x < self.width and 0 <= y < self.height):
            return self._build_response(
                error=f"Coordinates ({x}, {y}) out of bounds. Valid range: x=0-{self.width-1}, y=0-{self.height-1}",
                action_info={"action": "reveal", "x": x, "y": y, "result": "error: out of bounds"},
            )

        # Check game state
        if self.state != GameState.PLAYING:
            return self._build_response(
                error=f"Game is over ({self.state.value}). Use reset_game to start a new game.",
                action_info={"action": "reveal", "x": x, "y": y, "result": "error: game over"},
            )

        cell = self.board[y][x]

        # Cannot reveal flagged cell
        if cell.flagged:
            return self._build_response(
                error=f"Cell ({x}, {y}) is flagged. Remove flag first.",
                action_info={"action": "reveal", "x": x, "y": y, "result": "error: cell flagged"},
            )

        # Already revealed
        if cell.revealed:
            return self._build_response(
                error=f"Cell ({x}, {y}) is already revealed.",
                action_info={"action": "reveal", "x": x, "y": y, "result": "error: already revealed"},
            )

        # First click: place mines
        if self.first_click:
            self._place_mines(x, y)
            self.first_click = False

        # Reveal the cell
        if cell.has_mine:
            cell.revealed = True
            self.state = GameState.LOST
            # Reveal all mines
            for row in self.board:
                for c in row:
                    if c.has_mine:
                        c.revealed = True
            return self._build_response(
                message="BOOM! You hit a mine!",
                action_info={"action": "reveal", "x": x, "y": y, "result": "BOOM!"},
            )
        else:
            cells_opened = self._reveal_cascade(x, y)
            if self._check_win():
                self.state = GameState.WON
                return self._build_response(
                    message="Congratulations! You won!",
                    action_info={"action": "reveal", "x": x, "y": y, "result": f"WIN! opened {cells_opened} cells"},
                )
            return self._build_response(
                action_info={"action": "reveal", "x": x, "y": y, "result": f"opened {cells_opened} cells"},
            )

    def toggle_flag(self, x: int, y: int) -> dict | tuple[str, dict]:
        """Toggle flag on a cell."""
        # Validate coordinates
        if not (0 <= x < self.width and 0 <= y < self.height):
            return self._build_response(
                error=f"Coordinates ({x}, {y}) out of bounds. Valid range: x=0-{self.width-1}, y=0-{self.height-1}",
                action_info={"action": "flag", "x": x, "y": y, "result": "error: out of bounds"},
            )

        # Check game state
        if self.state != GameState.PLAYING:
            return self._build_response(
                error=f"Game is over ({self.state.value}). Use reset_game to start a new game.",
                action_info={"action": "flag", "x": x, "y": y, "result": "error: game over"},
            )

        cell = self.board[y][x]

        # Cannot flag revealed cell
        if cell.revealed:
            return self._build_response(
                error=f"Cell ({x}, {y}) is already revealed.",
                action_info={"action": "flag", "x": x, "y": y, "result": "error: already revealed"},
            )

        cell.flagged = not cell.flagged
        result = "placed" if cell.flagged else "removed"
        return self._build_response(
            message=f"Flag {result} at ({x}, {y})",
            action_info={"action": "flag", "x": x, "y": y, "result": result},
        )

    def get_board_display(self, reveal_all: bool = False, show_coords: bool = True) -> str:
        """Generate ASCII representation of the board."""
        lines = []

        if show_coords:
            # Header with column numbers
            col_width = len(str(self.width - 1))
            header = "   " + " ".join(str(i).rjust(col_width) for i in range(self.width))
            lines.append(header)

        for y, row in enumerate(self.board):
            cells = []
            for cell in row:
                if reveal_all or cell.revealed:
                    if cell.has_mine:
                        cells.append("*")
                    elif cell.adjacent_mines == 0:
                        cells.append("_")
                    else:
                        cells.append(str(cell.adjacent_mines))
                elif cell.flagged:
                    cells.append("F")
                else:
                    cells.append(".")

            if show_coords:
                col_width = len(str(self.width - 1))
                row_str = str(y).rjust(2) + " " + " ".join(c.rjust(col_width) for c in cells)
            else:
                row_str = " ".join(cells)  # 符號間有空格
            lines.append(row_str)

        return "\n".join(lines)

    def _get_revealed_count(self) -> int:
        """Count revealed non-mine cells."""
        return sum(1 for row in self.board for cell in row if cell.revealed and not cell.has_mine)

    def _get_total_safe(self) -> int:
        """Total safe cells (non-mine)."""
        return self.width * self.height - self.num_mines

    def _build_response(
        self,
        message: str | None = None,
        error: str | None = None,
        action_info: dict | None = None,
    ) -> dict | tuple[str, dict]:
        """Build standardized response dictionary or (text, metadata) tuple.

        action_info: {"action": "reveal"|"flag", "x": int, "y": int, "result": str, "cells_opened": int}
        """
        flagged_count = sum(1 for row in self.board for cell in row if cell.flagged)
        game_over = self.state != GameState.PLAYING
        mines_remaining = self.num_mines - flagged_count
        revealed_count = self._get_revealed_count()
        total_safe = self._get_total_safe()

        if self.text_mode:
            # Text mode: return (text, metadata) tuple for ToolResult wrapping
            board = self.get_board_display(reveal_all=game_over, show_coords=False)

            # Build status line
            status_line = f"state: {self.state.value} | mines: {mines_remaining} | progress: {revealed_count}/{total_safe}"

            # Build action line
            action_line = ""
            if action_info:
                action = action_info.get("action", "")
                x = action_info.get("x", 0)
                y = action_info.get("y", 0)
                result = action_info.get("result", "")
                action_line = f"{action}({x},{y}) -> {result}"
            elif error:
                action_line = f"error: {error}"

            # Combine text
            text = f"{board}\n\n{status_line}"
            if action_line:
                text += f"\n{action_line}"

            metadata = {
                "state": self.state.value,
                "mines_remaining": mines_remaining,
                "revealed_count": revealed_count,
                "total_safe": total_safe,
                "game_over": game_over,
            }
            if message:
                metadata["message"] = message
            if error:
                metadata["error"] = error
            if game_over:
                metadata["early_exit"] = True
            return (text, metadata)

        # JSON mode: with coords
        response = {
            "board": self.get_board_display(reveal_all=game_over, show_coords=True),
            "state": self.state.value,
            "mines_remaining": mines_remaining,
            "game_over": game_over,
        }

        if message:
            response["message"] = message
        if error:
            response["error"] = error
        if game_over:
            response["early_exit"] = True

        return response


# ----------------------------------------------------------------------------
# No-guess solver: returns True iff the full board can be revealed using only
# single-cell + subset deductions starting from the first-click cascade.
# ----------------------------------------------------------------------------

def _neighbours(x: int, y: int, w: int, h: int):
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                yield (nx, ny)


def _is_solvable_no_guess(w: int, h: int, mines: set, fx: int, fy: int) -> bool:
    """A 'perfect logician' simulator: starts from cascade at (fx, fy), then
    iteratively applies (a) single-cell, (b) subset rules. Returns True if it
    eventually reveals every safe cell without ever needing to guess."""
    # adjacency counts for non-mine cells
    adj: dict[tuple[int, int], int] = {}
    for y in range(h):
        for x in range(w):
            if (x, y) in mines:
                continue
            adj[(x, y)] = sum(1 for nx, ny in _neighbours(x, y, w, h) if (nx, ny) in mines)

    revealed: set[tuple[int, int]] = set()
    flagged: set[tuple[int, int]] = set()      # cells the solver concludes are mines
    safe_total = w * h - len(mines)

    def reveal(x: int, y: int):
        # cascade-reveal at (x, y); caller guarantees (x, y) ∉ mines
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in revealed:
                continue
            revealed.add((cx, cy))
            if adj[(cx, cy)] == 0:
                for nx, ny in _neighbours(cx, cy, w, h):
                    if (nx, ny) not in mines and (nx, ny) not in revealed:
                        stack.append((nx, ny))

    reveal(fx, fy)

    while len(revealed) < safe_total:
        progress = False
        cons: list[tuple[frozenset, int]] = []

        # (a) single-cell rule. Snapshot `revealed` because reveal() mutates it.
        for (x, y) in list(revealed):
            n = adj[(x, y)]
            if n == 0:
                continue
            unrev = []
            kmine = 0
            for nx, ny in _neighbours(x, y, w, h):
                if (nx, ny) in flagged:
                    kmine += 1
                elif (nx, ny) in revealed:
                    pass
                else:
                    unrev.append((nx, ny))
            if not unrev:
                continue
            rem = n - kmine
            if rem == 0:
                for c in unrev:
                    if c not in revealed:
                        reveal(*c)
                        progress = True
            elif rem == len(unrev):
                for c in unrev:
                    if c not in flagged:
                        flagged.add(c)
                        progress = True
            else:
                cons.append((frozenset(unrev), rem))

        if progress:
            continue

        # (b) subset rule: if A's unrevealed-frontier ⊂ B's, deduce on B−A
        seen = set()
        for i, (uA, rA) in enumerate(cons):
            for j, (uB, rB) in enumerate(cons):
                if i == j:
                    continue
                key = (uA, uB)
                if key in seen:
                    continue
                seen.add(key)
                if uA < uB:
                    diff = uB - uA
                    drem = rB - rA
                    if drem == 0:
                        for c in diff:
                            if c not in revealed:
                                reveal(*c)
                                progress = True
                    elif drem == len(diff):
                        for c in diff:
                            if c not in flagged:
                                flagged.add(c)
                                progress = True

        if not progress:
            return False  # solver stuck → board needs a guess

    return True
