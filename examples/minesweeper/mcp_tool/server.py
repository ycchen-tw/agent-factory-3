"""Minesweeper MCP Server."""

import argparse
from textwrap import dedent
from typing import Annotated

from fastmcp import FastMCP
from fastmcp.tools.tool import ToolResult
from mcp.types import TextContent

from core import MinesweeperGame


def create_minesweeper_server(
    width: int = 9,
    height: int = 9,
    num_mines: int = 10,
    seed: int | None = None,
    text_mode: bool = False,
    no_guess: bool = True,
) -> FastMCP:
    """Create a Minesweeper MCP server instance."""

    game = MinesweeperGame(
        width=width, height=height, num_mines=num_mines, seed=seed,
        text_mode=text_mode, no_guess=no_guess,
    )

    mcp = FastMCP(
        name="minesweeper",
        instructions=dedent("""
            This is a Minesweeper game server.

            Goal: Reveal all cells without hitting a mine.

            Board symbols:
            - `.` = unrevealed cell
            - `F` = flagged cell (suspected mine)
            - `_` = revealed cell with 0 adjacent mines
            - `1-8` = revealed cell showing adjacent mine count
            - `*` = mine (shown when game ends)

            Coordinates are 0-indexed. X is horizontal (left to right), Y is vertical (top to bottom).
        """).strip(),
    )

    @mcp.tool(
        description=dedent("""
            Reveal a cell at the given coordinates.

            - First click is always safe (mines placed after)
            - If the cell has 0 adjacent mines, neighbors are revealed automatically
            - Revealing a mine ends the game

            Returns the updated board and game state.
        """).strip()
    )
    def reveal(
        x: Annotated[int, "X coordinate (0-indexed, left to right)"],
        y: Annotated[int, "Y coordinate (0-indexed, top to bottom)"],
    ) -> dict | ToolResult:
        """Reveal a cell at the given coordinates."""
        result = game.reveal(x, y)
        if text_mode:
            text, metadata = result
            return ToolResult(
                content=[TextContent(type="text", text=text)],
                structured_content=metadata,
            )
        return result

    @mcp.tool(
        description=dedent("""
            Toggle a flag on a cell to mark a suspected mine.

            - Flagged cells cannot be revealed until unflagged
            - Use flags to track known mine locations
            - Call again on the same cell to remove the flag

            Returns the updated board and remaining mine count.
        """).strip()
    )
    def flag(
        x: Annotated[int, "X coordinate (0-indexed, left to right)"],
        y: Annotated[int, "Y coordinate (0-indexed, top to bottom)"],
    ) -> dict | ToolResult:
        """Toggle flag on a cell."""
        result = game.toggle_flag(x, y)
        if text_mode:
            text, metadata = result
            return ToolResult(
                content=[TextContent(type="text", text=text)],
                structured_content=metadata,
            )
        return result

    @mcp.tool(
        description="Reset the game with new parameters.",
        annotations={"include_in_prompt": False},
    )
    def reset_game(
        width: Annotated[int, "Board width (default: 9)"] = 9,
        height: Annotated[int, "Board height (default: 9)"] = 9,
        mines: Annotated[int, "Number of mines (default: 10)"] = 10,
    ) -> dict | ToolResult:
        """Reset the game with new board parameters."""
        nonlocal game
        game = MinesweeperGame(
            width=width, height=height, num_mines=mines, text_mode=text_mode
        )

        if text_mode:
            text = game.get_board_display(show_coords=False)
            metadata = {
                "state": game.state.value,
                "mines_remaining": mines,
                "message": f"Game reset: {width}x{height} board with {mines} mines",
                "game_over": False,
            }
            return ToolResult(
                content=[TextContent(type="text", text=text)],
                structured_content=metadata,
            )

        return {
            "message": f"Game reset: {width}x{height} board with {mines} mines",
            "board": game.get_board_display(),
            "state": game.state.value,
            "mines_remaining": mines,
        }

    return mcp


def main():
    parser = argparse.ArgumentParser(description="Minesweeper MCP Server")
    parser.add_argument("--width", type=int, default=9, help="Board width (default: 9)")
    parser.add_argument("--height", type=int, default=9, help="Board height (default: 9)")
    parser.add_argument("--mines", type=int, default=10, help="Number of mines (default: 10)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument(
        "--mode",
        choices=["json", "text"],
        default="json",
        help="Output mode: json (with coords) or text (no coords, for RL)",
    )
    parser.add_argument("--log-level", default="WARNING", help="Logging level")
    parser.add_argument("--no-banner", action="store_true", help="Disable startup banner")
    parser.add_argument(
        "--allow-guess", action="store_true",
        help="Allow boards that may require guessing. Default: no-guess (boards are guaranteed "
             "solvable via single-cell + subset rules from the first click).",
    )

    args = parser.parse_args()

    mcp = create_minesweeper_server(
        width=args.width,
        height=args.height,
        num_mines=args.mines,
        seed=args.seed,
        text_mode=(args.mode == "text"),
        no_guess=not args.allow_guess,
    )

    mcp.run(show_banner=not args.no_banner, log_level=args.log_level)


if __name__ == "__main__":
    main()
