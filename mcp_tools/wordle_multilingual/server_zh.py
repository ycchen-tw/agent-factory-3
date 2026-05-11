"""
Chewing (Zhuyin) Wordle MCP Server
"""
import argparse
import json
import random
from pathlib import Path
from typing import Annotated
from textwrap import dedent

from fastmcp import FastMCP

from core import BaseWordleGame


class ChewingWordleGame(BaseWordleGame):
    """Chewing (Zhuyin) Wordle game implementation"""

    def __init__(self, answer: str, answer_dict: dict, valid_dict: dict, max_attempts: int = 6):
        self.answer_dict = answer_dict
        self.valid_dict = valid_dict
        super().__init__(answer, max_attempts)

    def is_valid_guess(self, word: str) -> bool:
        return word in self.valid_dict

    def get_chinese(self, zhuyin: str) -> str:
        """Get Chinese characters for zhuyin (show top 2 words)"""
        words = []
        if zhuyin in self.answer_dict:
            words = self.answer_dict[zhuyin]
        elif zhuyin in self.valid_dict:
            words = self.valid_dict[zhuyin]

        if not words:
            return ""

        # Show top 2 words separated by slash
        return "/".join(words[:2])

    def get_display_text(self, word: str) -> str:
        chinese = self.get_chinese(word)
        return f"{word} ({chinese})" if chinese else word

    def normalize_word(self, word: str) -> str:
        # Zhuyin doesn't need normalization
        return word


def load_json_dict(path: Path) -> dict:
    """Load JSON dictionary"""
    if not path.exists():
        raise FileNotFoundError(f"Dictionary not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_chewing_wordle_server(
    answers_path: Path,
    valid_path: Path,
    target_word: str | None = None,
    random_seed: int | None = None
) -> FastMCP:

    # Load dictionaries
    answer_dict = load_json_dict(answers_path)
    valid_dict = load_json_dict(valid_path)

    # Get all answer patterns
    answer_patterns = list(answer_dict.keys())

    # Select target zhuyin
    if target_word:
        target = target_word
        if target not in answer_dict:
            raise ValueError(f"Target word '{target_word}' not in answers list")
    else:
        if random_seed is not None:
            random.seed(random_seed)
        target = random.choice(answer_patterns)

    # Create game instance
    game = ChewingWordleGame(
        answer=target,
        answer_dict=answer_dict,
        valid_dict=valid_dict
    )

    # Create MCP server
    mcp = FastMCP(
        name="ChewingWordleGame",
        instructions=dedent("""
            注音符號版 Wordle 遊戲

            【遊戲規則】
            - 每局遊戲會隨機選擇一個由 5 個注音符號組成的詞彙
            - 注音符號不包含聲調標記（ˉˊˇˋ˙），只使用聲母和韻母
            - 對應的漢字數量不一定（例如「妖精」2字 = ㄧㄠㄐㄧㄥ，「一下子」3字 = ㄧㄒㄧㄚㄗ）
            - 猜測的注音必須是真實存在的詞彙（在詞庫中）
            - 你有 6 次猜測機會
            - 每次猜測後會得到提示：
              * 🟩 綠色 (correct)：注音符號正確且位置正確
              * 🟨 黃色 (present)：注音符號存在但位置錯誤
              * ⬜ 灰色 (absent)：注音符號不在答案中

            【遊戲範例】
            假設答案是：ㄧㄠㄐㄧㄥ (妖精) ← 兩個字

            猜測 1: ㄧㄒㄧㄚㄗ (一下子) ← 三個字
            回應: 🟩⬜🟨⬜⬜
            說明：
            - 第1個 ㄧ：位置正確 🟩
            - 第2個 ㄒ：不在答案中 ⬜
            - 第3個 ㄧ：存在但位置錯誤 🟨（答案中 ㄧ 在第1和第4位）
            - 第4個 ㄚ：不在答案中 ⬜
            - 第5個 ㄗ：不在答案中 ⬜

            猜測 2: ㄧㄠㄐㄧㄥ (妖精)
            回應: 🟩🟩🟩🟩🟩
            恭喜答對！

            請使用 guess 工具來猜測注音。
        """).strip()
    )

    @mcp.tool(
        description=dedent("""
            猜測一個由 5 個注音符號組成的詞彙。

            回應包含：
            - 每個注音符號的狀態（correct/present/absent）
            - 已使用和剩餘的猜測次數
            - 遊戲狀態（進行中/獲勝/失敗）
            - 如果遊戲結束會顯示答案
        """).strip()
    )
    def guess(zhuyin: Annotated[str, "5 個注音符號組成的詞彙（不含聲調）"]) -> dict:
        tool_result = game.make_guess(zhuyin)

        if tool_result.get("game_over"):
            tool_result["early_exit"] = True

        return tool_result

    @mcp.tool(
        description="重置遊戲，開始新的一局（可選擇性指定隨機種子）",
        annotations={"include_in_prompt": False},
    )
    def reset_game(seed: Annotated[int | None, "隨機種子（可選）"] = None) -> dict:
        nonlocal game

        if seed is not None:
            random.seed(seed)
        elif random_seed is not None:
            random.seed(random_seed)

        new_target = random.choice(answer_patterns)
        game = ChewingWordleGame(
            answer=new_target,
            answer_dict=answer_dict,
            valid_dict=valid_dict
        )

        return {
            "message": "遊戲已重置",
            "max_attempts": game.max_attempts
        }

    return mcp


def main():
    parser = argparse.ArgumentParser(description="注音符號 Wordle 遊戲 MCP Server")
    parser.add_argument(
        "--answers",
        type=Path,
        default=Path(__file__).parent / "data/chewing/answer_dict.json",
        help="答案字典檔案路徑（JSON 格式）"
    )
    parser.add_argument(
        "--valid",
        type=Path,
        default=Path(__file__).parent / "data/chewing/valid_dict.json",
        help="有效猜測字典檔案路徑（JSON 格式）"
    )
    parser.add_argument(
        "--target",
        type=str,
        help="指定目標注音（可選）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="隨機種子（可選）"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="設置日誌級別（預設：None）"
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="不顯示啟動橫幅"
    )

    args = parser.parse_args()

    mcp = create_chewing_wordle_server(
        answers_path=args.answers,
        valid_path=args.valid,
        target_word=args.target,
        random_seed=args.seed
    )

    mcp.run(
        show_banner=not args.no_banner,
        log_level=args.log_level
    )


if __name__ == "__main__":
    main()
