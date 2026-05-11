"""
Japanese (Hiragana) Wordle MCP Server
"""
import argparse
import json
import random
from pathlib import Path
from typing import Annotated
from textwrap import dedent

from fastmcp import FastMCP

from core import BaseWordleGame


class JapaneseWordleGame(BaseWordleGame):
    """Japanese (Hiragana) Wordle game implementation"""

    def __init__(self, answer: str, answer_dict: dict, valid_dict: dict, max_attempts: int = 6):
        self.answer_dict = answer_dict
        self.valid_dict = valid_dict
        super().__init__(answer, max_attempts)

    def is_valid_guess(self, word: str) -> bool:
        return word in self.valid_dict

    def get_kanji(self, kana: str) -> str:
        """Get kanji for hiragana (show top 2 words)"""
        words = []
        if kana in self.answer_dict:
            words = self.answer_dict[kana]
        elif kana in self.valid_dict:
            words = self.valid_dict[kana]

        if not words:
            return ""

        # Show top 2 words separated by slash
        return "/".join(words[:2])

    def get_display_text(self, word: str) -> str:
        kanji = self.get_kanji(word)
        return f"{word} ({kanji})" if kanji else word

    def normalize_word(self, word: str) -> str:
        # Hiragana doesn't need normalization
        return word


def load_json_dict(path: Path) -> dict:
    """Load JSON dictionary"""
    if not path.exists():
        raise FileNotFoundError(f"Dictionary not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_japanese_wordle_server(
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

    # Select target hiragana
    if target_word:
        target = target_word
        if target not in answer_dict:
            raise ValueError(f"Target word '{target_word}' not in answers list")
    else:
        if random_seed is not None:
            random.seed(random_seed)
        target = random.choice(answer_patterns)

    # Create game instance
    game = JapaneseWordleGame(
        answer=target,
        answer_dict=answer_dict,
        valid_dict=valid_dict
    )

    # Create MCP server
    mcp = FastMCP(
        name="JapaneseWordleGame",
        instructions=dedent("""
            ひらがな版 Wordle ゲーム

            【ゲームルール】
            - 毎回、5文字のひらがなで構成された単語を当てます
            - 推測する単語は実在の語彙である必要があります（辞書に存在）
            - 6回まで推測できます
            - 各推測後、以下のヒントが表示されます：
              * 🟩 緑 (correct)：文字が正しく、位置も正しい
              * 🟨 黄 (present)：文字は存在するが、位置が違う
              * ⬜ 灰 (absent)：文字が答えに存在しない

            【ゲーム例】
            答え：けんきゅう (研究)

            推測 1: こうじょう (工場)
            結果: ⬜⬜⬜⬜🟩
            説明：最後の「う」だけ位置が正しい

            推測 2: きょうかい (教会)
            結果: 🟨⬜🟨⬜⬜
            説明：「き」と「う」は存在するが位置が違う

            推測 3: けんきゅう (研究)
            結果: 🟩🟩🟩🟩🟩
            正解！

            guess ツールを使って推測してください。
        """).strip()
    )

    @mcp.tool(
        description=dedent("""
            5文字のひらがなで構成された単語を推測します。

            応答には以下が含まれます：
            - 各文字の状態（correct/present/absent）
            - 使用済み・残りの推測回数
            - ゲーム状態（進行中/勝利/敗北）
            - ゲーム終了時には答えが表示されます
        """).strip()
    )
    def guess(kana: Annotated[str, "5文字のひらがなで構成された単語"]) -> dict:
        tool_result = game.make_guess(kana)

        if tool_result.get("game_over"):
            tool_result["early_exit"] = True

        return tool_result

    @mcp.tool(
        description="ゲームをリセットし、新しいゲームを開始します（オプション：シード値を指定可能）",
        annotations={"include_in_prompt": False},
    )
    def reset_game(seed: Annotated[int | None, "ランダムシード（オプション）"] = None) -> dict:
        nonlocal game

        if seed is not None:
            random.seed(seed)
        elif random_seed is not None:
            random.seed(random_seed)

        new_target = random.choice(answer_patterns)
        game = JapaneseWordleGame(
            answer=new_target,
            answer_dict=answer_dict,
            valid_dict=valid_dict
        )

        return {
            "message": "ゲームがリセットされました",
            "max_attempts": game.max_attempts
        }

    return mcp


def main():
    parser = argparse.ArgumentParser(description="ひらがな Wordle ゲーム MCP Server")
    parser.add_argument(
        "--answers",
        type=Path,
        default=Path(__file__).parent / "data/japanese/answer_dict.json",
        help="答え辞書ファイルパス（JSON形式）"
    )
    parser.add_argument(
        "--valid",
        type=Path,
        default=Path(__file__).parent / "data/japanese/valid_dict.json",
        help="有効な推測辞書ファイルパス（JSON形式）"
    )
    parser.add_argument(
        "--target",
        type=str,
        help="目標単語を指定（オプション）"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="ランダムシード（オプション）"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="ログレベルを設定（デフォルト：None）"
    )
    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="起動バナーを非表示"
    )

    args = parser.parse_args()

    mcp = create_japanese_wordle_server(
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
