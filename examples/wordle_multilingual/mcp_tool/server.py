"""Unified multilingual Wordle MCP server.

Single entry point covering 4 modes:
  english   — 5-letter words, 6 attempts            (cwackerfuss-lineage)
  chewing   — 5 bopomofo (zhuyin), 6 attempts       (wordshk lineage)
  japanese  — 5 katakana, 6 attempts                (tango lineage)
  handle    — 4-hanzi idiom, 10 attempts            (antfu/handle, 4-axis grading)

Each training rollout spawns its own process with a fixed target, e.g.:

    python server.py --mode english  --target apple
    python server.py --mode chewing  --target ㄧㄠㄐㄧㄥ
    python server.py --mode japanese --target ジュウジロ
    python server.py --mode handle   --target 路不拾遗

The tool exposed is always named `guess` — language-specific instructions
are baked into the FastMCP server's `instructions` field per mode.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import dedent
from typing import Annotated

from fastmcp import FastMCP

from games import GAME_CLASSES, BaseWordleGame
from games.handle import HandleWordleGame


DATA_DIR = Path(__file__).parent / "data"

DEFAULT_MAX_ATTEMPTS = {
    "english": 6,
    "chewing": 6,
    "japanese": 10,   # ことのはたんご original: 10 tries (日本語の語彙難度を考慮)
    "handle": 10,
}

# ── Per-mode loaders ─────────────────────────────────────────────────────


def _load_lines(path: Path) -> set[str]:
    return {ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_english(target: str, max_attempts: int):
    cls = GAME_CLASSES["english"]
    valid = _load_lines(DATA_DIR / "english" / "valid.txt")
    answers = _load_lines(DATA_DIR / "english" / "answers.txt")
    target = target.lower()
    if target not in answers:
        raise ValueError(f"english: target '{target}' not in answers list")
    return cls(answer=target, valid_guesses=valid | answers, max_attempts=max_attempts)


def _build_chewing(target: str, max_attempts: int):
    cls = GAME_CLASSES["chewing"]
    valid = _load_lines(DATA_DIR / "chewing" / "valid.txt")
    answers = _load_lines(DATA_DIR / "chewing" / "answers.txt")
    display = _load_json(DATA_DIR / "chewing" / "display.json")
    if target not in answers:
        raise ValueError(f"chewing: target '{target}' not in answers list")
    return cls(answer=target, valid_guesses=valid | answers, display_map=display, max_attempts=max_attempts)


def _build_japanese(target: str, max_attempts: int):
    cls = GAME_CLASSES["japanese"]
    valid = _load_lines(DATA_DIR / "japanese" / "valid.txt")
    answers = _load_lines(DATA_DIR / "japanese" / "answers.txt")
    display = _load_json(DATA_DIR / "japanese" / "display.json")
    if target not in answers:
        raise ValueError(f"japanese: target '{target}' not in answers list")
    return cls(answer=target, valid_guesses=valid | answers, display_map=display, max_attempts=max_attempts)


def _build_handle(target: str, max_attempts: int):
    valid = _load_lines(DATA_DIR / "handle" / "valid.txt")
    answers = _load_lines(DATA_DIR / "handle" / "answers.txt")
    polyphones = _load_json(DATA_DIR / "handle" / "polyphones.json")
    if target not in answers:
        raise ValueError(f"handle: target '{target}' not in answers list")
    return HandleWordleGame(
        answer=target,
        valid_guesses=valid | answers,
        polyphones=polyphones,
        max_attempts=max_attempts,
    )


_BUILDERS = {
    "english": _build_english,
    "chewing": _build_chewing,
    "japanese": _build_japanese,
    "handle": _build_handle,
}

# ── Per-mode instructions / tool docs ────────────────────────────────────


def _instructions(mode: str, max_attempts: int) -> str:
    if mode == "english":
        return dedent(f"""
            English Wordle: guess a 5-letter English word in {max_attempts} attempts.

            Each guess returns per-letter feedback:
              - 'correct': letter is in the word AND in the right position
              - 'present': letter is in the word but in the WRONG position
              - 'absent' : letter is NOT in the word

            Invalid input (wrong length / not in word list) still consumes one attempt.
        """).strip()
    if mode == "chewing":
        return dedent(f"""
            注音符號 Wordle：用 5 個注音符號猜出對應的中文詞（{max_attempts} 次機會）。

            注音符號不含聲調標記（ˉˊˇˋ˙）。猜測必須是真實存在的詞彙。
            每次猜測後，每個注音符號會收到提示：
              - 'correct'：注音正確且位置正確
              - 'present'：注音存在但位置錯誤
              - 'absent' ：注音不在答案中

            例（教學示意；實際每局答案由 server 端從答案表抽取）：
            若答案是 ㄧㄠㄐㄧㄥ（妖精），猜 ㄧㄒㄧㄚㄗ（一下子）→ 🟩⬜🟨⬜⬜。
            無效輸入（長度錯誤 / 詞庫沒有）也會消耗一次嘗試。
        """).strip()
    if mode == "japanese":
        return dedent(f"""
            カタカナ Wordle（ことのはたんご系）：5 文字のカタカナで、辞書に存在する
            日本語の単語（主に名詞）を当ててください。{max_attempts} 回まで挑戦できます。

            【入力規則】
            - 入力は 5 文字。カタカナを基本としますが、ひらがな入力も受け付け、
              内部で自動的にカタカナへ変換して評価します（例：「じゅうじろ」→「ジュウジロ」）。
            - 小書き仮名（ャ ュ ョ ッ）、長音符（ー）、濁点/半濁点付き（ガギグ／パピプ 等）は
              それぞれ独立した 1 文字として扱われます。
              例：「ジュウジロ」= ジ + ュ + ウ + ジ + ロ（5 文字）
            - 漢字・ローマ字・記号は無効。
            - 出題・入力ともに概ね名詞。動詞や形容詞は辞書に含まれません。
            - 日本語の語彙は膨大なため、実在する単語でも辞書に未登録の場合があります。
              辞書外の入力は「無効」として 1 回の試行を消費します。
            - 答えには外来語や、同じ文字が複数回現れる単語も含まれます。

            【フィードバック】各文字位置について以下のいずれか：
              - 'correct'：文字が正しく、位置も正しい   🟩
              - 'present'：文字は存在するが、位置が違う 🟨
              - 'absent' ：文字が答えに存在しない        ⬜

            【重複文字のルール】
            推測に同じ文字を 2〜3 個使った場合、その文字が答えに含まれる個数までしか
            「correct / present」が割り当てられません。「correct」（位置一致）を優先し、
            残った個数を左から順に「present」として埋め、それ以上は「absent」になります。

              例 1：答え「シュウカイ」（先頭に「シ」1 個）、推測「シシオドシ」
                → シ🟩 シ⬜ オ⬜ ド⬜ シ⬜
                  位置 0 で「シ」が確定 → 残りの「シ」は答えに 0 個 → absent。

              例 2：答え「スソナオシ」（末尾に「シ」1 個）、推測「シシオドシ」
                → シ⬜ シ⬜ オ🟨 ド⬜ シ🟩
                  位置 4 で「シ」が確定。他の「シ」は割り当て先がなく absent。

            【総合例】答え「キョウシツ」（教室）、推測「キュウツイ」（急追）
              → キ🟩 ョ⬜ ウ🟩 シ⬜ ツ🟨

            日本語の特性上、英語版 Wordle より難易度はかなり高めです。
        """).strip()
    if mode == "handle":
        return dedent(f"""
            漢兜 Handle：用 4 個漢字組成的成語猜出答案（{max_attempts} 次機會）。

            每次猜測對每個漢字位置回傳 **4 個獨立評分**：
              - char_status   ：漢字本身（correct / present / absent）
              - initial_status：拼音聲母
              - final_status  ：拼音韻母
              - tone_status   ：聲調（1/2/3/4，0 表示輕聲）

            'present' 表示「該成分存在於答案某處，但這個位置不對」，'absent' 表示
            完全不存在。即使漢字錯了，聲母/韻母/聲調仍可能命中。

            無效輸入（非 4 字 / 不在成語表）也會消耗一次嘗試。
        """).strip()
    raise ValueError(f"unknown mode {mode!r}")


def _tool_description(mode: str) -> str:
    if mode == "english":
        return "Submit a 5-letter English word. Returns per-letter feedback and game state."
    if mode == "chewing":
        return "Submit a 5-symbol bopomofo string. Returns per-symbol feedback and game state."
    if mode == "japanese":
        return "Submit a 5-character kana string. Returns per-kana feedback and game state."
    if mode == "handle":
        return "Submit a 4-hanzi Chinese idiom. Returns per-char 4-axis (char/initial/final/tone) feedback."
    raise ValueError(f"unknown mode {mode!r}")


_INPUT_LABEL = {
    "english":  "The 5-letter English word to guess.",
    "chewing":  "The 5 bopomofo symbols to guess (no tone marks).",
    "japanese": "The 5-character kana word to guess.",
    "handle":   "The 4-character Chinese idiom to guess.",
}


# ── Server construction ─────────────────────────────────────────────────


def build_server(mode: str, target: str, max_attempts: int | None = None) -> FastMCP:
    if mode not in _BUILDERS:
        raise ValueError(f"unknown mode '{mode}', expected one of {list(_BUILDERS)}")
    if max_attempts is None:
        max_attempts = DEFAULT_MAX_ATTEMPTS[mode]

    game = _BUILDERS[mode](target, max_attempts)

    mcp = FastMCP(
        name=f"WordleGame[{mode}]",
        instructions=_instructions(mode, max_attempts),
    )

    tool_doc = f"{_tool_description(mode)}\n\nInput: {_INPUT_LABEL[mode]}"

    @mcp.tool(description=tool_doc)
    def guess(word: Annotated[str, "The word/idiom to guess; see tool description for the exact format."]) -> dict:
        result = game.make_guess(word)
        if result.get("game_over"):
            result["early_exit"] = True
        return result

    return mcp


def main():
    parser = argparse.ArgumentParser(description="Unified multilingual Wordle MCP server")
    parser.add_argument("--mode", required=True, choices=list(_BUILDERS))
    parser.add_argument("--target", required=True, help="Target word/idiom for this game")
    parser.add_argument(
        "--max-attempts", type=int, default=None,
        help=f"Override max attempts. Defaults: {DEFAULT_MAX_ATTEMPTS}",
    )
    parser.add_argument("--no-banner", action="store_true")
    args = parser.parse_args()

    mcp = build_server(args.mode, args.target, args.max_attempts)
    mcp.run(show_banner=not args.no_banner)


if __name__ == "__main__":
    main()
