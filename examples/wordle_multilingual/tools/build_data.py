"""Build wordle_multilingual data files from dev/wordles/ source repos.

Re-extracts:
  english  ← react-wordle-classic (MarkZither/react-wordle, 5-letter slice)
  chewing  ← react-wordle-wordshk
  japanese ← tango (kotonoha-tango CSVs)
  handle   ← antfu/handle

Run once to regenerate examples/wordle_multilingual/mcp_tool/data/.
"""
from __future__ import annotations

import csv
import json
import re
from collections import OrderedDict
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = EXAMPLE_DIR.parents[1]
SRC = REPO_ROOT / "dev" / "wordles"
DST = EXAMPLE_DIR / "mcp_tool" / "data"

WORD_LENGTH = {"english": 5, "chewing": 5, "japanese": 5, "handle": 4}


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_english() -> tuple[int, int]:
    """5-letter slice from MarkZither's two-list fork."""
    base = SRC / "react-wordle-classic" / "src" / "constants"
    pattern = re.compile(r"'([a-z]{5})'")
    answers = sorted({m.lower() for m in pattern.findall((base / "wordlist.ts").read_text())})
    valid = sorted(
        {m.lower() for m in pattern.findall((base / "validGuesses.ts").read_text())}
        | set(answers)
    )
    _write_lines(DST / "english" / "answers.txt", answers)
    _write_lines(DST / "english" / "valid.txt", valid)
    return len(answers), len(valid)


def build_chewing() -> tuple[int, int, int]:
    """Bopomofo wordlist.ts: '"ㄏㄨㄢㄍㄨ", // 環顧'
    validGuesses.ts: '"ㄅㄚㄅㄟㄗ":"八輩子",'  (value may be 'a/b' multi-hanzi)
    """
    base = SRC / "react-wordle-wordshk" / "src" / "constants"
    display: "OrderedDict[str, list[str]]" = OrderedDict()
    answers: list[str] = []

    answer_line = re.compile(r'^\s*"([^"]+)"\s*,?\s*(?://\s*(.+?))?\s*$')
    for line in (base / "wordlist.ts").read_text().splitlines():
        m = answer_line.match(line)
        if not m:
            continue
        zhuyin = m.group(1)
        if not zhuyin or "ㄅ" > zhuyin[0]:  # quick reject non-bopomofo
            continue
        if len(zhuyin) != WORD_LENGTH["chewing"]:
            continue
        if zhuyin not in display:
            display[zhuyin] = []
            answers.append(zhuyin)
        hanzi = (m.group(2) or "").strip()
        if hanzi and hanzi not in display[zhuyin]:
            display[zhuyin].append(hanzi)

    valid_pair = re.compile(r'"([^"]+)"\s*:\s*"([^"]+)"')
    valid: list[str] = []
    seen_v: set[str] = set()
    for m in valid_pair.finditer((base / "validGuesses.ts").read_text()):
        zhuyin, hanzi_str = m.group(1), m.group(2)
        if len(zhuyin) != WORD_LENGTH["chewing"]:
            continue
        if zhuyin not in seen_v:
            valid.append(zhuyin)
            seen_v.add(zhuyin)
        if zhuyin not in display:
            display[zhuyin] = []
        for h in hanzi_str.split("/"):
            h = h.strip()
            if h and h not in display[zhuyin]:
                display[zhuyin].append(h)

    for a in answers:
        if a not in seen_v:
            valid.append(a)
            seen_v.add(a)

    _write_lines(DST / "chewing" / "answers.txt", answers)
    _write_lines(DST / "chewing" / "valid.txt", sorted(valid))
    _write_json(DST / "chewing" / "display.json", dict(display))
    return len(answers), len(valid), len(display)


def build_japanese() -> tuple[int, int, int]:
    """tango Q_fil_ippan.csv: kanji,kana   ;  A_data_new.csv: kana only."""
    base = SRC / "tango" / "kotonoha-tango" / "public" / "data"
    display: "OrderedDict[str, list[str]]" = OrderedDict()
    answers: list[str] = []
    L = WORD_LENGTH["japanese"]

    with (base / "Q_fil_ippan.csv").open(encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            kanji, kana = row[0].strip(), row[1].strip()
            if len(kana) != L:
                continue
            if kana not in display:
                display[kana] = []
                answers.append(kana)
            if kanji and kanji != kana and kanji not in display[kana]:
                display[kana].append(kanji)

    valid: list[str] = []
    seen_v: set[str] = set()
    with (base / "A_data_new.csv").open(encoding="utf-8") as f:
        for line in f:
            kana = line.strip()
            if len(kana) == L and kana not in seen_v:
                valid.append(kana)
                seen_v.add(kana)
    for a in answers:
        if a not in seen_v:
            valid.append(a)
            seen_v.add(a)

    _write_lines(DST / "japanese" / "answers.txt", answers)
    _write_lines(DST / "japanese" / "valid.txt", sorted(valid))
    _write_json(DST / "japanese" / "display.json", dict(display))
    return len(answers), len(valid), len(display)


def build_handle() -> tuple[int, int, int]:
    """handle answers/list.ts: tuples like ['idiom', 'hint'].
    data/idioms.txt: all valid 4-hanzi idioms.
    data/polyphones.json: pronunciation override map.
    """
    base = SRC / "handle" / "src"
    answer_text = (base / "answers" / "list.ts").read_text()
    answer_idioms: list[str] = []
    seen: set[str] = set()
    L = WORD_LENGTH["handle"]
    for m in re.finditer(r"\['([^']+)'", answer_text):
        idiom = m.group(1)
        if len(idiom) == L and idiom not in seen:
            answer_idioms.append(idiom)
            seen.add(idiom)

    valid: list[str] = []
    seen_v: set[str] = set()
    with (base / "data" / "idioms.txt").open(encoding="utf-8") as f:
        for line in f:
            idiom = line.strip()
            if len(idiom) == L and idiom not in seen_v:
                valid.append(idiom)
                seen_v.add(idiom)
    for a in answer_idioms:
        if a not in seen_v:
            valid.append(a)
            seen_v.add(a)

    polyphones = json.loads((base / "data" / "polyphones.json").read_text(encoding="utf-8"))

    _write_lines(DST / "handle" / "answers.txt", answer_idioms)
    _write_lines(DST / "handle" / "valid.txt", sorted(valid))
    _write_json(DST / "handle" / "polyphones.json", polyphones)
    return len(answer_idioms), len(valid), len(polyphones)


if __name__ == "__main__":
    en_a, en_v = build_english()
    print(f"english:  {en_a:>6d} answers / {en_v:>6d} valid")
    zh_a, zh_v, zh_d = build_chewing()
    print(f"chewing:  {zh_a:>6d} answers / {zh_v:>6d} valid / {zh_d:>6d} display entries")
    ja_a, ja_v, ja_d = build_japanese()
    print(f"japanese: {ja_a:>6d} answers / {ja_v:>6d} valid / {ja_d:>6d} display entries")
    he_a, he_v, he_p = build_handle()
    print(f"handle:   {he_a:>6d} answers / {he_v:>6d} valid / {he_p:>6d} polyphones")
