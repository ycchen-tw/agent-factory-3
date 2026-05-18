# Wordle Multilingual MCP

A single MCP server that serves **four Wordle variants** under a unified `guess` tool. Spawn one process per rollout with `--mode` and `--target`; the LLM only ever sees one tool name.

| Mode | Length | Attempts | Symbol set | Scoring axes |
|---|---|---|---|---|
| `english`  | 5 | 6  | a–z latin letters             | letter        |
| `chewing`  | 5 | 6  | Bopomofo / 注音 (no tone marks) | symbol        |
| `japanese` | 5 | 6  | Katakana                      | kana          |
| `handle`   | 4 | 10 | Hanzi (4-character idiom)     | **char + pinyin initial + final + tone** (4 independent axes) |

The first three use the classic Wordle two-pass `correct / present / absent` rule per symbol. `handle` runs the same two-pass rule **independently on each of four axes**, mirroring [antfu/handle](https://github.com/antfu/handle).

> *Created 2026-05-19.*

---

## Reference repos

The game logic, word lists, and (where applicable) display annotations are reconstructed from these upstream projects. Original repos retain their own licenses; please consult them before redistributing.

| Mode | Source repo | What we took |
|---|---|---|
| `english`  | [MarkZither/react-wordle](https://github.com/MarkZither/react-wordle) (fork of the now-deleted [cwackerfuss/react-wordle](https://github.com/cwackerfuss/react-wordle)) | 5-letter slice of `src/constants/wordlist.ts` (answers) and `src/constants/validGuesses.ts` (valid guesses) |
| `chewing`  | [wordshk/react-wordle](https://github.com/wordshk/react-wordle) | `wordlist.ts` (注音 answers) and `validGuesses.ts` (注音 → 漢字 display map) |
| `japanese` | [rikito-ohnishi/tango](https://github.com/rikito-ohnishi/tango) (ことのはたんご) | `Q_fil_ippan.csv` (kanji ↔ kana puzzle pool) and `A_data_new.csv` (kana validity list) |
| `handle`   | [antfu/handle](https://github.com/antfu/handle) (汉兜) | `src/answers/list.ts` (curated daily idiom pool), `src/data/idioms.txt` (full idiom dictionary), `src/data/polyphones.json` (special-pronunciation overrides) |

Pinyin decomposition for `handle` uses [`pypinyin`](https://github.com/mozillazg/python-pinyin) (TONE3 style), with the upstream `polyphones.json` taking precedence for known multi-syllable idioms.

The 4-axis scoring algorithm follows handle's reference implementation at [`src/logic/utils.ts`](https://github.com/antfu/handle/blob/main/src/logic/utils.ts): per-axis two-pass scoring with separate "unmatched pools" for char / tone / pinyin parts, plus the polyphone override that re-uses the answer's pinyin when the same hanzi appears in the guess.

---

## Quick start

```bash
# Launch one game (stdio MCP server).
.venv/bin/python examples/wordle_multilingual/mcp_tool/server.py \
    --mode english --target apple --no-banner

# Other modes — target must come from the corresponding answers.txt:
#   --mode chewing  --target ㄏㄨㄢㄍㄨ
#   --mode japanese --target ジュウジロ
#   --mode handle   --target 路不拾遗
```

The server exposes one tool, `guess`, with mode-specific instructions baked into the `instructions` field. Connect with any MCP client — `fastmcp.Client` is the canonical option:

```python
import asyncio, sys
from fastmcp import Client

config = {"mcpServers": {"wordle": {
    "command": sys.executable,
    "args": ["examples/wordle_multilingual/mcp_tool/server.py",
             "--mode", "english", "--target", "apple", "--no-banner"],
}}}

async def play():
    async with Client(config) as c:
        r = await c.call_tool("guess", {"word": "table"})
        print(r.data)

asyncio.run(play())
```

---

## Response shape

**english / chewing / japanese** — one status per symbol:

```jsonc
{
  "mode": "english",
  "attempts": 1, "attempts_remaining": 5,
  "won": false, "lost": false, "game_over": false,
  "guess": "table",
  "results": [
    {"char": "t", "status": "absent"},
    {"char": "a", "status": "correct"},
    {"char": "b", "status": "absent"},
    {"char": "l", "status": "present"},
    {"char": "e", "status": "absent"}
  ]
}
```

For `chewing` / `japanese`, the `guess` and `answer` strings additionally carry the hanzi/kanji annotation in parentheses, e.g. `"ジュウジロ (十字路)"`.

**handle** — four statuses per character position:

```jsonc
{
  "mode": "handle",
  "guess": "恬不知耻",
  "char_results": [
    {"char": "恬", "char_status": "absent",
     "initial": "t",  "initial_status": "absent",
     "final":   "ian","final_status":   "absent",
     "tone":    2,    "tone_status":    "present"},
    ...
  ]
}
```

Invalid input (wrong length / not in valid list) still consumes one attempt and returns an `error` field, matching the contract of [`examples/wordle/mcp_tool/server.py`](../wordle/mcp_tool/server.py).

When the game ends, the response carries `won` or `lost`, `game_over: true`, `early_exit: true`, and `answer` (with display annotation where applicable).

---

## File layout

```
wordle_multilingual/
├── README.md
├── train.py                      # RL training entry — Japanese mode (mirrors examples/wordle/train.py)
├── serve.sh                      # sglang TP=2 launcher
├── run_train.sh                  # accelerate launch FSDP2 training
├── accelerate_config.yaml        # FSDP2 settings
├── mcp_tool/
│   ├── server.py                 # unified entry point (--mode --target)
│   ├── games/
│   │   ├── base.py               # two-pass scoring base class
│   │   ├── english.py
│   │   ├── chewing.py            # + hanzi display
│   │   ├── japanese.py           # + kanji display
│   │   └── handle.py             # 4-axis scoring, pypinyin + polyphones
│   └── data/
│       ├── english/    answers.txt, valid.txt
│       ├── chewing/    answers.txt, valid.txt, display.json
│       ├── japanese/   answers.txt, valid.txt, display.json
│       └── handle/     answers.txt, valid.txt, polyphones.json
└── tools/
    ├── build_data.py             # regenerate data/ from ../../dev/wordles/
    └── playtest.py               # fastmcp.Client smoke test — 4 modes × 3 paths
```

---

## Data sizes (after extraction)

| Mode | answers | valid (incl. answers) | display / polyphones |
|---|---:|---:|---:|
| english  |  5,757 | 16,094 | — |
| chewing  |  1,439 | 21,721 | 21,721 |
| japanese |  7,910 | 37,736 |  7,910 |
| handle   |    424 | 26,410 |  3,401 polyphones |

`handle`'s answer pool is small by design — it's antfu's hand-curated daily-puzzle list from 2022-01-01 through the 2023-02-28 freeze (~14 months × 1 puzzle/day ≈ 424 items).

---

## Regenerating data

The data files under `mcp_tool/data/` are derived artifacts from the four reference repos cloned into `../../dev/wordles/`. To rebuild:

```bash
# Make sure ../../dev/wordles/ contains the four upstream repos.
.venv/bin/python examples/wordle_multilingual/tools/build_data.py
```

Build output:

```
english:    5757 answers /  16094 valid
chewing:    1439 answers /  21721 valid /  21721 display entries
japanese:   7910 answers /  37736 valid /   7910 display entries
handle:      424 answers /  26410 valid /   3401 polyphones
```

---

## Verifying

```bash
.venv/bin/python examples/wordle_multilingual/tools/playtest.py
```

Runs each mode through three scenarios (win, exhaust-attempts, invalid-input) via `fastmcp.Client` over stdio and asserts response shape + state transitions. Exits non-zero on any failure.

---

## Dependencies

Listed in the repo-root `pyproject.toml`:

- `fastmcp` — MCP server runtime
- `pypinyin` — handle's pinyin decomposition (added for this example)

Both are installed in `.venv` by the repo's standard `uv sync` flow.

---

## Training (Japanese mode)

A full RL training pipeline is provided for the **japanese** mode, mirroring
[`examples/wordle/`](../wordle/) (gpt-oss-120b LoRA on 4× H100). Only the
dataset wiring, MCP `--mode japanese` argument, prompt, and wandb run name
differ from the English demo — all hyperparameters (DDIS, LoRA r=8,
routing replay, merged weight sync, etc.) are identical.

### Hardware

- 4× H100 80GB

```
GPU 0,1   sglang TP=2 inference (optional DFlash)
GPU 2,3   FSDP2 training (2 ranks, LoRA on attention)
```

### Run it

```bash
# Terminal 1: launch sglang server
export MODEL_PATH=/path/to/gpt-oss-120b
export DRAFT_MODEL_PATH=/path/to/dflash-draft   # optional; unset to disable DFlash
bash examples/wordle_multilingual/serve.sh

# Wait until curl localhost:30100/health returns 200 (3–5 min cold start).

# Terminal 2: launch training
export MODEL_PATH=/path/to/gpt-oss-120b
bash examples/wordle_multilingual/run_train.sh
```

Logs land in `runs/wordle_ja_demo/logs/`.

### Dataset

`make_dataset()` in `train.py` reads `mcp_tool/data/japanese/answers.txt`
(7,910 katakana words) and produces one item per target. Each item carries
its own `mcp_config` that spawns the unified server with
`--mode japanese --target <word>`.

### Reward

Binary, same as the English demo: 1.0 if any tool step returned
`won=True` in its output, otherwise 0.0. See `wordle_reward` in `train.py`.

### Adapting to other modes

The training pipeline can be retargeted by changing two constants near the
top of `train.py`:

```python
WORDLE_MODE    = "japanese"   # → "english" | "chewing" | "japanese" | "handle"
WORDLE_ANSWERS = EXAMPLE_DIR / "mcp_tool" / "data" / WORDLE_MODE / "answers.txt"
```

You'll also want to update `prompt`, `dev_instructions`, and
`wandb_run_name` so the LLM is told what game it's playing. `handle` mode
emits richer per-character feedback (`char_results` with 4 axes) and the
default `max_attempts` becomes 10; the rest of the pipeline is unchanged.

---

## Relationship to other code in this repo

- [`examples/wordle/`](../wordle/) — the single-mode English Wordle training demo (gpt-oss-120b LoRA on 4× H100). This example shares the same per-rollout MCP-subprocess pattern, expands to four modes, and ships a parallel Japanese training pipeline.
- [`mcp_tools/wordle_multilingual/`](../../mcp_tools/wordle_multilingual/) — an earlier, separate-server-per-language implementation covering english / chewing / japanese. Kept for compatibility; this `examples/` version is the unified successor and additionally implements the `handle` mode.
