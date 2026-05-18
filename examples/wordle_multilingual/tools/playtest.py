"""End-to-end smoke test for the unified Wordle MCP server.

For each of the 4 modes:
  1. Spawn server.py with a known target.
  2. Use fastmcp.Client over stdio to call the `guess` tool.
  3. Exercise three paths: win, lose (exhaust attempts), invalid input.

Failures abort with a non-zero exit code so this is CI-friendly.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from fastmcp import Client

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
SERVER = EXAMPLE_DIR / "mcp_tool" / "server.py"
DATA = EXAMPLE_DIR / "mcp_tool" / "data"


def _first_answer(mode: str) -> str:
    return (DATA / mode / "answers.txt").read_text(encoding="utf-8").splitlines()[0].strip()


def _other_answer(mode: str, target: str) -> str:
    for line in (DATA / mode / "answers.txt").read_text(encoding="utf-8").splitlines():
        v = line.strip()
        if v and v != target:
            return v
    raise RuntimeError(f"no second answer available for mode {mode}")


def _client_for(mode: str, target: str, max_attempts: int | None = None) -> Client:
    args = [str(SERVER), "--mode", mode, "--target", target, "--no-banner"]
    if max_attempts is not None:
        args += ["--max-attempts", str(max_attempts)]
    config = {
        "mcpServers": {
            "wordle": {"command": sys.executable, "args": args}
        }
    }
    return Client(config)


def _extract_payload(result) -> dict:
    """fastmcp.Client.call_tool returns CallToolResult; pull our JSON-ish dict."""
    if hasattr(result, "data") and isinstance(result.data, dict):
        return result.data
    if hasattr(result, "structured_content") and isinstance(result.structured_content, dict):
        return result.structured_content
    if hasattr(result, "content") and result.content:
        first = result.content[0]
        if hasattr(first, "text"):
            try:
                return json.loads(first.text)
            except json.JSONDecodeError:
                pass
    raise RuntimeError(f"could not extract payload from {result!r}")


async def _run_scenario(mode: str, target: str, scenario: str, *, max_attempts: int | None = None):
    """scenario ∈ {'win','lose','invalid'}"""
    async with _client_for(mode, target, max_attempts) as client:
        tools = await client.list_tools()
        assert any(t.name == "guess" for t in tools), f"[{mode}] no `guess` tool"

        if scenario == "win":
            res = await client.call_tool("guess", {"word": target})
            payload = _extract_payload(res)
            assert payload["won"] is True, f"[{mode}/win] expected won=True, got {payload}"
            assert payload["game_over"] is True
            assert payload["attempts"] == 1
            return payload

        if scenario == "lose":
            # Use a wrong-but-valid guess up to max_attempts.
            wrong = _other_answer(mode, target)
            last = None
            limit = max_attempts or (10 if mode == "handle" else 6)
            for _ in range(limit):
                res = await client.call_tool("guess", {"word": wrong})
                last = _extract_payload(res)
            assert last is not None
            assert last["lost"] is True, f"[{mode}/lose] expected lost=True, got {last}"
            assert last["game_over"] is True
            assert last["won"] is False
            assert last["answer"], f"[{mode}/lose] expected answer field once game is over"
            return last

        if scenario == "invalid":
            # First call: wrong-length word triggers an error but still consumes a turn.
            res = await client.call_tool("guess", {"word": "x"})
            payload = _extract_payload(res)
            assert "error" in payload, f"[{mode}/invalid] expected error field, got {payload}"
            assert payload["attempts"] == 1
            return payload

        raise ValueError(f"unknown scenario {scenario}")


def _print_block(title: str, payload: dict):
    print(f"  {title}")
    if "results" in payload:
        glyph = {"correct": "🟩", "present": "🟨", "absent": "⬜"}
        chars = [f"{glyph[r['status']]}{r['char']}" for r in payload["results"]]
        print(f"    feedback: {' '.join(chars)}")
    if "char_results" in payload:
        glyph = {"correct": "🟩", "present": "🟨", "absent": "⬜"}
        for r in payload["char_results"]:
            print(
                f"    {r['char']}  char={glyph[r['char_status']]} "
                f"ini={glyph[r['initial_status']]}{r['initial'] or '·'} "
                f"fin={glyph[r['final_status']]}{r['final']} "
                f"tone={glyph[r['tone_status']]}{r['tone']}"
            )
    keys = ("attempts", "attempts_remaining", "won", "lost", "game_over", "error", "answer")
    state = {k: payload[k] for k in keys if k in payload}
    print(f"    state:    {state}")


async def main():
    modes = ["english", "chewing", "japanese", "handle"]
    summary: list[tuple[str, str, str]] = []

    for mode in modes:
        target = _first_answer(mode)
        print(f"\n=== {mode}  target = {target!r}  ===")
        for scenario in ("win", "lose", "invalid"):
            try:
                payload = await _run_scenario(mode, target, scenario)
                _print_block(f"[{scenario}]", payload)
                summary.append((mode, scenario, "PASS"))
            except AssertionError as e:
                print(f"  [{scenario}] FAIL: {e}")
                summary.append((mode, scenario, "FAIL"))
            except Exception as e:
                print(f"  [{scenario}] ERROR: {e!r}")
                summary.append((mode, scenario, "ERROR"))

    print("\n=== summary ===")
    for mode, scenario, status in summary:
        print(f"  {mode:>9s} / {scenario:<8s} {status}")
    failed = [s for s in summary if s[2] != "PASS"]
    if failed:
        print(f"\n{len(failed)} scenario(s) did not pass.")
        sys.exit(1)
    print("\nAll scenarios PASS.")


if __name__ == "__main__":
    asyncio.run(main())
