"""MCP HTTP Server with bwrap-sandboxed Python execution (v4).

Single-process HTTP server that manages per-session bwrap Python subprocesses.
Each MCP client session gets its own isolated bwrap sandbox for code execution.

Architecture:
    FastMCP HTTP Server (1 process, handles MCP protocol)
      └── per session: bwrap Python subprocess (lightweight, OS-level sandbox)

Usage:
    uv run python mcp_tools/python_tool/server_v4.py --port 8811

Environment variables:
    PYTHON_TOOL_TIMEOUT: Execution timeout in seconds (default: 10.0)
    PYTHON_TOOL_PRERUN: Python code to execute at startup in each sandbox
    PYTHON_TOOL_MAX_OUTPUT_CHARS: Max output characters (default: 0 = unlimited)
    PYTHON_TOOL_TRACEBACK_MODE: Traceback mode (default: "user_frames")
    PYTHON_TOOL_MAX_HEAP_MB: Max heap size per sandbox in MB (default: 16384)
    BWRAP_IDLE_TIMEOUT: Seconds before idle sessions are reaped (default: 900)
    MAX_SESSIONS: Maximum concurrent bwrap sessions (default: 600)
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import signal
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastmcp import FastMCP, Context

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_DIR = Path(__file__).resolve().parent.parent.parent
EXECUTOR_PATH = Path(__file__).resolve().parent / "bwrap_executor.py"
PYTHON_EXE = sys.executable

TIMEOUT = float(os.environ.get("PYTHON_TOOL_TIMEOUT", "10.0"))
OUTER_TIMEOUT = TIMEOUT + 10.0  # safety margin over inner SIGALRM
IDLE_TIMEOUT = float(os.environ.get("BWRAP_IDLE_TIMEOUT", "900"))
MAX_OUTPUT_CHARS = os.environ.get("PYTHON_TOOL_MAX_OUTPUT_CHARS", "0")
PRERUN = os.environ.get("PYTHON_TOOL_PRERUN", "")
TRACEBACK_MODE = os.environ.get("PYTHON_TOOL_TRACEBACK_MODE", "user_frames")
MAX_HEAP_MB = os.environ.get("PYTHON_TOOL_MAX_HEAP_MB", "16384")
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "600"))

TOOL_DESCRIPTION = (
    f"Execute Python in persistent session (timeout: {TIMEOUT}s). "
    "Variables preserved across calls. Timeout triggers KeyboardInterrupt (state preserved). "
    "Sandboxed: no network, read-only filesystem, no subprocess creation."
)

# ---------------------------------------------------------------------------
# bwrap command
# ---------------------------------------------------------------------------


def _build_bwrap_cmd() -> list[str]:
    """Build the bwrap command for spawning sandboxed Python."""
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError("bwrap (bubblewrap) not found in PATH")

    return [
        bwrap,
        "--ro-bind", "/", "/",       # read-only root
        "--dev", "/dev",              # minimal /dev
        "--proc", "/proc",            # /proc for resource checks
        "--tmpfs", "/tmp",            # writable /tmp
        "--unshare-net",              # no network
        "--unshare-pid",              # PID namespace isolation
        "--die-with-parent",          # kill if parent dies
        "--",
        PYTHON_EXE, "-u",            # unbuffered
        str(EXECUTOR_PATH),
    ]


BWRAP_CMD = _build_bwrap_cmd()

BWRAP_ENV = {
    "HOME": os.environ.get("HOME", "/tmp"),
    "PATH": os.environ.get("PATH", "/usr/bin"),
    "PYTHON_TOOL_TIMEOUT": str(TIMEOUT),
    "PYTHON_TOOL_MAX_OUTPUT_CHARS": MAX_OUTPUT_CHARS,
    "PYTHON_TOOL_PRERUN": PRERUN,
    "PYTHON_TOOL_TRACEBACK_MODE": TRACEBACK_MODE,
    "PYTHON_TOOL_MAX_HEAP_MB": MAX_HEAP_MB,
    # numba/galois need writable cache dir (bwrap root is read-only)
    "NUMBA_CACHE_DIR": "/tmp/numba_cache",
}


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


@dataclass
class BwrapSession:
    process: asyncio.subprocess.Process
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_used: float = field(default_factory=time.monotonic)
    session_id: str = ""


_sessions: dict[str, BwrapSession] = {}
_sessions_lock = asyncio.Lock()

# Will be set after FastMCP app is built, before server starts
_session_manager = None

# Stats
_stats_spawned = 0
_stats_killed = 0
_stats_reaped_idle = 0
_stats_reaped_disconnect = 0
_stats_reaped_dead = 0


async def _spawn_bwrap(session_id: str) -> BwrapSession:
    """Spawn a new bwrap-sandboxed Python process."""
    global _stats_spawned
    proc = await asyncio.create_subprocess_exec(
        *BWRAP_CMD,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env=BWRAP_ENV,
    )
    _stats_spawned += 1
    logger.info(f"Spawned bwrap session={session_id[:12]}... pid={proc.pid} "
                f"(total={len(_sessions) + 1}, spawned={_stats_spawned})")
    return BwrapSession(process=proc, session_id=session_id)


async def get_or_create_session(session_id: str) -> BwrapSession | str:
    """Get existing session or create a new one. Returns error string if over limit."""
    async with _sessions_lock:
        if session_id in _sessions:
            s = _sessions[session_id]
            if s.process.returncode is None:
                s.last_used = time.monotonic()
                return s
            logger.warning(f"Session {session_id[:12]}... process died, recreating")
            del _sessions[session_id]

        if len(_sessions) >= MAX_SESSIONS:
            return f"[ERROR] Server at capacity ({MAX_SESSIONS} sessions). Try again later."

        s = await _spawn_bwrap(session_id)
        _sessions[session_id] = s
        return s


async def kill_session(session_id: str) -> None:
    """Kill a session's bwrap process and remove from registry."""
    global _stats_killed
    async with _sessions_lock:
        s = _sessions.pop(session_id, None)
    if s is None:
        return
    try:
        s.process.kill()
        await asyncio.wait_for(s.process.wait(), timeout=5.0)
    except Exception:
        pass
    _stats_killed += 1
    logger.debug(f"Killed session {session_id[:12]}...")


async def execute_in_bwrap(session: BwrapSession, code: str) -> str:
    """Send code to bwrap executor and return output."""
    async with session.lock:
        session.last_used = time.monotonic()
        proc = session.process

        # Check process is alive
        if proc.returncode is not None:
            return "[ERROR] Executor process crashed. State lost. Try again."

        request = json.dumps({"code": code}) + "\n"
        proc.stdin.write(request.encode())
        await proc.stdin.drain()

        try:
            line = await asyncio.wait_for(
                proc.stdout.readline(),
                timeout=OUTER_TIMEOUT,
            )
        except asyncio.TimeoutError:
            await kill_session(session.session_id)
            return f"[ERROR] Executor timeout after {OUTER_TIMEOUT:.0f}s (process killed)"

        if not line:
            await kill_session(session.session_id)
            return "[ERROR] Executor process crashed. State lost. Try again."

        try:
            resp = json.loads(line)
            return resp.get("output", "[ERROR] No output in response")
        except json.JSONDecodeError:
            await kill_session(session.session_id)
            return "[ERROR] Executor returned invalid response (process killed)"


# ---------------------------------------------------------------------------
# Idle reaper + disconnect detection
# ---------------------------------------------------------------------------

_STATS_LOG_INTERVAL = 300  # log stats every 5 minutes
_last_stats_log = 0.0


def _get_active_transport_ids() -> set[str] | None:
    """Get active (non-terminated) MCP session IDs from the transport layer.

    Sessions where the client sent DELETE have is_terminated=True but remain
    in _server_instances. We filter those out so the reaper can detect them.
    Returns None if session_manager is unavailable (reaper falls back to idle-only).
    """
    if _session_manager is None:
        return None
    try:
        return {
            sid for sid, transport in _session_manager._server_instances.items()
            if not transport.is_terminated
        }
    except Exception:
        return None


async def _idle_reaper():
    """Periodically kill idle, dead, or disconnected sessions.

    Three cleanup triggers:
    1. idle: last_used > IDLE_TIMEOUT
    2. dead: bwrap process exited
    3. disconnect: MCP session no longer in transport layer (client disconnected)
    """
    global _stats_reaped_idle, _stats_reaped_disconnect, _stats_reaped_dead
    global _last_stats_log

    while True:
        await asyncio.sleep(30)
        try:
            now = time.monotonic()
            transport_ids = _get_active_transport_ids()
            to_kill: list[tuple[str, str]] = []

            async with _sessions_lock:
                for sid, s in _sessions.items():
                    # 1. Disconnected: not in transport layer anymore
                    if transport_ids is not None and sid not in transport_ids:
                        to_kill.append((sid, "disconnect"))
                    # 2. Idle timeout
                    elif now - s.last_used > IDLE_TIMEOUT:
                        to_kill.append((sid, "idle"))
                    # 3. Process died
                    elif s.process.returncode is not None:
                        to_kill.append((sid, "dead"))

            for sid, reason in to_kill:
                await kill_session(sid)
                if reason == "idle":
                    _stats_reaped_idle += 1
                elif reason == "disconnect":
                    _stats_reaped_disconnect += 1
                elif reason == "dead":
                    _stats_reaped_dead += 1

            if to_kill:
                logger.info(
                    f"Reaped {len(to_kill)} sessions "
                    f"({sum(1 for _, r in to_kill if r == 'disconnect')} disconnect, "
                    f"{sum(1 for _, r in to_kill if r == 'idle')} idle, "
                    f"{sum(1 for _, r in to_kill if r == 'dead')} dead), "
                    f"{len(_sessions)} remaining"
                )

            # Periodic stats log
            if now - _last_stats_log > _STATS_LOG_INTERVAL:
                _last_stats_log = now
                transport_count = len(transport_ids) if transport_ids is not None else "?"
                logger.info(
                    f"[stats] sessions={len(_sessions)} transport={transport_count} "
                    f"spawned={_stats_spawned} killed={_stats_killed} "
                    f"reaped(idle={_stats_reaped_idle} disconnect={_stats_reaped_disconnect} "
                    f"dead={_stats_reaped_dead})"
                )

        except Exception:
            logger.exception("Reaper error (will retry next cycle)")


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(server):
    """Server lifespan: start idle reaper, cleanup all sessions on shutdown."""
    global _last_stats_log
    _last_stats_log = time.monotonic()
    reaper = asyncio.create_task(_idle_reaper())
    logger.info(
        f"Server started (timeout={TIMEOUT}s, idle_timeout={IDLE_TIMEOUT}s, "
        f"max_heap={MAX_HEAP_MB}MB, max_sessions={MAX_SESSIONS})"
    )
    try:
        yield {}
    finally:
        reaper.cancel()
        # Kill all remaining sessions
        sids = list(_sessions.keys())
        for sid in sids:
            await kill_session(sid)
        logger.info(f"Server shutdown, cleaned up {len(sids)} sessions")


mcp = FastMCP(
    name="builtin_python",
    instructions=TOOL_DESCRIPTION,
    lifespan=lifespan,
)


@mcp.tool(
    name="python",
    description=TOOL_DESCRIPTION,
)
async def python_tool(code: str, ctx: Context) -> str:
    """Execute Python code in a sandboxed session."""
    session_id = ctx.session_id
    result = await get_or_create_session(session_id)
    if isinstance(result, str):
        return result  # error message
    return await execute_in_bwrap(result, code)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _extract_session_manager(app):
    """Extract StreamableHTTPSessionManager from the Starlette app routes."""
    for route in app.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is not None and hasattr(endpoint, "session_manager"):
            return endpoint.session_manager
    # Try nested routes (Mount)
    for route in app.routes:
        sub_routes = getattr(route, "routes", None)
        if sub_routes:
            for sub in sub_routes:
                endpoint = getattr(sub, "endpoint", None)
                if endpoint is not None and hasattr(endpoint, "session_manager"):
                    return endpoint.session_manager
    return None


def _kill_orphan_bwrap_processes():
    """Kill leftover bwrap_executor processes from previous server instances."""
    import subprocess
    try:
        result = subprocess.run(
            ["pgrep", "-f", "bwrap_executor.py"],
            capture_output=True, text=True,
        )
        pids = [int(p) for p in result.stdout.strip().split("\n") if p.strip()]
        if not pids:
            return
        my_pid = os.getpid()
        killed = 0
        for pid in pids:
            if pid == my_pid:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
                killed += 1
            except (ProcessLookupError, PermissionError):
                pass
        if killed:
            logger.info(f"Killed {killed} orphan bwrap_executor processes from previous runs")
    except Exception as e:
        logger.warning(f"Orphan cleanup failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP Python Tool Server (bwrap)")
    parser.add_argument("--port", type=int, default=8811)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    # Clean up orphan bwrap processes from previous runs
    _kill_orphan_bwrap_processes()

    # Build the ASGI app manually so we can extract the session manager
    starlette_app = mcp.http_app(transport="streamable-http")
    _session_manager = _extract_session_manager(starlette_app)
    if _session_manager is not None:
        logger.info("Transport session tracking enabled (disconnect detection active)")
    else:
        logger.warning("Could not find session manager — disconnect detection disabled, "
                       "falling back to idle timeout only")

    # Run with uvicorn
    import uvicorn
    uvicorn.run(
        starlette_app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
    )
