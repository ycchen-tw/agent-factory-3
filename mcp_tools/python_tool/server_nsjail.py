"""MCP HTTP Server with nsjail-sandboxed Python execution.

Drop-in replacement for server_v4p1.py using nsjail instead of bwrap.
nsjail provides namespace isolation, seccomp-bpf, cgroup resource limits,
and a protobuf config format.

Usage:
    SANDBOX_DIR=/tmp/sandbox_python uv run python mcp_tools/python_tool/server_nsjail.py --port 8811

Environment variables:
    SANDBOX_DIR: Path to sandbox dir on local disk (default: /tmp/sandbox_python)
                 Contains: python/ (runtime copy), venv/ (packages), bwrap_executor.py
    PYTHON_TOOL_TIMEOUT: Execution timeout in seconds (default: 10.0)
    PYTHON_TOOL_MAX_OUTPUT_CHARS: Max output characters (default: 0 = unlimited)
    PYTHON_TOOL_PRERUN: Python code to execute at session startup
    PYTHON_TOOL_TRACEBACK_MODE: "user_frames" | "last_frame" | "full"
    PYTHON_TOOL_MAX_HEAP_MB: Max heap memory in MB (default: 16384)
    BWRAP_IDLE_TIMEOUT: Idle session timeout in seconds (default: 900)
    MAX_SESSIONS: Max concurrent sandboxed sessions (default: 600)
    NSJAIL_PATH: Path to nsjail binary (default: auto-detect)
    NSJAIL_CGROUP_MOUNT: cgroup2 mount point (default: /sys/fs/cgroup)
    NSJAIL_USE_CGROUPV2: Use cgroup v2 (default: true)
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import signal
import sys
import tempfile
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
SANDBOX_DIR = Path(os.environ.get("SANDBOX_DIR", "/tmp/sandbox_python"))
SANDBOX_VENV = SANDBOX_DIR / "venv"
SANDBOX_PYTHON = SANDBOX_VENV / "bin" / "python3"
# Reuse the same executor — it's sandbox-agnostic
SANDBOX_EXECUTOR = SANDBOX_DIR / "bwrap_executor.py"

TIMEOUT = float(os.environ.get("PYTHON_TOOL_TIMEOUT", "10.0"))
OUTER_TIMEOUT = TIMEOUT + 10.0  # safety margin over inner SIGALRM
IDLE_TIMEOUT = float(os.environ.get("BWRAP_IDLE_TIMEOUT", "900"))
MAX_OUTPUT_CHARS = os.environ.get("PYTHON_TOOL_MAX_OUTPUT_CHARS", "0")
PRERUN = os.environ.get("PYTHON_TOOL_PRERUN", "")
TRACEBACK_MODE = os.environ.get("PYTHON_TOOL_TRACEBACK_MODE", "user_frames")
MAX_HEAP_MB = os.environ.get("PYTHON_TOOL_MAX_HEAP_MB", "16384")
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "600"))

NSJAIL_USE_CGROUPV2 = os.environ.get("NSJAIL_USE_CGROUPV2", "true").lower() in ("1", "true", "yes")
NSJAIL_CGROUP_MOUNT = os.environ.get("NSJAIL_CGROUP_MOUNT", "/sys/fs/cgroup")

TOOL_DESCRIPTION = (
    f"Execute Python in persistent session (timeout: {TIMEOUT}s). "
    "Variables preserved across calls. Timeout triggers KeyboardInterrupt (state preserved). "
    "Sandboxed: no network, read-only filesystem, no subprocess creation."
)

# ---------------------------------------------------------------------------
# nsjail command construction
# ---------------------------------------------------------------------------


def _find_nsjail() -> str:
    """Find nsjail binary."""
    explicit = os.environ.get("NSJAIL_PATH")
    if explicit:
        if os.path.isfile(explicit) and os.access(explicit, os.X_OK):
            return explicit
        raise RuntimeError(f"NSJAIL_PATH={explicit} is not executable")

    nsjail = shutil.which("nsjail")
    if nsjail:
        return nsjail

    # Common install locations
    for candidate in ["/usr/local/bin/nsjail", "/usr/bin/nsjail"]:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    raise RuntimeError(
        "nsjail not found. Install it:\n"
        "  apt install nsjail  (Debian/Ubuntu)\n"
        "  or build from https://github.com/google/nsjail"
    )


def _build_nsjail_cfg() -> str:
    """Generate nsjail protobuf config content."""
    mounts = []

    # System libraries (read-only)
    for path in ["/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc"]:
        if os.path.isdir(path):
            mounts.append(f"""
mount {{
    src: "{path}"
    dst: "{path}"
    is_bind: true
    rw: false
}}""")

    # Device and proc filesystems
    mounts.append("""
mount {
    dst: "/dev"
    fstype: "tmpfs"
    rw: true
}
mount {
    src: "/dev/null"
    dst: "/dev/null"
    is_bind: true
    rw: true
}
mount {
    src: "/dev/zero"
    dst: "/dev/zero"
    is_bind: true
    rw: false
}
mount {
    src: "/dev/urandom"
    dst: "/dev/urandom"
    is_bind: true
    rw: false
}
mount {
    dst: "/proc"
    fstype: "proc"
    rw: false
}""")

    # Writable /tmp (tmpfs)
    mounts.append("""
mount {
    dst: "/tmp"
    fstype: "tmpfs"
    rw: true
    options: "size=1073741824"
}""")

    # Sandbox dir (read-only bind, AFTER /tmp tmpfs so it's visible)
    mounts.append(f"""
mount {{
    src: "{SANDBOX_DIR}"
    dst: "{SANDBOX_DIR}"
    is_bind: true
    rw: false
}}""")

    max_heap_bytes = int(MAX_HEAP_MB) * 1024 * 1024

    # cgroup config
    cgroup_block = ""
    if NSJAIL_USE_CGROUPV2:
        cgroup_block = f"""
use_cgroupv2: true
cgroupv2_mount: "{NSJAIL_CGROUP_MOUNT}"
"""

    cfg = f"""
name: "python_sandbox"
description: "Python tool sandbox"

mode: ONCE

# Namespace isolation
clone_newnet: true
clone_newpid: true
clone_newns: true
clone_newuts: true
clone_newipc: true

# No time limit at nsjail level — handled by SIGALRM in executor
# (nsjail time_limit kills the process; we want KeyboardInterrupt for state preservation)
time_limit: 0

# Keep stdin open for JSON protocol
keep_env: false
silent: true
skip_setsid: true
disable_rl: true

# Don't log to stderr (would corrupt our pipe)
log_level: 3

# Resource limits
rlimit_as_type: SOFT
rlimit_cpu_type: INF
rlimit_fsize_type: SOFT
rlimit_nofile_type: SOFT
rlimit_data: {max_heap_bytes // 1024 // 1024}

# Hostname inside jail
hostname: "sandbox"
cwd: "/tmp"

{cgroup_block}

# Environment
envar: "HOME=/tmp"
envar: "PATH=/usr/bin:/usr/sbin:/bin:/sbin"
envar: "PYTHON_TOOL_TIMEOUT={TIMEOUT}"
envar: "PYTHON_TOOL_MAX_OUTPUT_CHARS={MAX_OUTPUT_CHARS}"
envar: "PYTHON_TOOL_PRERUN={PRERUN}"
envar: "PYTHON_TOOL_TRACEBACK_MODE={TRACEBACK_MODE}"
envar: "PYTHON_TOOL_MAX_HEAP_MB={MAX_HEAP_MB}"
envar: "NUMBA_CACHE_DIR=/tmp/numba_cache"
envar: "OPENBLAS_NUM_THREADS=4"
envar: "OMP_NUM_THREADS=4"
envar: "TQDM_DISABLE=1"
envar: "LD_LIBRARY_PATH=/usr/lib64:/lib64"

# Mounts
{"".join(mounts)}

# Command to execute
exec_bin {{
    path: "{SANDBOX_PYTHON}"
    arg: "{SANDBOX_PYTHON}"
    arg: "-u"
    arg: "{SANDBOX_EXECUTOR}"
}}
"""
    return cfg


def _build_nsjail_cmd() -> tuple[list[str], Path]:
    """Build nsjail command and write config file.

    Returns (command_list, config_file_path).
    """
    nsjail = _find_nsjail()

    if not SANDBOX_PYTHON.exists():
        raise RuntimeError(
            f"Sandbox Python not found at {SANDBOX_PYTHON}. "
            f"Run serve_python_tool.sh to build the sandbox env first."
        )
    if not SANDBOX_EXECUTOR.exists():
        raise RuntimeError(
            f"Sandbox executor not found at {SANDBOX_EXECUTOR}. "
            f"Run serve_python_tool.sh to build the sandbox env first."
        )

    cfg_content = _build_nsjail_cfg()

    # Write config to a persistent file (not temp — survives process lifetime)
    cfg_dir = SANDBOX_DIR / "nsjail"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "python_sandbox.cfg"
    cfg_path.write_text(cfg_content)

    cmd = [nsjail, "--config", str(cfg_path)]
    return cmd, cfg_path


NSJAIL_CMD, NSJAIL_CFG_PATH = _build_nsjail_cmd()


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


@dataclass
class NsjailSession:
    process: asyncio.subprocess.Process
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_used: float = field(default_factory=time.monotonic)
    session_id: str = ""


_sessions: dict[str, NsjailSession] = {}
_sessions_lock = asyncio.Lock()

_session_manager = None

_stats_spawned = 0
_stats_killed = 0
_stats_reaped_idle = 0
_stats_reaped_disconnect = 0
_stats_reaped_dead = 0


async def _spawn_nsjail(session_id: str) -> NsjailSession:
    """Spawn a new nsjail-sandboxed Python process."""
    global _stats_spawned
    proc = await asyncio.create_subprocess_exec(
        *NSJAIL_CMD,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        )
    _stats_spawned += 1
    logger.info(f"Spawned nsjail session={session_id[:12]}... pid={proc.pid} "
                f"(total={len(_sessions) + 1}, spawned={_stats_spawned})")
    return NsjailSession(process=proc, session_id=session_id)


async def get_or_create_session(session_id: str) -> NsjailSession | str:
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

        s = await _spawn_nsjail(session_id)
        _sessions[session_id] = s
        return s


async def kill_session(session_id: str) -> None:
    """Kill a session's nsjail process and remove from registry."""
    global _stats_killed
    async with _sessions_lock:
        s = _sessions.pop(session_id, None)
    if s is None:
        return
    try:
        s.process.kill()
        await asyncio.wait_for(s.process.wait(), timeout=5.0)
    except Exception:
        logger.debug(f"kill_session {session_id[:12]}... wait timeout (nsjail may be zombie)")
    _stats_killed += 1
    logger.debug(f"Killed session {session_id[:12]}...")


async def execute_in_nsjail(session: NsjailSession, code: str) -> str:
    """Send code to nsjail executor and return output."""
    async with session.lock:
        session.last_used = time.monotonic()
        proc = session.process

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

_STATS_LOG_INTERVAL = 300

_last_stats_log = 0.0


def _get_active_transport_ids() -> set[str] | None:
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
                    if transport_ids is not None and sid not in transport_ids:
                        to_kill.append((sid, "disconnect"))
                    elif now - s.last_used > IDLE_TIMEOUT:
                        to_kill.append((sid, "idle"))
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
    global _last_stats_log
    _last_stats_log = time.monotonic()
    reaper = asyncio.create_task(_idle_reaper())
    logger.info(
        f"Server started (timeout={TIMEOUT}s, idle_timeout={IDLE_TIMEOUT}s, "
        f"max_heap={MAX_HEAP_MB}MB, max_sessions={MAX_SESSIONS}, "
        f"sandbox_dir={SANDBOX_DIR}, jail=nsjail)"
    )
    try:
        yield {}
    finally:
        reaper.cancel()
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
        return result
    return await execute_in_nsjail(result, code)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _extract_session_manager(app):
    for route in app.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is not None and hasattr(endpoint, "session_manager"):
            return endpoint.session_manager
    for route in app.routes:
        sub_routes = getattr(route, "routes", None)
        if sub_routes:
            for sub in sub_routes:
                endpoint = getattr(sub, "endpoint", None)
                if endpoint is not None and hasattr(endpoint, "session_manager"):
                    return endpoint.session_manager
    return None


def _kill_orphan_executor_processes():
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
            logger.info(f"Killed {killed} orphan executor processes from previous runs")
    except Exception as e:
        logger.warning(f"Orphan cleanup failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP Python Tool Server (nsjail sandbox)")
    parser.add_argument("--port", type=int, default=8811)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    # Validate sandbox venv
    if not SANDBOX_PYTHON.exists():
        logger.error(f"Sandbox Python not found at {SANDBOX_PYTHON}. "
                     f"Run serve_python_tool.sh first.")
        sys.exit(1)

    logger.info(f"Using sandbox dir: {SANDBOX_DIR}")
    logger.info(f"nsjail config: {NSJAIL_CFG_PATH}")
    logger.info(f"nsjail command: {' '.join(NSJAIL_CMD)}")

    _kill_orphan_executor_processes()

    starlette_app = mcp.http_app(transport="streamable-http")
    _session_manager = _extract_session_manager(starlette_app)
    if _session_manager is not None:
        logger.info("Transport session tracking enabled (disconnect detection active)")
    else:
        logger.warning("Could not find session manager — disconnect detection disabled")

    import uvicorn
    uvicorn.run(
        starlette_app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
    )
