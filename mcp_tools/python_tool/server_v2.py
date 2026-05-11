"""Secure Python Execution MCP Server (v2 - Sync + KeyboardInterrupt)

Single-file MCP server with:
- Sync execution (signal works in main thread)
- KeyboardInterrupt on timeout (like Jupyter)
- State preserved after interrupt
- Float timeout via setitimer
- Resource limits (RAM, no file write, no fork)
- Prerun code for module preloading
- Configurable import restrictions and traceback formatting

Environment variables:
    PYTHON_TOOL_TIMEOUT: Execution timeout in seconds (default: 10.0)
    PYTHON_TOOL_MAX_HEAP_MB: Max heap size in MB (default: 4096)
    PYTHON_TOOL_PRERUN: Python code to execute at startup (e.g., imports)
    PYTHON_TOOL_RESTRICT_IMPORTS: "1" to restrict imports (default), "0" to allow all
    PYTHON_TOOL_MAX_OUTPUT_CHARS: Max output characters (default: 0 = unlimited)
        Truncates middle (keep head 1/3 + tail 2/3) when exceeded.
    PYTHON_TOOL_TRACEBACK_MODE: Traceback formatting mode (default: "user_frames")
        - "user_frames": Filter non-user frames, keep full user call chain
        - "last_frame": Keep only the last frame
        - "full": Show complete traceback

Example PYTHON_TOOL_PRERUN:
    import math
    import numpy
    import sympy
    import mpmath
    mpmath.mp.dps = 64
"""

import io
import os
import sys
import signal
import inspect
import builtins
import resource
import traceback


def apply_resource_limits():
    """Apply OS-level resource limits to prevent OOM.

    Environment variables:
        PYTHON_TOOL_MAX_HEAP_MB: Max heap size in MB (default: 4096)

    Note: Uses RLIMIT_DATA (heap) instead of RLIMIT_AS (virtual memory)
    because numpy/scipy need ~3GB virtual address space for mmap'd .so files.
    NPROC/FSIZE not set because numpy needs threads for initialization.
    """
    max_heap_mb = int(os.environ.get("PYTHON_TOOL_MAX_HEAP_MB", "4096"))
    max_heap_bytes = max_heap_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_DATA, (max_heap_bytes, max_heap_bytes))


class SecurePythonSession:
    """Secure Python execution with Jupyter-like interrupt.

    - Timeout triggers KeyboardInterrupt (not kill)
    - Namespace preserved after interrupt
    - Supports float timeout via setitimer
    """

    # Blocked modules for user code
    BLOCKED_MODULES = frozenset({
        'os', 'subprocess', 'multiprocessing', 'pty',
        'socket', 'urllib', 'http', 'ftplib', 'smtplib',
        'pathlib', 'shutil', 'tempfile', 'fileinput', 'glob',
        'pickle', 'marshal', 'shelve', 'dbm',
        'gc', 'ctypes', 'cffi',
        'pandas', 'matplotlib', 'seaborn', 'plotly',
        'importlib',
    })

    # Dangerous builtins to block
    BLOCKED_BUILTINS = frozenset({
        'open', 'eval', 'exec', 'compile',
        'getattr', 'setattr', 'delattr',
        'vars', 'globals', 'locals',
    })

    def __init__(
        self,
        timeout: float = 60.0,
        prerun: str = "",
        restrict_imports: bool = True,
        traceback_mode: str = "user_frames",
        max_output_chars: int = 0,
    ):
        self._timeout = timeout
        self._prerun = prerun
        self._restrict_imports = restrict_imports
        self._traceback_mode = traceback_mode
        self._max_output_chars = max_output_chars
        self._namespace = self._create_namespace()
        if self._prerun.strip():
            self._run_prerun(self._prerun)

    def _create_safe_import(self):
        """Import function that blocks dangerous modules for user code."""
        import importlib

        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            base_module = name.split('.')[0]

            if base_module in self.BLOCKED_MODULES:
                # Check if called from user code (filename is <string>)
                frame = inspect.currentframe()
                try:
                    for _ in range(10):
                        if frame is None:
                            break
                        if frame.f_code.co_filename in ('<stdin>', '<string>'):
                            raise ImportError(
                                f"Import '{name}' blocked. "
                                f"Allowed: numpy, scipy, math, itertools, collections, etc."
                            )
                        frame = frame.f_back
                finally:
                    del frame

            return importlib.__import__(name, globals, locals, fromlist, level)

        return safe_import

    def _create_namespace(self) -> dict:
        """Create isolated namespace with safe builtins."""
        safe_builtins = {}
        for name in dir(builtins):
            if not name.startswith('_') or name in ('__build_class__', '__import__'):
                safe_builtins[name] = getattr(builtins, name)

        if self._restrict_imports:
            safe_builtins['__import__'] = self._create_safe_import()
            for name in self.BLOCKED_BUILTINS:
                safe_builtins[name] = None

        return {
            "__builtins__": safe_builtins,
            "__name__": "__main__",
        }

    def _run_prerun(self, code: str):
        """Execute prerun code with full builtins (trusted startup code).

        Prerun code runs without security restrictions since it's provided
        by the server operator, not the user. Modules imported here become
        available in the namespace for user code.
        """
        # Temporarily use real builtins for prerun
        real_builtins = self._namespace["__builtins__"].copy()
        self._namespace["__builtins__"] = builtins.__dict__
        try:
            exec(code, self._namespace)
        finally:
            # Restore safe builtins, but keep imported modules
            self._namespace["__builtins__"] = real_builtins

    def _format_traceback(self, tb_string: str) -> str:
        """Format traceback based on configured mode.

        Modes:
            - "full": Return complete traceback
            - "user_frames": Filter non-user frames, keep full user call chain
            - "last_frame": Keep only the last frame
        """
        if self._traceback_mode == "full":
            return tb_string

        lines = tb_string.strip().split('\n')

        if self._traceback_mode == "user_frames":
            # Filter non-user frames (keep <string> source frames)
            # A frame consists of: File line + optional code lines (indented, not starting with File)
            result = []
            i = 0
            while i < len(lines):
                line = lines[i]
                stripped = line.strip()

                if stripped.startswith('File "'):
                    is_user_frame = '<string>' in line
                    if is_user_frame:
                        result.append(line)
                    # Skip subsequent code/caret lines for this frame
                    i += 1
                    while i < len(lines):
                        next_line = lines[i]
                        next_stripped = next_line.strip()
                        # Stop if we hit another File line or error message (not indented code)
                        if next_stripped.startswith('File "') or (next_stripped and not next_line.startswith(' ')):
                            break
                        # This is a code/caret line belonging to current frame
                        if is_user_frame:
                            result.append(next_line)
                        i += 1
                    continue
                else:
                    # Header (Traceback...) or error message
                    result.append(line)
                    i += 1

            return '\n'.join(result)

        # last_frame: Keep only the last frame
        frame_indices = [i for i, line in enumerate(lines) if line.strip().startswith('File "')]
        if len(frame_indices) > 2:
            return '\n'.join(lines[frame_indices[-1]:])
        return tb_string

    def execute(self, code: str, timeout: float | None = None) -> str:
        """Execute code with timeout. Raises KeyboardInterrupt on timeout."""
        if '__subclasses__' in code:
            return "[SECURITY] Access to __subclasses__ is blocked"

        effective_timeout = timeout or self._timeout
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr

        def timeout_handler(signum, frame):
            raise KeyboardInterrupt(f"Timeout after {effective_timeout:.1f}s")

        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, effective_timeout)

        # FD-level redirection: capture C library output (e.g., scipy solver)
        # Redirect fd 1,2 to /dev/null to prevent pollution of MCP JSONRPC stream
        stdout_fd = sys.__stdout__.fileno()
        stderr_fd = sys.__stderr__.fileno()
        saved_stdout_fd = os.dup(stdout_fd)
        saved_stderr_fd = os.dup(stderr_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stdout_fd)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)

        try:
            sys.stdout, sys.stderr = stdout_buf, stderr_buf
            self._exec_with_last_expr(code)

        except KeyboardInterrupt as e:
            stderr_buf.write(f"[Interrupted] {e}\n")

        except SystemExit as e:
            stderr_buf.write(f"SystemExit: {e.code}\n")

        except Exception:
            stderr_buf.write(self._format_traceback(traceback.format_exc()))

        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
            sys.stdout, sys.stderr = old_stdout, old_stderr
            # Restore original file descriptors
            os.dup2(saved_stdout_fd, stdout_fd)
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)

        stdout = stdout_buf.getvalue()
        stderr = stderr_buf.getvalue()

        if stderr:
            output = f"{stdout.rstrip()}\n{stderr}" if stdout.strip() else stderr
        elif not stdout.strip():
            output = "[No output. Use print() to see results.]"
        else:
            output = stdout
        return self._truncate_output(output)

    def _truncate_output(self, output: str) -> str:
        """Truncate middle of output if it exceeds max_output_chars.

        Keeps head (1/3) + tail (2/3) since tail usually contains
        errors and final results which are more important.
        """
        limit = self._max_output_chars
        if limit <= 0 or len(output) <= limit:
            return output
        head_size = limit // 3
        tail_size = limit - head_size
        marker = f"\n\n... [Truncated: {len(output)} chars exceeded limit of {limit}] ...\n\n"
        return output[:head_size] + marker + output[-tail_size:]

    def _exec_with_last_expr(self, code: str):
        """Execute code, print last expression result (like Jupyter)."""
        import ast
        try:
            tree = ast.parse(code)
            if not tree.body:
                return

            # If last statement is an expression, eval and print it
            if isinstance(tree.body[-1], ast.Expr):
                if len(tree.body) > 1:
                    stmts = ast.Module(body=tree.body[:-1], type_ignores=[])
                    exec(compile(stmts, '<string>', 'exec'), self._namespace)

                last_expr = ast.Expression(body=tree.body[-1].value)
                result = eval(compile(last_expr, '<string>', 'eval'), self._namespace)
                if result is not None:
                    print(repr(result))
            else:
                exec(code, self._namespace)

        except SyntaxError:
            exec(code, self._namespace)

    def reset(self):
        """Clear namespace (like Jupyter restart kernel)."""
        self._namespace = self._create_namespace()
        if self._prerun.strip():
            self._run_prerun(self._prerun)


# === MCP Server ===

from typing import Annotated
from contextlib import asynccontextmanager
from pydantic import Field
from fastmcp import FastMCP
import fastmcp
import psutil

fastmcp.settings.log_level = 'CRITICAL'

TIMEOUT = float(os.environ.get("PYTHON_TOOL_TIMEOUT", "10.0"))
PRERUN = os.environ.get("PYTHON_TOOL_PRERUN", "")
RESTRICT_IMPORTS = os.environ.get("PYTHON_TOOL_RESTRICT_IMPORTS", "1") == "1"
TRACEBACK_MODE = os.environ.get("PYTHON_TOOL_TRACEBACK_MODE", "user_frames")
MAX_OUTPUT_CHARS = int(os.environ.get("PYTHON_TOOL_MAX_OUTPUT_CHARS", "0"))
SESSION = SecurePythonSession(
    timeout=TIMEOUT,
    prerun=PRERUN,
    restrict_imports=RESTRICT_IMPORTS,
    traceback_mode=TRACEBACK_MODE,
    max_output_chars=MAX_OUTPUT_CHARS,
)


@asynccontextmanager
async def cleanup_children_lifespan(server):
    """Cleanup child processes (e.g., PuLP solvers) on server shutdown."""
    try:
        yield {}
    finally:
        try:
            for child in psutil.Process().children(recursive=True):
                child.kill()
        except Exception:
            pass


mcp = FastMCP(
    name="builtin_python",
    instructions=f"Execute Python in persistent session (timeout: {TIMEOUT}s). "
                 "Variables preserved across calls. Timeout triggers KeyboardInterrupt (state preserved). "
                 "Blocked: file I/O, network, subprocess. "
                 "Allowed: numpy, scipy, math, itertools, collections, etc.",
    lifespan=cleanup_children_lifespan,
)


@mcp.tool(
    name="python",
    description=f"Execute Python in persistent session (timeout: {TIMEOUT}s). "
                "Variables preserved across calls. Timeout triggers KeyboardInterrupt (state preserved). "
                "Blocked: file I/O, network, subprocess. "
                "Allowed: numpy, scipy, math, itertools, collections, etc.",
)
def python(
    code: Annotated[str, Field(
        description="Python code to execute",
        json_schema_extra={"rawStringParam": True},
    )]
) -> str:
    return SESSION.execute(code)


if __name__ == "__main__":
    apply_resource_limits()
    mcp.run(show_banner=False, log_level='CRITICAL')
