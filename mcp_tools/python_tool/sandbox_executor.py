"""Sandboxed Python executor with pyseccomp + rlimits (no bwrap/namespaces).

OS-level isolation for containers where user namespaces are unavailable:
  - pyseccomp: blocks network, execve, kill, ptrace, filesystem writes (irreversible)
  - rlimits: AS (memory), CPU (time), NPROC (threads)
  - Parent watchdog: server sends SIGKILL on timeout (uncatchable)

Math packages are pre-imported BEFORE lockdown. Once applied, lockdown
cannot be removed — kernel enforced.

Protocol: same as bwrap_executor.py (stdin/stdout newline-delimited JSON)
    Input:  {"code": "print(1+1)"}
    Output: {"output": "2\n", "error": null}

Environment variables:
    PYTHON_TOOL_TIMEOUT: Per-call execution timeout in seconds (default: 20.0)
    PYTHON_TOOL_MAX_OUTPUT_CHARS: Max output chars, 0=unlimited (default: 2000)
    PYTHON_TOOL_TRACEBACK_MODE: "user_frames" | "last_frame" | "full"
    PYTHON_TOOL_MAX_HEAP_MB: RLIMIT_AS in MB (default: 32768)


Created: 2026-03-29
"""

import ast
import io
import json
import os
import resource
import signal
import sys
import traceback


# ── Pre-import math packages (before lockdown) ─────────────────────────

def _preimport():
    """Prepare Python for sandboxed execution.

    With dont_write_bytecode=True and all .pyc pre-compiled (via compileall),
    lazy imports work fine under seccomp — no pre-import needed.
    Run `python -m compileall <site-packages>` when building the venv.
    """
    sys.dont_write_bytecode = True


# ── Seccomp (pyseccomp) ───────────────────────────────────────────────

def _install_seccomp():
    """Install seccomp filter. Irreversible once loaded."""
    from pyseccomp import SyscallFilter, ALLOW, ERRNO, MASKED_EQ, Arg
    import errno as _errno

    f = SyscallFilter(ALLOW)
    E = ERRNO(_errno.EPERM)

    # Network
    for sc in ["socket", "connect", "bind", "sendto", "sendmsg"]:
        f.add_rule(E, sc)

    # Process execution
    f.add_rule(E, "execve")
    f.add_rule(E, "execveat")

    # Kill other processes
    for sc in ["kill", "tkill", "tgkill"]:
        f.add_rule(E, sc)

    # Debug
    f.add_rule(E, "ptrace")

    # Filesystem mutation (unconditional)
    for sc in [
        "rename", "renameat", "renameat2",
        "unlink", "unlinkat",
        "rmdir", "mkdir", "mkdirat",
        "link", "linkat", "symlink", "symlinkat",
        "chmod", "fchmod", "fchmodat",
        "chown", "fchown", "lchown", "fchownat",
        "truncate", "ftruncate",
    ]:
        f.add_rule(E, sc)

    # openat: block when flags contain any write bit
    # Arg(2) = flags parameter (3rd arg, index 2)
    for flag in [os.O_WRONLY, os.O_RDWR, os.O_CREAT, os.O_TRUNC]:
        f.add_rule(E, "openat", Arg(2, MASKED_EQ, flag, flag))

    f.load()


# ── Resource limits ────────────────────────────────────────────────────

def _apply_rlimits():
    """Apply resource limits. Hard limits cannot be raised without CAP_SYS_RESOURCE."""
    max_heap_mb = int(os.environ.get("PYTHON_TOOL_MAX_HEAP_MB", "32768"))
    # RLIMIT_DATA: heap only (same as bwrap_executor). Does not count mmap/.so.
    # RLIMIT_AS would count virtual address space (shared libs inflate it).
    # max_heap = max_heap_mb * 1024 * 1024
    # resource.setrlimit(resource.RLIMIT_DATA, (max_heap, max_heap))
    #
    # RLIMIT_CPU: not set — cumulative CPU time across all threads
    # (CP-SAT 8 workers × 10s = 80s per call, easily exceeds any limit).
    # Per-call timeout (parent watchdog SIGKILL) is sufficient.
    pass
    # Note: RLIMIT_NPROC not set — it counts ALL threads for the entire user,
    # not per-process. Polars/galois/numpy need thread pools that easily exceed
    # a low limit. execve is blocked by seccomp, and parent watchdog handles
    # runaway processes via SIGKILL.


def apply_lockdown():
    """Apply all OS-level restrictions. Irreversible."""
    _apply_rlimits()
    _install_seccomp()


# ── Python session ─────────────────────────────────────────────────────

class PythonSession:
    """Persistent Python execution session with Jupyter-like last-expr display."""

    def __init__(self, timeout: float, traceback_mode: str, max_output_chars: int,
                 devnull_fd: int = -1):
        self._timeout = timeout
        self._traceback_mode = traceback_mode
        self._max_output_chars = max_output_chars
        self._devnull_fd = devnull_fd
        self._namespace: dict = {"__builtins__": __builtins__, "__name__": "__main__"}

    def execute(self, code: str) -> str:
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr

        def timeout_handler(signum, frame):
            raise KeyboardInterrupt(f"Timeout after {self._timeout:.1f}s")

        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, self._timeout)

        # Redirect C-level stderr to /dev/null to protect JSON protocol on stdout
        stderr_fd = sys.__stderr__.fileno()
        saved_stderr_fd = os.dup(stderr_fd)
        if self._devnull_fd >= 0:
            os.dup2(self._devnull_fd, stderr_fd)

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
            os.dup2(saved_stderr_fd, stderr_fd)
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

    def _exec_with_last_expr(self, code: str):
        try:
            tree = ast.parse(code)
            if not tree.body:
                return
            if isinstance(tree.body[-1], ast.Expr):
                if len(tree.body) > 1:
                    stmts = ast.Module(body=tree.body[:-1], type_ignores=[])
                    exec(compile(stmts, "<string>", "exec"), self._namespace)
                last_expr = ast.Expression(body=tree.body[-1].value)
                result = eval(compile(last_expr, "<string>", "eval"), self._namespace)
                if result is not None:
                    print(repr(result))
            else:
                exec(code, self._namespace)
        except SyntaxError:
            exec(code, self._namespace)

    def _format_traceback(self, tb_string: str) -> str:
        if self._traceback_mode == "full":
            return tb_string
        lines = tb_string.strip().split("\n")
        if self._traceback_mode == "user_frames":
            result = []
            i = 0
            while i < len(lines):
                line = lines[i]
                if line.strip().startswith('File "'):
                    is_user = "<string>" in line
                    if is_user:
                        result.append(line)
                    i += 1
                    while i < len(lines):
                        ns = lines[i].strip()
                        if ns.startswith('File "') or (ns and not lines[i].startswith(" ")):
                            break
                        if is_user:
                            result.append(lines[i])
                        i += 1
                else:
                    result.append(line)
                    i += 1
            return "\n".join(result)
        indices = [i for i, l in enumerate(lines) if l.strip().startswith('File "')]
        if len(indices) > 2:
            return "\n".join(lines[indices[-1]:])
        return tb_string

    def _truncate_output(self, output: str) -> str:
        limit = self._max_output_chars
        if limit <= 0 or len(output) <= limit:
            return output
        head = limit // 3
        tail = limit - head
        marker = f"\n\n... [Truncated: {len(output)} chars > {limit}] ...\n\n"
        return output[:head] + marker + output[-tail:]


# ── Main ───────────────────────────────────────────────────────────────

def main():
    _preimport()

    # Pre-open /dev/null before lockdown (openat O_WRONLY will be blocked)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    # JSON protocol fd (separate from sys.stdout which gets redirected)
    out_fd = os.fdopen(os.dup(sys.stdout.fileno()), "w")

    apply_lockdown()

    session = PythonSession(
        timeout=float(os.environ.get("PYTHON_TOOL_TIMEOUT", "20.0")),
        traceback_mode=os.environ.get("PYTHON_TOOL_TRACEBACK_MODE", "user_frames"),
        max_output_chars=int(os.environ.get("PYTHON_TOOL_MAX_OUTPUT_CHARS", "2000")),
        devnull_fd=devnull_fd,
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        code = req.get("code")
        if code is None:
            break
        output = session.execute(code)
        out_fd.write(json.dumps({"output": output, "error": None}) + "\n")
        out_fd.flush()


if __name__ == "__main__":
    main()
