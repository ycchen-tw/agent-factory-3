"""Lightweight Python executor for bwrap sandbox.

No MCP dependencies. Communicates via stdin/stdout newline-delimited JSON.
Security is enforced by bwrap (OS-level), not Python-level restrictions.

Protocol:
    Input:  {"code": "print(1+1)"}
    Output: {"output": "2\n", "error": null}
    Shutdown: stdin EOF or {"code": null}

Environment variables:
    PYTHON_TOOL_TIMEOUT: Execution timeout in seconds (default: 10.0)
    PYTHON_TOOL_PRERUN: Python code to execute at startup (e.g., imports)
    PYTHON_TOOL_MAX_OUTPUT_CHARS: Max output characters (default: 0 = unlimited)
    PYTHON_TOOL_TRACEBACK_MODE: "user_frames" | "last_frame" | "full" (default: "user_frames")
"""

import ast
import io
import json
import os
import resource
import signal
import sys
import traceback


def apply_resource_limits():
    """Apply OS-level resource limits."""
    max_heap_mb = int(os.environ.get("PYTHON_TOOL_MAX_HEAP_MB", "16384"))
    max_heap = max_heap_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_DATA, (max_heap, max_heap))
    # Note: RLIMIT_NPROC not set because numpy/scipy need threads for init.
    # Fork prevention is handled by bwrap --unshare-pid.


class PythonSession:
    """Python execution session with Jupyter-like behavior.

    - Last expression auto-printed
    - Namespace persists across calls
    - Timeout via SIGALRM raises KeyboardInterrupt (state preserved)
    """

    def __init__(
        self,
        timeout: float = 10.0,
        prerun: str = "",
        traceback_mode: str = "user_frames",
        max_output_chars: int = 0,
    ):
        self._timeout = timeout
        self._traceback_mode = traceback_mode
        self._max_output_chars = max_output_chars
        self._namespace: dict = {"__builtins__": __builtins__, "__name__": "__main__"}
        if prerun.strip():
            exec(prerun, self._namespace)

    def execute(self, code: str, timeout: float | None = None) -> str:
        """Execute code with timeout."""
        effective_timeout = timeout or self._timeout
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr

        def timeout_handler(signum, frame):
            raise KeyboardInterrupt(f"Timeout after {effective_timeout:.1f}s")

        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, effective_timeout)

        # Redirect fd 2 to /dev/null to prevent C library stderr from
        # corrupting the JSON protocol on fd 1.
        # Note: fd 1 is NOT redirected because scipy.optimize import hangs
        # when fd 1 is dup2'd. C library stdout on fd 1 is captured by the
        # server (server_v4 reads stdout via asyncio pipe).
        stderr_fd = sys.__stderr__.fileno()
        saved_stderr_fd = os.dup(stderr_fd)
        devnull = os.open(os.devnull, os.O_WRONLY)
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
        """Execute code, print last expression result (like Jupyter)."""
        try:
            tree = ast.parse(code)
            if not tree.body:
                return
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

    def _format_traceback(self, tb_string: str) -> str:
        """Format traceback based on configured mode."""
        if self._traceback_mode == "full":
            return tb_string

        lines = tb_string.strip().split('\n')

        if self._traceback_mode == "user_frames":
            result = []
            i = 0
            while i < len(lines):
                line = lines[i]
                stripped = line.strip()
                if stripped.startswith('File "'):
                    is_user_frame = '<string>' in line
                    if is_user_frame:
                        result.append(line)
                    i += 1
                    while i < len(lines):
                        next_stripped = lines[i].strip()
                        if next_stripped.startswith('File "') or (next_stripped and not lines[i].startswith(' ')):
                            break
                        if is_user_frame:
                            result.append(lines[i])
                        i += 1
                    continue
                else:
                    result.append(line)
                    i += 1
            return '\n'.join(result)

        # last_frame
        frame_indices = [i for i, line in enumerate(lines) if line.strip().startswith('File "')]
        if len(frame_indices) > 2:
            return '\n'.join(lines[frame_indices[-1]:])
        return tb_string

    def _truncate_output(self, output: str) -> str:
        """Truncate middle of output if exceeds limit."""
        limit = self._max_output_chars
        if limit <= 0 or len(output) <= limit:
            return output
        head_size = limit // 3
        tail_size = limit - head_size
        marker = f"\n\n... [Truncated: {len(output)} chars exceeded limit of {limit}] ...\n\n"
        return output[:head_size] + marker + output[-tail_size:]


def main():
    apply_resource_limits()

    session = PythonSession(
        timeout=float(os.environ.get("PYTHON_TOOL_TIMEOUT", "10.0")),
        prerun=os.environ.get("PYTHON_TOOL_PRERUN", ""),
        traceback_mode=os.environ.get("PYTHON_TOOL_TRACEBACK_MODE", "user_frames"),
        max_output_chars=int(os.environ.get("PYTHON_TOOL_MAX_OUTPUT_CHARS", "0")),
    )

    # Use fd 1 for JSON protocol, ensure unbuffered
    out_fd = os.fdopen(os.dup(sys.stdout.fileno()), 'w')

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
        resp = json.dumps({"output": output, "error": None})
        out_fd.write(resp + "\n")
        out_fd.flush()


if __name__ == "__main__":
    main()
