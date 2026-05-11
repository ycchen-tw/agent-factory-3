"""Secure lightweight Python execution session - Optimized version.

Balanced security with practical needs for ARC-AGI:
- Allows numpy, scipy for algorithms
- Blocks file I/O, network, subprocess
- Minimal overhead (~0.1ms per execution)

Memory: ~7MB per session
"""

import io
import sys
import traceback
import signal
import threading
import inspect
from typing import Optional


class ExecutionTimeout(Exception):
    """Raised when code execution times out."""
    pass


class SecurityViolation(Exception):
    """Raised when code attempts restricted operations."""
    pass


class SecureLightweightPythonSession:
    """Secure Python execution with balanced restrictions.
    
    Security features:
    - No file I/O (blocked via import restrictions)
    - No network access (blocked via import restrictions)
    - No subprocess execution
    - Allows numpy, scipy, itertools, etc. for algorithms
    
    Example:
        >>> session = SecureLightweightPythonSession(timeout=60.0)
        >>> session.execute("import numpy as np; print(np.array([1,2,3]))")
        '[1 2 3]\\n'
    """

    def __init__(self, timeout: float = 60.0):
        """
        Args:
            timeout: Default execution timeout in seconds
        """
        self._timeout = timeout
        self._namespace = self._create_secure_namespace()
        
    def _create_safe_builtins(self) -> dict:
        """Create minimal safe builtins dictionary.
        
        Allows most Python features except:
        - Direct file I/O: open
        - Code execution: eval, exec, compile
        - Dangerous reflection: getattr, setattr (allow hasattr for debugging)
        """
        import builtins
        
        # Start with all builtins
        safe_dict = {}
        for name in dir(builtins):
            if not name.startswith('_') or name in ('__build_class__', '__import__'):
                safe_dict[name] = getattr(builtins, name)
        
        # Custom __import__ with restrictions
        safe_dict['__import__'] = self._create_safe_import()
        
        # Block the most dangerous builtins
        dangerous = {
            'open',      # File I/O
            'eval',      # Code execution
            'exec',      # Code execution  
            'compile',   # Code execution
            'getattr',   # Can bypass restrictions
            'setattr',   # Can modify restricted objects
            'delattr',   # Can delete restrictions
            'vars',      # Can access __dict__
            'globals',   # Can access global scope
            'locals',    # Can access local scope
        }
        
        for name in dangerous:
            safe_dict[name] = None
        
        return safe_dict
    
    def _create_safe_import(self):
        """Create import function with module restrictions.
        
        Uses stack inspection to distinguish user code from library internals.
        """
        import importlib  # Import once at creation time
        
        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            # Blocked modules for direct user access
            blocked_modules = {
                'os', 'subprocess', 'multiprocessing', 'pty',
                'socket', 'urllib', 'http', 'ftplib', 'smtplib',
                'pathlib', 'shutil', 'tempfile', 'fileinput', 'glob',
                'pickle', 'marshal', 'shelve', 'dbm',
                'gc', 'ctypes', 'cffi',
                'pandas', 'matplotlib', 'seaborn', 'plotly',
                'importlib',  # Can bypass import restrictions
                'sys',        # Can access frames and modules
            }
            
            base_module = name.split('.')[0]
            
            if base_module in blocked_modules:
                # Check if this is from user code (exec'd code)
                # User code will have filename '<stdin>' or '<string>' in the stack
                frame = inspect.currentframe()
                is_user_code = False
                try:
                    # Walk up the stack
                    for _ in range(10):  # Check up to 10 frames
                        if frame is None:
                            break
                        filename = frame.f_code.co_filename
                        if filename in ('<stdin>', '<string>'):
                            is_user_code = True
                            break
                        frame = frame.f_back
                finally:
                    del frame  # Avoid reference cycles
                
                if is_user_code:
                    raise SecurityViolation(
                        f"Direct import of '{name}' is blocked for security.\n"
                        f"Blocked: file I/O, network, subprocess, plotting.\n"
                        f"Allowed: numpy, scipy, math, itertools, collections, etc."
                    )
            
            return importlib.__import__(name, globals, locals, fromlist, level)
        
        return safe_import
    
    def _create_secure_namespace(self) -> dict:
        """Create isolated namespace with safe builtins."""
        return {
            "__builtins__": self._create_safe_builtins(),
            "__name__": "__main__",
        }
    
    @staticmethod
    def _extract_error_summary(tb_string: str) -> str:
        """Extract concise error summary from full traceback."""
        lines = tb_string.strip().split('\n')
        if not lines:
            return tb_string

        last_line = lines[-1]

        if 'SyntaxError' in last_line or 'IndentationError' in last_line:
            num_lines = min(4, len(lines))
            return '\n'.join(lines[-num_lines:])

        frame_indices = [i for i, line in enumerate(lines)
                         if line.strip().startswith('File')]

        if len(frame_indices) <= 2:
            return tb_string

        last_frame_start = frame_indices[-1]
        return '\n'.join(lines[last_frame_start:])

    def execute(self, code: str, timeout: float | None = None) -> str:
        """Execute code and return combined stdout/stderr output.

        Args:
            code: Python code to execute
            timeout: Optional timeout override for this execution

        Returns:
            Combined stdout/stderr output as string

        Raises:
            ExecutionTimeout: If execution exceeds timeout
            SecurityViolation: If code attempts restricted operations
        """
        # Quick security checks (very fast string search)
        if '__subclasses__' in code:
            return "[SECURITY] Access to __subclasses__ is blocked"
        
        effective_timeout = timeout or self._timeout

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        is_main_thread = threading.current_thread() is threading.main_thread()

        def timeout_handler(signum, frame):
            raise ExecutionTimeout(f"Execution timeout after {effective_timeout}s")

        try:
            sys.stdout = stdout_buffer
            sys.stderr = stderr_buffer

            old_handler = None
            if is_main_thread:
                try:
                    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(int(effective_timeout))
                except (ValueError, OSError):
                    is_main_thread = False

            try:
                self._exec_with_last_expr(code)
            except ExecutionTimeout:
                raise
            except SecurityViolation as e:
                stderr_buffer.write(f"[SECURITY] {e}\n")
            except SystemExit as e:
                stderr_buffer.write(f"SystemExit: code {e.code}\n")
            except Exception:
                tb = traceback.format_exc()
                tb_summary = self._extract_error_summary(tb)
                stderr_buffer.write(tb_summary)
            finally:
                if is_main_thread and old_handler is not None:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        stdout = stdout_buffer.getvalue()
        stderr = stderr_buffer.getvalue()

        if stderr:
            stdout = f"{stdout.rstrip()}\n{stderr}" if stdout else stderr

        if not stdout.strip():
            stdout = "[WARN] No output. Use print() to see output."

        return stdout

    def _exec_with_last_expr(self, code: str):
        """Execute code and display last expression result (like Jupyter)."""
        import ast
        try:
            tree = ast.parse(code)

            if not tree.body:
                return

            if isinstance(tree.body[-1], ast.Expr):
                if len(tree.body) > 1:
                    statements = ast.Module(body=tree.body[:-1], type_ignores=[])
                    exec(compile(statements, '<stdin>', 'exec'), self._namespace)

                last_expr = ast.Expression(body=tree.body[-1].value)
                result = eval(compile(last_expr, '<stdin>', 'eval'), self._namespace)

                if result is not None:
                    print(repr(result))
            else:
                exec(code, self._namespace)
        except SyntaxError:
            exec(code, self._namespace)

    def close(self):
        """Clean up resources."""
        if self._namespace is not None:
            self._namespace.clear()
            self._namespace = None

    def __del__(self):
        if hasattr(self, '_namespace') and self._namespace is not None:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# Async wrapper
class AsyncSecureLightweightPythonSession:
    """Async wrapper for SecureLightweightPythonSession."""

    def __init__(self, timeout: float = 60.0):
        import asyncio
        self._session = SecureLightweightPythonSession(timeout)
        self._lock = asyncio.Lock()

    async def execute(self, code: str, timeout: float | None = None) -> str:
        """Execute code asynchronously in thread pool."""
        import asyncio
        async with self._lock:
            effective_timeout = timeout or self._session._timeout
            try:
                task = asyncio.to_thread(self._session.execute, code, timeout)
                return await asyncio.wait_for(task, timeout=effective_timeout * 1.5)
            except asyncio.TimeoutError:
                return f"[ERROR] Execution timeout after {effective_timeout}s"

    def close(self):
        self._session.close()

    def __del__(self):
        if hasattr(self, '_session') and self._session is not None:
            self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


if __name__ == "__main__":
    print("=== Testing Optimized Secure Session ===\n")
    
    session = SecureLightweightPythonSession(timeout=5.0)
    
    # Test 1: numpy (should work now!)
    print("Test 1: numpy support")
    result = session.execute("""
import numpy as np
arr = np.array([[1, 2, 3], [4, 5, 6]])
print('Array:', arr)
print('Sum:', arr.sum())
""")
    print(result)
    
    # Test 2: scipy (should work)
    print("\nTest 2: scipy support")
    result = session.execute("""
try:
    from scipy import signal
    print('scipy available!')
except ImportError:
    print('scipy not installed (OK)')
""")
    print(result)
    
    # Test 3: File I/O still blocked
    print("\nTest 3: File I/O still blocked")
    result = session.execute("open('test.txt', 'w')")
    print(result)
    
    # Test 4: pandas blocked (can read/write files)
    print("\nTest 4: pandas blocked")
    result = session.execute("import pandas as pd")
    print(result)
    
    # Test 5: Performance check
    print("\nTest 5: Performance")
    import time
    times = []
    for _ in range(100):
        start = time.perf_counter()
        session.execute("x = [i**2 for i in range(100)]")
        times.append((time.perf_counter() - start) * 1000)
    print(f"Average: {sum(times)/len(times):.3f}ms")
    
    session.close()
    print("\n=== Tests completed ===")