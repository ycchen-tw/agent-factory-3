"""Secure Python Execution MCP Server

Provides safe Python code execution with:
- Session state management (variables persist across executions)
- Async execution support
- Security restrictions: no file I/O, network access, or subprocess execution
- Allowed imports: numpy, scipy, math, itertools, collections, json, re, etc.

Memory per session: ~7MB
"""
from typing import Annotated
from pydantic import Field
from fastmcp import FastMCP
from secure_python_session import AsyncSecureLightweightPythonSession

import fastmcp
fastmcp.settings.log_level = 'CRITICAL'

# Initialize secure Python session
PYTHON_SESSION = AsyncSecureLightweightPythonSession(timeout=10.0)

mcp = FastMCP("Secure Python Execution Server")

@mcp.tool(
    name="python",
    description="Execute Python code in a persistent session. Variables and state are preserved across executions. SECURITY: File I/O and network access are disabled. Allowed imports: numpy, scipy, math, itertools, functools, collections, json, re, etc.",
)
async def python(
    code: Annotated[str, Field(description="Python code to execute.")]
) -> str:
    """Execute Python code and return output."""
    return await PYTHON_SESSION.execute(code)

if __name__ == "__main__":
    try:
        mcp.run(show_banner=False)
    except Exception:
        # Suppress error stack on shutdown
        pass