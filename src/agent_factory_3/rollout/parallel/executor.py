"""Core rollout execution logic."""

import asyncio
import logging
import os
import time
import traceback
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from fastmcp import Client as McpClient

from openai_harmony import Conversation, Role

from ..config import LoopConfig
from ..llm_backend import SGLangBackend, VLLMBackend
from ..conversation_builder import ConversationBuilder
from ..mcp_executor import McpExecutor
from ..runner import UnifiedReactRunner
from ..types import ReactResult
from .config import RolloutConfig, RolloutResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Debug tracing for MCP lifecycle
# ---------------------------------------------------------------------------
_MCP_DEBUG = os.environ.get("MCP_DEBUG", "0") == "1"
_MCP_DEBUG_LOGGER: logging.Logger | None = None


def _get_mcp_logger() -> logging.Logger:
    """Get or create dedicated MCP debug logger that writes to a separate file."""
    global _MCP_DEBUG_LOGGER
    if _MCP_DEBUG_LOGGER is not None:
        return _MCP_DEBUG_LOGGER

    _MCP_DEBUG_LOGGER = logging.getLogger("mcp_debug")
    _MCP_DEBUG_LOGGER.setLevel(logging.DEBUG)
    _MCP_DEBUG_LOGGER.propagate = False  # don't pollute main log

    log_path = os.environ.get("MCP_DEBUG_LOG")
    if log_path:
        handler = logging.FileHandler(log_path, mode="a")
    else:
        handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [pid=%(process)d] %(message)s",
        datefmt="%H:%M:%S",
    ))
    _MCP_DEBUG_LOGGER.addHandler(handler)
    return _MCP_DEBUG_LOGGER


def _count_fds() -> int:
    try:
        return len(os.listdir(f"/proc/{os.getpid()}/fd"))
    except OSError:
        return -1


def _count_children() -> int:
    try:
        import subprocess
        r = subprocess.run(["pgrep", "-c", "-P", str(os.getpid())],
                           capture_output=True, text=True)
        return int(r.stdout.strip()) if r.stdout.strip() else 0
    except Exception:
        return -1


def _mcp_debug(rid: str, msg: str) -> None:
    if _MCP_DEBUG:
        _get_mcp_logger().info(f"[{rid}] {msg}")


async def execute_rollout(
    config: RolloutConfig,
    server_url: str,
    stop_event: Optional[asyncio.Event] = None,
    *,
    base_conversation: Optional[Conversation] = None,
    base_tokens: Optional[List[int]] = None,
) -> RolloutResult:
    """執行單個 rollout — 核心邏輯"""
    start_time = time.time()
    rid = config.rollout_id

    if base_tokens is not None and base_conversation is None:
        raise ValueError("base_tokens requires base_conversation")

    rollout_result = None
    try:
        async with AsyncExitStack() as stack:
            # 1. Setup MCP clients
            _mcp_debug(rid, f"spawn_start fds={_count_fds()} children={_count_children()}")
            mcp_clients = await _setup_mcp_clients(
                config.mcp_config, stack,
                mcp_spawn_interval=config.loop_config.mcp_spawn_interval,
                rollout_id=rid,
            )
            _mcp_debug(rid, f"spawn_done clients={len(mcp_clients)} "
                       f"fds={_count_fds()} children={_count_children()} "
                       f"elapsed={time.time() - start_time:.2f}s")

            # 2. Create McpExecutor
            mcp_executor = McpExecutor(
                clients=mcp_clients if mcp_clients else None,
                tool_call_timeout=config.loop_config.tool_call_timeout,
                filter_by_include_in_prompt=config.loop_config.filter_by_include_in_prompt,
                description_overrides=_extract_description_overrides(config.mcp_config),
            )

            # 3. Get tool configs
            tool_configs = await mcp_executor.get_tool_configs()
            logger.debug(f"[{rid}] Tool configs: {[tc.name for tc in tool_configs]}")

            builder = ConversationBuilder(config.conv_config)

            # 4. Create LLM backend.
            # cache_salt is populated by GroupConfigFactory based on RLFlow's
            # cache_salt_mode; None means "no salt".
            llm_backend = _create_backend(
                config.loop_config, server_url,
                cache_salt=config.loop_config.cache_salt,
            )

            try:
                # 5. Create runner and execute
                runner = UnifiedReactRunner(
                    config=config.loop_config,
                    llm_backend=llm_backend,
                    mcp_executor=mcp_executor,
                )

                initial_tokens = None
                if base_conversation is not None:
                    conversation = builder.append_user_message(base_conversation, config.user_prompt)
                    if base_tokens is not None:
                        user_msg = conversation.messages[-1]
                        user_msg_tokens = runner.encoding.render_conversation_for_completion(
                            Conversation.from_messages([user_msg]), Role.ASSISTANT,
                        )
                        initial_tokens = list(base_tokens) + user_msg_tokens
                else:
                    conversation = builder.build(config.user_prompt, tool_configs)

                result = await runner.run(
                    conversation,
                    initial_tokens=initial_tokens,
                    stop_event=stop_event,
                    record_config=config.record_config,
                    rollout_id=rid,
                )

                # 6. Build result (save before stack cleanup)
                _mcp_debug(rid, f"rollout_done elapsed={time.time() - start_time:.1f}s")
                rollout_result = _make_result(
                    config=config, result=result,
                    start_time=start_time, server_url=server_url,
                )

            finally:
                await llm_backend.close()

    except Exception as e:
        # If rollout already succeeded but MCP disconnect failed, keep the result
        if rollout_result is not None:
            logger.warning(f"[{rid}] MCP disconnect error (rollout data preserved): {e}")
            return rollout_result
        _mcp_debug(rid, f"rollout_error type={type(e).__name__} msg={e} "
                   f"fds={_count_fds()} children={_count_children()} "
                   f"elapsed={time.time() - start_time:.2f}s")
        logger.exception(f"[{rid}] Rollout failed: {e}")
        return _make_result(
            config=config, error=e,
            start_time=start_time, server_url=server_url,
        )

    return rollout_result


# =============================================================================
# Helper functions
# =============================================================================


def _create_backend(loop_config: LoopConfig, server_url: str, *, cache_salt: str | None = None):
    """Create the appropriate LLM backend."""
    if loop_config.backend == "vllm":
        if loop_config.model_name is None:
            raise ValueError("model_name is required for vLLM backend")
        return VLLMBackend(
            base_url=server_url,
            model_name=loop_config.model_name,
        )
    elif loop_config.backend == "sglang":
        return SGLangBackend(
            base_url=server_url,
            lora_path=loop_config.lora_adapter_name,
            cache_salt=cache_salt,
            num_hidden_layers=loop_config.num_hidden_layers,
            num_experts_per_tok=loop_config.num_experts_per_tok,
        )
    else:
        raise ValueError(f"Unknown backend: {loop_config.backend}")


class _CrossProcessSpawnLimiter:
    """Cross-process rate limiter using file lock.

    Ensures at most one MCP server spawn globally every `min_interval` seconds,
    across ALL worker processes. Uses fcntl.flock for cross-process synchronization.
    """

    def __init__(self, min_interval: float = 0.25, lock_path: str | None = None):
        self._min_interval = min_interval
        self._lock_path = lock_path or f"/tmp/.mcp_spawn_{os.getuid()}.lock"

    async def __aenter__(self):
        import fcntl
        import os
        loop = asyncio.get_event_loop()

        # Acquire file lock, read last spawn time, sleep if needed, write new time, release lock.
        # Lock is held only during the timing check — NOT during the actual MCP spawn.
        def _acquire_and_gate():
            fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o666)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                # Read last spawn time
                try:
                    os.lseek(fd, 0, os.SEEK_SET)
                    data = os.read(fd, 32)
                    last_spawn = float(data) if data else 0.0
                except (ValueError, OSError):
                    last_spawn = 0.0
                # Sleep if too soon
                now = time.time()
                wait = last_spawn + self._min_interval - now
                if wait > 0:
                    time.sleep(wait)
                # Write new timestamp
                now = time.time()
                os.lseek(fd, 0, os.SEEK_SET)
                os.ftruncate(fd, 0)
                os.write(fd, str(now).encode())
                # Release lock immediately
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

        await loop.run_in_executor(None, _acquire_and_gate)
        return self

    async def __aexit__(self, *exc_info):
        pass  # lock already released in __aenter__


_MCP_SPAWN_LIMITER: _CrossProcessSpawnLimiter | None = None


def _get_spawn_limiter(min_interval: float = 0.25) -> _CrossProcessSpawnLimiter:
    """Get or create the global cross-process spawn rate limiter."""
    global _MCP_SPAWN_LIMITER
    if _MCP_SPAWN_LIMITER is None:
        _MCP_SPAWN_LIMITER = _CrossProcessSpawnLimiter(min_interval=min_interval)
    return _MCP_SPAWN_LIMITER


def _extract_description_overrides(
    mcp_config: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """從 mcp_config 抽出 tool description override 設定。

    Returns:
        {namespace: {"instructions": str|None, "tools": {tool_name: description}}}
    """
    if not mcp_config or "mcpServers" not in mcp_config:
        return {}
    overrides: Dict[str, Dict[str, Any]] = {}
    for key, server_config in mcp_config["mcpServers"].items():
        tool_descs = server_config.get("toolDescriptions")
        instr_override = server_config.get("instructionsOverride")
        if tool_descs is not None or instr_override is not None:
            overrides[key] = {
                "instructions": instr_override,
                "tools": tool_descs or {},
            }
    return overrides


async def _setup_mcp_clients(
    mcp_config: Optional[Dict[str, Any]],
    stack: AsyncExitStack,
    *,
    mcp_spawn_interval: float = 0.25,
    rollout_id: str = "",
) -> Dict[str, McpClient]:
    """從配置創建 MCP clients"""
    if not mcp_config or "mcpServers" not in mcp_config:
        return {}

    clients: Dict[str, McpClient] = {}
    servers = mcp_config["mcpServers"]

    for config_key, server_config in servers.items():
        single_server_config = {
            "mcpServers": {
                config_key: server_config,
            }
        }
        client = McpClient(single_server_config)
        t0 = time.time()
        try:
            if mcp_spawn_interval > 0:
                async with _get_spawn_limiter(min_interval=mcp_spawn_interval):
                    _mcp_debug(rollout_id, f"client_enter key={config_key}")
                    await stack.enter_async_context(client)
            else:
                _mcp_debug(rollout_id, f"client_enter key={config_key}")
                await stack.enter_async_context(client)
            _mcp_debug(rollout_id, f"client_ready key={config_key} "
                       f"init_time={time.time() - t0:.2f}s")
        except Exception as e:
            _mcp_debug(rollout_id, f"client_FAIL key={config_key} "
                       f"type={type(e).__name__} msg={e} "
                       f"init_time={time.time() - t0:.2f}s "
                       f"fds={_count_fds()} children={_count_children()}")
            raise
        stack.push_async_callback(_safe_close_client, client, rollout_id, config_key)
        clients[config_key] = client

    return clients


async def _safe_close_client(
    client: McpClient,
    rollout_id: str = "",
    config_key: str = "",
) -> None:
    t0 = time.time()
    try:
        _mcp_debug(rollout_id, f"client_close key={config_key}")
        await client.close()
        _mcp_debug(rollout_id, f"client_closed key={config_key} "
                   f"time={time.time() - t0:.2f}s")
    except Exception as e:
        _mcp_debug(rollout_id, f"client_close_error key={config_key} "
                   f"type={type(e).__name__} msg={e}")
        logger.debug(f"Error closing MCP client (may already be closed): {e}")


def _make_result(
    config: RolloutConfig,
    start_time: float,
    server_url: str,
    result: Optional[ReactResult] = None,
    error: Optional[Exception] = None,
) -> RolloutResult:
    """構建 RolloutResult（成功或失敗統一入口）"""
    end_time = time.time()
    if error is not None:
        return RolloutResult(
            rollout_id=config.rollout_id,
            result=None,
            start_time=start_time,
            end_time=end_time,
            elapsed_time=end_time - start_time,
            server_url=server_url,
            success=False,
            error=str(error),
            traceback=traceback.format_exc(),
        )
    return RolloutResult(
        rollout_id=config.rollout_id,
        result=result,
        start_time=start_time,
        end_time=end_time,
        elapsed_time=end_time - start_time,
        server_url=server_url,
        success=True,
    )
