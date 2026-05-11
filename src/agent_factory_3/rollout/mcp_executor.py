"""MCP client executor for tool execution and prompt loading."""

import json
import logging
import re
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel
from mcp.types import Tool
from fastmcp import Client as McpClient
from openai_harmony import ToolDescription, ToolNamespaceConfig

logger = logging.getLogger(__name__)

# =============================================================================
# Namespace Convention
# =============================================================================

SYSTEM_NAMESPACE_PREFIXES = ("builtin_", "system_")


def is_system_namespace(namespace: str) -> bool:
    """Check if namespace should go in system message (vs developer message)."""
    return namespace.startswith(SYSTEM_NAMESPACE_PREFIXES)


# =============================================================================
# Data Structures
# =============================================================================


class ToolMetadata(BaseModel):
    """Metadata about a tool for argument transformation."""
    namespace: str
    name: str
    raw_string_param: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None


class ToolErrorType(str, Enum):
    """Types of tool execution errors."""
    TIMEOUT = "timeout"
    EXCEPTION = "exception"


class ToolCallReport(BaseModel):
    """Report of a single tool call execution."""
    tool_name: str
    tool_input: str
    tool_output: str
    elapsed: float
    error: Optional[ToolErrorType] = None
    early_exit: bool = False
    structured_output: Optional[Dict[str, Any]] = None


class McpExecutorError(Exception):
    pass


class McpExecutor:
    """負責所有 MCP client 互動：工具執行、工具配置獲取、prompt 獲取

    Namespace 統一使用 mcp_config 的 key（而非 server 自報的 name）。
    """

    def __init__(
        self,
        clients: Optional[Dict[str, McpClient]],
        tool_call_timeout: float = 60.0,
        filter_by_include_in_prompt: bool = True,
        description_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.clients: Dict[str, McpClient] = clients or {}
        self.tool_call_timeout = tool_call_timeout
        self.filter_by_include_in_prompt = filter_by_include_in_prompt
        self._tool_metadata: Dict[str, ToolMetadata] = {}
        self._description_overrides = description_overrides or {}

    @property
    def has_clients(self) -> bool:
        return bool(self.clients)

    # -------------------------------------------------------------------------
    # Schema processing
    # -------------------------------------------------------------------------

    @staticmethod
    def _trim_schema(schema: dict) -> dict:
        """Convert MCP JSON Schema to Harmony's variant."""
        if "title" in schema:
            del schema["title"]
        if "default" in schema and schema["default"] is None:
            del schema["default"]
        if "anyOf" in schema:
            types = [
                type_dict["type"] for type_dict in schema["anyOf"]
                if "type" in type_dict and type_dict["type"] != 'null'
            ]
            if types:
                schema["type"] = types if len(types) > 1 else types[0]
            del schema["anyOf"]
        if "properties" in schema:
            schema["properties"] = {
                k: McpExecutor._trim_schema(v)
                for k, v in schema["properties"].items()
            }
        return schema

    def _post_process_tools(self, tool_list: List[Tool]) -> List[Tool]:
        """Adapt MCP tool descriptions for Harmony."""
        for tool in tool_list:
            tool.inputSchema = McpExecutor._trim_schema(tool.inputSchema)
        if self.filter_by_include_in_prompt:
            return [
                tool for tool in tool_list
                if tool.annotations is None or tool.annotations.include_in_prompt
            ]
        return tool_list

    # -------------------------------------------------------------------------
    # Tool discovery
    # -------------------------------------------------------------------------

    async def get_tool_configs(self) -> List[ToolNamespaceConfig]:
        """Get tool configurations from all MCP clients."""
        if not self.clients:
            return []

        tool_configs = []
        for namespace, client in self.clients.items():
            ns_overrides = self._description_overrides.get(namespace, {})
            tool_desc_overrides = ns_overrides.get("tools", {})

            override_instr = ns_overrides.get("instructions")
            server_description = (
                override_instr
                if override_instr is not None
                else client.initialize_result.instructions
            )
            tool_list = await client.list_tools()
            tool_list = self._post_process_tools(tool_list)

            tools = []
            for tool in tool_list:
                raw_string_param = self._extract_raw_string_param(tool.inputSchema)

                tool_key = f"{namespace}.{tool.name}"
                self._tool_metadata[tool_key] = ToolMetadata(
                    namespace=namespace,
                    name=tool.name,
                    raw_string_param=raw_string_param,
                    input_schema=tool.inputSchema,
                )

                tools.append(ToolDescription.new(
                    name=tool.name,
                    description=tool_desc_overrides.get(tool.name, tool.description),
                    parameters=tool.inputSchema,
                ))

            tool_config = ToolNamespaceConfig(
                name=namespace,
                description=server_description,
                tools=tools,
            )
            tool_configs.append(tool_config)

        return tool_configs

    @staticmethod
    def _extract_raw_string_param(input_schema: dict) -> Optional[str]:
        properties = input_schema.get("properties", {})
        for param_name, param_schema in properties.items():
            if param_schema.get("rawStringParam"):
                return param_name
        return None

    def _infer_single_string_param(self, tool_key: str) -> Optional[str]:
        """If a tool has exactly one required string parameter, return its name.

        This enables raw-string invocation for tools like python (code: str)
        without needing the non-standard rawStringParam schema extension.
        """
        metadata = self._tool_metadata.get(tool_key)
        if not metadata or not metadata.input_schema:
            return None
        schema = metadata.input_schema
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        # Must have exactly one required param and it must be a string
        if len(required) != 1:
            return None
        param_name = next(iter(required))
        param_schema = properties.get(param_name, {})
        if param_schema.get("type") == "string":
            return param_name
        return None

    # -------------------------------------------------------------------------
    # Content parsing
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_content(content: str) -> Dict[str, Any] | str | None:
        if not content:
            return None
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content

    # -------------------------------------------------------------------------
    # Tool reference resolution
    # -------------------------------------------------------------------------

    def _resolve_reference(self, reference: str) -> Tuple[str, str]:
        """Resolve tool reference to (namespace, name) tuple."""
        if "." in reference:
            namespace, name = reference.split(".", 1)
            return namespace, name

        for namespace in self.clients:
            if is_system_namespace(namespace) and namespace.endswith(f"_{reference}"):
                return namespace, reference

        raise ValueError(f"Unknown tool reference: {reference}")

    def _prepare_args(
        self, namespace: str, tool_name: str, raw_args: Dict[str, Any] | str | None
    ) -> Dict[str, Any]:
        if isinstance(raw_args, dict):
            return raw_args

        tool_key = f"{namespace}.{tool_name}"
        metadata = self._tool_metadata.get(tool_key)

        # Explicit rawStringParam in schema takes priority
        if metadata and metadata.raw_string_param and isinstance(raw_args, str):
            return {metadata.raw_string_param: raw_args}

        if isinstance(raw_args, str):
            # Try JSON first
            try:
                return json.loads(raw_args)
            except json.JSONDecodeError:
                pass

            # Auto-infer: tool has exactly one required string param → map directly
            single_param = self._infer_single_string_param(tool_key)
            if single_param:
                return {single_param: raw_args}

            raise ValueError(
                f"Tool {tool_key} received non-JSON string and has no single "
                f"string parameter to auto-map"
            )

        return raw_args or {}

    # -------------------------------------------------------------------------
    # Tool execution
    # -------------------------------------------------------------------------

    async def call_tool(self, recipient: str, content: str) -> ToolCallReport:
        """Call tool via appropriate MCP client."""
        start = time.perf_counter()
        tool_name = recipient
        try:
            namespace, tool_name = self._resolve_reference(recipient)
            if namespace not in self.clients:
                raise ValueError(f"Unknown tool namespace: {recipient}")

            tool_args = self._parse_content(content)
            tool_args = self._prepare_args(namespace, tool_name, tool_args)

            tool_outputs = await self.clients[namespace].call_tool(
                tool_name,
                arguments=tool_args,
                timeout=self.tool_call_timeout,
            )
            elapsed = time.perf_counter() - start

            text_outputs = [
                block.text for block in tool_outputs.content
                if getattr(block, "type", "") == "text"
            ]
            tool_output_text = "\n".join(text_outputs) if text_outputs else "<no text output>"

            structured_output = tool_outputs.structured_content or {}
            early_exit = structured_output.get("early_exit", False)

            return ToolCallReport(
                tool_name=tool_name,
                tool_input=content,
                tool_output=tool_output_text,
                elapsed=elapsed,
                early_exit=early_exit,
                structured_output=structured_output if structured_output else None,
            )
        except TimeoutError:
            return ToolCallReport(
                tool_name=tool_name,
                tool_input=content,
                tool_output="Tool execution timed out.",
                elapsed=time.perf_counter() - start,
                error=ToolErrorType.TIMEOUT,
            )
        except Exception as exc:
            return ToolCallReport(
                tool_name=tool_name,
                tool_input=content,
                tool_output=str(exc),
                elapsed=time.perf_counter() - start,
                error=ToolErrorType.EXCEPTION,
            )

    # -------------------------------------------------------------------------
    # Prompt loading
    # -------------------------------------------------------------------------

    async def get_prompt(self, reference: str) -> str:
        if not self.clients:
            raise McpExecutorError(
                f"Prompt '{reference}' requested but no MCP client configured."
            )
        namespace, prompt_name = self._resolve_reference(reference)
        prompt = await self.clients[namespace].get_prompt(prompt_name)
        return prompt.messages[0].content.text

    async def resolve_prompts(
        self,
        text: str,
        strict: bool = False,
    ) -> str:
        """解析文字中的 {{mcp_prompt:XXX}} 語法"""
        if not self.clients:
            return text

        pattern = re.compile(r'\{\{mcp_prompt:([^}]+)\}\}')
        prompt_names = pattern.findall(text)

        for prompt_name in set(prompt_names):
            try:
                prompt_result = await self.get_prompt(prompt_name)
                text = text.replace(
                    f"{{{{mcp_prompt:{prompt_name}}}}}",
                    prompt_result
                )
            except Exception as e:
                if strict:
                    raise
                else:
                    logger.warning(f"Failed to load MCP prompt '{prompt_name}': {e}")

        return text
