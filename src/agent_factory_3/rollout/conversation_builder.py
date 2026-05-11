"""Conversation builder for constructing initial prompts."""

from typing import List, Optional

from openai_harmony import (
    Conversation,
    DeveloperContent,
    Message,
    ReasoningEffort,
    Role,
    SystemContent,
    ToolNamespaceConfig,
)

from .config import ConversationConfig
from .mcp_executor import is_system_namespace


class ConversationBuilder:
    """無狀態的 Conversation 構建器（純同步，不依賴 MCP）

    工具分流邏輯：
    - builtin_* / system_* namespace → System message
    - 其他 namespace → Developer message
    """

    def __init__(self, config: ConversationConfig):
        self.config = config

    def build(
        self,
        user_prompt: str,
        tool_configs: Optional[List[ToolNamespaceConfig]] = None,
    ) -> Conversation:
        """構建初始 Conversation（第一輪對話）"""
        system_tools: List[ToolNamespaceConfig] = []
        dev_tools: List[ToolNamespaceConfig] = []

        for tc in (tool_configs or []):
            if is_system_namespace(tc.name):
                system_tools.append(tc)
            else:
                dev_tools.append(tc)

        messages = []

        system_msg = self._build_system_message(system_tools)
        messages.append(Message.from_role_and_content(Role.SYSTEM, system_msg))

        developer_msg = self._build_developer_message(dev_tools)
        if developer_msg is not None:
            messages.append(Message.from_role_and_content(Role.DEVELOPER, developer_msg))

        messages.append(Message.from_role_and_content(Role.USER, user_prompt))

        return Conversation.from_messages(messages)

    def append_user_message(
        self,
        conversation: Conversation,
        user_prompt: str,
    ) -> Conversation:
        """在現有 Conversation 上添加新的 user message（multi-turn）"""
        messages = list(conversation.messages)
        messages.append(Message.from_role_and_content(Role.USER, user_prompt))
        return Conversation.from_messages(messages)

    # -------------------------------------------------------------------------
    # System message
    # -------------------------------------------------------------------------

    def _build_system_message(
        self,
        system_tool_configs: List[ToolNamespaceConfig],
    ) -> SystemContent:
        system_msg = (
            SystemContent.new()
            .with_model_identity(self.config.model_identity)
            .with_conversation_start_date(self.config.conversation_start_date)
            .with_reasoning_effort(self._map_reasoning_effort(self.config.reasoning_effort))
        )

        for tc in system_tool_configs:
            harmony_ns = self._to_harmony_namespace(tc.name)

            # Python tool 特殊處理：不顯示 schema，讓 LLM 直接輸出 raw code
            if harmony_ns == "python":
                tools = []
            else:
                tools = tc.tools

            transformed_config = ToolNamespaceConfig(
                name=harmony_ns,
                description=tc.description,
                tools=tools,
            )
            system_msg = system_msg.with_tools(transformed_config)

        return system_msg

    @staticmethod
    def _to_harmony_namespace(namespace: str) -> str:
        for prefix in ("builtin_", "system_"):
            if namespace.startswith(prefix):
                return namespace[len(prefix):]
        return namespace

    @staticmethod
    def _map_reasoning_effort(value: str) -> ReasoningEffort:
        mapping = {
            "low": ReasoningEffort.LOW,
            "medium": ReasoningEffort.MEDIUM,
            "high": ReasoningEffort.HIGH,
        }
        return mapping.get(value.lower(), ReasoningEffort.MEDIUM)

    # -------------------------------------------------------------------------
    # Developer message
    # -------------------------------------------------------------------------

    def _build_developer_message(
        self,
        dev_tool_configs: List[ToolNamespaceConfig],
    ) -> Optional[DeveloperContent]:
        developer_msg = DeveloperContent.new()

        if self.config.dev_instructions:
            developer_msg = developer_msg.with_instructions(self.config.dev_instructions)

        for tc in dev_tool_configs:
            developer_msg = developer_msg.with_tools(tc)

        if developer_msg.instructions or developer_msg.tools:
            return developer_msg
        return None
