"""Adapter for converting OpenAI Harmony format to internal types."""

from typing import Any, Dict, Optional

from openai_harmony import Conversation as HarmonyConversation
from openai_harmony import DeveloperContent as HarmonyDeveloperContent
from openai_harmony import Message as HarmonyMessage
from openai_harmony import Role as HarmonyRole
from openai_harmony import SystemContent as HarmonySystemContent
from openai_harmony import TextContent as HarmonyTextContent
from openai_harmony import ToolNamespaceConfig

from ..core.types import (
    Author,
    ContentPart,
    Conversation,
    DeveloperPart,
    Message,
    Role,
    SystemPart,
    TextPart,
)


class HarmonyAdapter:
    """Converts Harmony format to internal format."""

    _ROLE_MAP = {
        HarmonyRole.SYSTEM: Role.SYSTEM,
        HarmonyRole.DEVELOPER: Role.DEVELOPER,
        HarmonyRole.USER: Role.USER,
        HarmonyRole.ASSISTANT: Role.ASSISTANT,
        HarmonyRole.TOOL: Role.TOOL,
    }

    def adapt_conversation(self, conv: HarmonyConversation) -> Conversation:
        """Convert a Harmony Conversation to internal Conversation."""
        messages = [self.adapt_message(m) for m in conv.messages]
        return Conversation(messages=messages)

    def adapt_message(self, msg: HarmonyMessage) -> Message:
        """Convert a Harmony Message to internal Message."""
        author = Author(
            role=self._ROLE_MAP[msg.author.role],
            name=msg.author.name,
        )

        content = []
        for c in msg.content:
            part = self._adapt_content(c)
            if part is not None:
                content.append(part)

        return Message(
            author=author,
            content=content,
            recipient=msg.recipient,
            channel=msg.channel,
        )

    def _adapt_content(self, item) -> Optional[ContentPart]:
        """Convert a single content item to internal ContentPart."""
        if isinstance(item, HarmonyTextContent):
            return TextPart(text=item.text)

        if isinstance(item, HarmonySystemContent):
            return SystemPart(
                model_identity=item.model_identity,
                reasoning_effort=(
                    item.reasoning_effort.value if item.reasoning_effort else None
                ),
                conversation_start_date=item.conversation_start_date,
                knowledge_cutoff=item.knowledge_cutoff,
                tools=self._serialize_tools(item.tools),
            )

        if isinstance(item, HarmonyDeveloperContent):
            return DeveloperPart(
                instructions=item.instructions,
                tools=self._serialize_tools(item.tools),
            )

        # Duck typing fallback for text-like content
        if hasattr(item, "text"):
            return TextPart(text=item.text)

        # Unknown content type - return None to filter out
        return None

    def _serialize_tools(
        self, tools: Optional[Dict[str, ToolNamespaceConfig]]
    ) -> Optional[Dict[str, Any]]:
        """Serialize tools dict to JSON-compatible format."""
        if tools is None:
            return None
        return {
            name: config.model_dump() for name, config in tools.items()
        }
