"""Core data types for the Harmony Conversation Visualizer."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class Role(str, Enum):
    """Message author role."""

    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


# ============ Content Parts ============


@dataclass
class TextPart:
    """Plain text content."""

    text: str = ""
    type: str = field(default="text", init=False)


@dataclass
class SystemPart:
    """System configuration content."""

    model_identity: Optional[str] = None
    reasoning_effort: Optional[str] = None  # "Low", "Medium", "High"
    conversation_start_date: Optional[str] = None
    knowledge_cutoff: Optional[str] = None
    tools: Optional[Dict[str, Any]] = None  # tool namespaces
    type: str = field(default="system", init=False)


@dataclass
class DeveloperPart:
    """Developer configuration content."""

    instructions: Optional[str] = None
    tools: Optional[Dict[str, Any]] = None
    type: str = field(default="developer", init=False)


ContentPart = Union[TextPart, SystemPart, DeveloperPart]


# ============ Message ============


@dataclass
class Author:
    """Message author with role and optional name."""

    role: Role
    name: Optional[str] = None


@dataclass
class Message:
    """A single message in a conversation."""

    author: Author
    content: List[ContentPart]
    recipient: Optional[str] = None  # Tool call target
    channel: Optional[str] = None  # Message channel
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============ Conversation ============


@dataclass
class Conversation:
    """A single conversation containing multiple messages."""

    messages: List[Message]
    id: Optional[str] = None
    title: Optional[str] = None


@dataclass
class ConversationGroup:
    """A group of conversations (for comparing rollouts)."""

    group_id: str
    conversations: List[Conversation]


@dataclass
class ViewerData:
    """Complete data for the viewer."""

    groups: List[ConversationGroup]
    title: str = "Conversation Viewer"


# ============ Rollout Viewer Types ============


@dataclass
class RolloutData:
    """Data for a single rollout execution."""

    rollout_id: str
    success: bool
    conversation: Optional[Conversation] = None

    # Reward & Advantage
    weighted_reward: Optional[float] = None
    reward_components: Optional[Dict[str, float]] = None
    raw_advantage: Optional[float] = None
    advantage: Optional[float] = None  # Final advantage used for training (may be normalized)

    # Stats
    num_rounds: Optional[int] = None
    completion_tokens: Optional[int] = None
    elapsed_time: Optional[float] = None
    end_reason: Optional[str] = None

    # Weight version tracking
    weight_versions: Optional[List[str]] = None  # unique versions used, e.g. ["v0", "v1"]

    # Error info
    error: Optional[str] = None
    traceback: Optional[str] = None

    # Trainability (annotated by SampleProcessor)
    trainable: bool = True
    skip_reason: Optional[str] = None

    # Config snapshot (key parameters only)
    config_snapshot: Optional[Dict[str, Any]] = None


@dataclass
class GroupData:
    """A group of rollouts from the same prompt/task."""

    group_id: str
    rollouts: List[RolloutData]
    filter_reason: Optional[str] = None  # None, 'too_few', 'all_failed', 'all_solved', 'zero_loss'
    reward_baseline: Optional[float] = None

    @property
    def is_filtered(self) -> bool:
        return self.filter_reason is not None

    @property
    def group_size(self) -> int:
        return len(self.rollouts)


@dataclass
class RolloutViewerData:
    """Complete data for the rollout viewer."""

    groups: List[GroupData]
    title: str = "Rollout Viewer"
