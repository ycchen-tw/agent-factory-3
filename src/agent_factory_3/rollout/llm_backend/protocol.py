"""LLM backend protocol definition."""

import asyncio
from typing import List, Optional, Protocol, runtime_checkable

from ..config import RecordConfig, SamplingParams
from .types import GenerationResult


@runtime_checkable
class LLMBackend(Protocol):
    """Protocol for LLM generation backends.

    統一介面：呼叫端透過 stream 參數手動控制是否用 streaming。
    - stream=True: streaming 模式，支援中斷（stop_event）、logprobs、top_logprobs
    - stream=False: non-streaming 模式，支援 routing_indices、usage
    """

    async def generate(
        self,
        prompt_tokens: List[int],
        max_tokens: int,
        stop_token_ids: List[int],
        sampling: SamplingParams,
        *,
        stream: bool = False,
        stop_event: Optional[asyncio.Event] = None,
        record: Optional[RecordConfig] = None,
    ) -> GenerationResult:
        """Generate completion tokens.

        Args:
            prompt_tokens: Input token IDs.
            max_tokens: Maximum tokens to generate.
            stop_token_ids: Token IDs that trigger generation stop.
            sampling: Sampling parameters.
            stream: Whether to use streaming mode (caller-controlled).
            stop_event: Event to signal early termination (streaming only).
            record: What to record. None = inference mode (no logprobs/routing).

        Returns:
            GenerationResult with requested fields populated.
        """
        ...

    async def close(self) -> None:
        """Clean up resources (close connections, etc.)."""
        ...
