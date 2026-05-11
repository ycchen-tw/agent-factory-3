"""LLM backend response types."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class GenerationResult:
    """Unified generation result for all backends.

    Attributes:
        token_ids: Generated token IDs.
        finish_reason: "stop" | "length" | None
        logprobs: Per-token log probabilities (training mode only).
        top_logprobs: Per-token top-k logprobs, list of {token_id: logprob}.
        routing_indices: MoE routing indices, shape (seqlen, layers, topk).
        usage: Token usage statistics.
    """

    token_ids: List[int]
    finish_reason: Optional[str]
    logprobs: Optional[List[float]]
    top_logprobs: Optional[List[Dict[int, float]]]
    routing_indices: Optional[Any]  # np.ndarray[uint8] shape [T, L, K] or None
    cached_tokens: Optional[int]
    usage: Optional[Dict[str, Any]]
    entropy: Optional[List[float]] = None
    weight_version: Optional[str] = None
