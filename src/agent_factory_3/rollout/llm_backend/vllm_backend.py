"""vLLM backend using OpenAI-compatible API."""

import asyncio
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from ..config import RecordConfig, SamplingParams
from .types import GenerationResult


class VLLMBackend:
    """vLLM backend using OpenAI-compatible /v1/completions API."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "EMPTY",
        timeout: float = 3600,
    ):
        self.model_name = model_name
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=0,
        )

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
        if stream:
            return await self._generate_streaming(
                prompt_tokens, max_tokens, stop_token_ids, sampling,
                stop_event, record,
            )
        else:
            return await self._generate_non_streaming(
                prompt_tokens, max_tokens, stop_token_ids, sampling,
                record,
            )

    async def _generate_streaming(
        self,
        prompt_tokens: List[int],
        max_tokens: int,
        stop_token_ids: List[int],
        sampling: SamplingParams,
        stop_event: Optional[asyncio.Event],
        record: Optional[RecordConfig],
    ) -> GenerationResult:
        return_logprobs = record is not None and record.logprobs
        top_logprobs_k = record.top_logprobs if record else 0

        extra_body: Dict[str, Any] = {
            "stop_token_ids": stop_token_ids,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            "return_token_ids": True,
        }
        if sampling.min_p is not None:
            extra_body["min_p"] = sampling.min_p
        if top_logprobs_k > 0:
            extra_body["return_tokens_as_token_ids"] = True

        logprobs_param = max(1 if return_logprobs else 0, top_logprobs_k)

        api_stream = await self.client.completions.create(
            model=self.model_name,
            prompt=list(prompt_tokens),
            max_tokens=max_tokens,
            temperature=sampling.temperature,
            seed=sampling.seed,
            stream=True,
            logprobs=logprobs_param if logprobs_param > 0 else None,
            extra_body=extra_body,
        )

        token_ids: List[int] = []
        logprobs: Optional[List[float]] = [] if return_logprobs else None
        top_logprobs: Optional[List[Dict[int, float]]] = [] if top_logprobs_k > 0 else None
        finish_reason: Optional[str] = None

        try:
            async for chunk in api_stream:
                if stop_event is not None and stop_event.is_set():
                    return GenerationResult(
                        token_ids=token_ids,
                        finish_reason="stop",
                        logprobs=logprobs if logprobs else None,
                        top_logprobs=top_logprobs if top_logprobs else None,
                        routing_indices=None,
                        usage=None,
                    )

                if chunk.choices:
                    choice = chunk.choices[0]
                    if choice.token_ids:
                        token_ids.extend(choice.token_ids)
                    if choice.logprobs:
                        if return_logprobs and choice.logprobs.token_logprobs:
                            logprobs.extend(choice.logprobs.token_logprobs)
                        if top_logprobs_k > 0 and choice.logprobs.top_logprobs:
                            for pos_lps in choice.logprobs.top_logprobs:
                                parsed = {
                                    int(k.split(":")[1]): v
                                    for k, v in pos_lps.items()
                                }
                                top_logprobs.append(parsed)
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
        finally:
            await api_stream.close()

        return GenerationResult(
            token_ids=token_ids,
            finish_reason=finish_reason,
            logprobs=logprobs if logprobs else None,
            top_logprobs=top_logprobs if top_logprobs else None,
            routing_indices=None,
            cached_tokens=None,
            usage=None,
        )

    async def _generate_non_streaming(
        self,
        prompt_tokens: List[int],
        max_tokens: int,
        stop_token_ids: List[int],
        sampling: SamplingParams,
        record: Optional[RecordConfig],
    ) -> GenerationResult:
        return_logprobs = record is not None and record.logprobs
        return_routing = record is not None and record.routing_indices
        return_usage = record is not None and record.usage

        extra_body: Dict[str, Any] = {
            "stop_token_ids": stop_token_ids,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k,
            "return_token_ids": True,
        }
        if sampling.min_p is not None:
            extra_body["min_p"] = sampling.min_p

        response = await self.client.completions.create(
            model=self.model_name,
            prompt=list(prompt_tokens),
            max_tokens=max_tokens,
            temperature=sampling.temperature,
            seed=sampling.seed,
            stream=False,
            logprobs=1 if return_logprobs else None,
            extra_body=extra_body,
        )

        choice = response.choices[0]
        token_ids = choice.token_ids

        # Extract logprobs
        logprobs = None
        if return_logprobs:
            if choice.logprobs is None:
                raise RuntimeError(
                    "return_logprobs=True but API response lacks 'logprobs'. "
                    "Check if the model/API supports logprobs."
                )
            raw_logprobs = choice.logprobs.token_logprobs
            if len(raw_logprobs) != len(token_ids):
                raise RuntimeError(
                    f"logprobs/tokens length mismatch: "
                    f"logprobs={len(raw_logprobs)}, tokens={len(token_ids)}"
                )
            logprobs = raw_logprobs

        # Extract routing_indices
        routing_indices = None
        if return_routing:
            routing_info = choice.moe_routing_info
            if routing_info is None:
                raise RuntimeError(
                    "routing_indices=True but API response lacks 'moe_routing_info'. "
                    "Model/API may not support MoE routing info."
                )
            routing_indices = routing_info

        # Extract usage
        usage = None
        if return_usage and response.usage is not None:
            usage = response.usage.model_dump(exclude_none=True)

        return GenerationResult(
            token_ids=token_ids,
            finish_reason=choice.finish_reason,
            logprobs=logprobs,
            top_logprobs=None,
            routing_indices=routing_indices,
            cached_tokens=None,
            usage=usage,
        )

    async def close(self) -> None:
        await self.client.close()
