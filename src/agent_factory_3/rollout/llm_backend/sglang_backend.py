"""sglang backend using native /generate endpoint."""

import asyncio
import base64
import json
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np

from ..config import RecordConfig, SamplingParams
from .types import GenerationResult


class SGLangBackend:
    """sglang backend using native /generate endpoint.

    sglang 一個 server = 一個 model，不需要 model_name。
    LoRA 透過 per-request lora_path 指定。
    """

    def __init__(
        self,
        base_url: str,
        *,
        lora_path: Optional[str] = None,
        cache_salt: Optional[str] = None,
        stream_output: bool = True,
        num_hidden_layers: Optional[int] = None,
        num_experts_per_tok: Optional[int] = None,
        timeout: float = 3600,
        max_connections: int = 100,
    ):
        """Initialize sglang backend.

        Args:
            base_url: Server URL.
            lora_path: Per-request LoRA adapter path.
            stream_output: Whether server was started with --stream-output.
                True (default): output_ids is incremental (only new tokens per chunk).
                False: output_ids is cumulative (full list each chunk).
            num_hidden_layers: For routing indices reshape.
            num_experts_per_tok: For routing indices reshape.
        """
        # Strip /v1 suffix if present (sglang doesn't use it)
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.lora_path = lora_path
        self.cache_salt = cache_salt
        self.stream_output = stream_output
        self.num_layers = num_hidden_layers
        self.num_experts_per_tok = num_experts_per_tok
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.connector = aiohttp.TCPConnector(
            limit=max_connections,
            ttl_dns_cache=300,
            keepalive_timeout=30,  # must be < server's SGLANG_TIMEOUT_KEEP_ALIVE to avoid stale connection reuse
        )
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                connector=self.connector,
                timeout=self.timeout,
            )
        return self._session

    def _build_sampling_params(self, sampling: SamplingParams, max_tokens: int, stop_token_ids: List[int]) -> Dict[str, Any]:
        """Build sglang sampling_params dict from SamplingParams."""
        params: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "top_k": sampling.top_k if sampling.top_k > 0 else -1,
            "stop_token_ids": stop_token_ids,
        }
        if sampling.min_p is not None:
            params["min_p"] = sampling.min_p
        if sampling.seed is not None:
            params["sampling_seed"] = sampling.seed
        return params

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
        return_entropy = record is not None and record.entropy
        top_logprobs_k = record.top_logprobs if record else 0

        payload: Dict[str, Any] = {
            "input_ids": prompt_tokens,
            "sampling_params": self._build_sampling_params(sampling, max_tokens, stop_token_ids),
            "stream": True,
            "return_logprob": return_logprobs or top_logprobs_k > 0,
            "logprob_start_len": len(prompt_tokens),
            "return_entropy": return_entropy,
            "top_logprobs_num": top_logprobs_k,
            "return_routed_experts": False,
        }
        if self.lora_path is not None:
            payload["lora_path"] = self.lora_path
        if self.cache_salt is not None:
            payload["extra_key"] = self.cache_salt

        session = await self._get_session()
        url = f"{self.base_url}/generate"

        # sglang streaming 有兩種模式（由 server --stream-output flag 決定）：
        # - stream_output=True:  output_ids 是 incremental（只有新 tokens）
        # - stream_output=False: output_ids 是 cumulative（完整 list）
        # logprobs/top_logprobs 在兩種模式下都是 cumulative。
        token_ids: List[int] = []
        finish_reason: Optional[str] = None
        last_output_token_logprobs: Optional[List] = None
        last_output_top_logprobs: Optional[List] = None
        last_output_token_entropy: Optional[List[float]] = None
        weight_version: Optional[str] = None
        incremental = self.stream_output

        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()

            buffer = b""
            done = False
            async for chunk in resp.content.iter_any():
                if done:
                    break
                if stop_event is not None and stop_event.is_set():
                    return self._build_streaming_result(
                        token_ids, "stop", return_logprobs, return_entropy,
                        top_logprobs_k,
                        last_output_token_logprobs, last_output_top_logprobs,
                        last_output_token_entropy, weight_version,
                    )

                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line_str = line.decode("utf-8").strip()
                    if not line_str:
                        continue
                    if line_str == "data: [DONE]":
                        done = True
                        break
                    if line_str.startswith("data: "):
                        data = json.loads(line_str[6:])
                        if "output_ids" in data:
                            if incremental:
                                token_ids.extend(data["output_ids"])
                            else:
                                token_ids = data["output_ids"]
                        if "meta_info" in data:
                            meta = data["meta_info"]
                            if "finish_reason" in meta:
                                finish_reason = self._normalize_finish_reason(meta["finish_reason"])
                            if "weight_version" in meta:
                                weight_version = meta["weight_version"]
                            # logprobs 都是 cumulative，取最新快照
                            if "output_token_logprobs" in meta:
                                last_output_token_logprobs = meta["output_token_logprobs"]
                            if "output_top_logprobs" in meta:
                                last_output_top_logprobs = meta["output_top_logprobs"]
                            if "output_token_entropy" in meta:
                                last_output_token_entropy = meta["output_token_entropy"]

        return self._build_streaming_result(
            token_ids, finish_reason, return_logprobs, return_entropy,
            top_logprobs_k,
            last_output_token_logprobs, last_output_top_logprobs,
            last_output_token_entropy, weight_version,
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
        return_entropy = record is not None and record.entropy
        return_routing = record is not None and record.routing_indices
        return_usage = record is not None and record.usage

        payload: Dict[str, Any] = {
            "input_ids": prompt_tokens,
            "sampling_params": self._build_sampling_params(sampling, max_tokens, stop_token_ids),
            "stream": False,
            "return_logprob": return_logprobs,
            "logprob_start_len": len(prompt_tokens),
            "return_entropy": return_entropy,
            "return_routed_experts": return_routing,
        }
        if self.lora_path is not None:
            payload["lora_path"] = self.lora_path
        if self.cache_salt is not None:
            payload["extra_key"] = self.cache_salt

        session = await self._get_session()
        url = f"{self.base_url}/generate"

        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()

        token_ids = data["output_ids"]
        meta_info = data.get("meta_info", {})

        finish_reason = self._normalize_finish_reason(meta_info.get("finish_reason"))

        # Extract logprobs
        logprobs = None
        if return_logprobs:
            output_logprobs = meta_info.get("output_token_logprobs")
            if output_logprobs is None:
                raise RuntimeError(
                    "return_logprob=True but response lacks 'output_token_logprobs'. "
                    "Check if the model/API supports logprobs."
                )
            logprobs = [lp[0] for lp in output_logprobs]
            if len(logprobs) != len(token_ids):
                raise RuntimeError(
                    f"logprobs/tokens length mismatch: "
                    f"logprobs={len(logprobs)}, tokens={len(token_ids)}"
                )

        # Extract entropy
        entropy = None
        if return_entropy:
            output_entropy = meta_info.get("output_token_entropy")
            if output_entropy is None:
                raise RuntimeError(
                    "return_entropy=True but response lacks 'output_token_entropy'. "
                    "Check if the sglang server supports return_entropy."
                )
            entropy = output_entropy  # list[float], no extraction needed
            if len(entropy) != len(token_ids):
                raise RuntimeError(
                    f"entropy/tokens length mismatch: "
                    f"entropy={len(entropy)}, tokens={len(token_ids)}"
                )

        # Extract routing_indices
        routing_indices = None
        cached_tokens_count = meta_info.get("cached_tokens")
        if return_routing:
            routed_experts_b64 = meta_info.get("routed_experts")
            if routed_experts_b64 is None:
                raise RuntimeError(
                    "return_routed_experts=True but response lacks 'routed_experts'. "
                    "Make sure server is started with --enable-return-routed-experts."
                )
            routing_indices = self._decode_routing_indices(routed_experts_b64)

        # Extract usage
        usage = None
        if return_usage:
            completion_tokens = meta_info.get("completion_tokens")
            prompt_tokens_count = meta_info.get("prompt_tokens")
            if completion_tokens is not None:
                usage = {"completion_tokens": completion_tokens}
                if prompt_tokens_count is not None:
                    usage["prompt_tokens"] = prompt_tokens_count
                if cached_tokens_count is not None:
                    usage["cached_tokens"] = cached_tokens_count

        return GenerationResult(
            token_ids=token_ids,
            finish_reason=finish_reason,
            logprobs=logprobs,
            top_logprobs=None,
            entropy=entropy,
            routing_indices=routing_indices,
            cached_tokens=cached_tokens_count,
            usage=usage,
            weight_version=meta_info.get("weight_version"),
        )

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        if self.connector is not None:
            await self.connector.close()

    def _build_streaming_result(
        self,
        token_ids: List[int],
        finish_reason: Optional[str],
        return_logprobs: bool,
        return_entropy: bool,
        top_logprobs_k: int,
        last_output_token_logprobs: Optional[List],
        last_output_top_logprobs: Optional[List],
        last_output_token_entropy: Optional[List[float]],
        weight_version: Optional[str] = None,
    ) -> GenerationResult:
        logprobs: Optional[List[float]] = None
        if return_logprobs and last_output_token_logprobs:
            logprobs = [lp[0] for lp in last_output_token_logprobs]

        top_logprobs: Optional[List[Dict[int, float]]] = None
        if top_logprobs_k > 0 and last_output_top_logprobs:
            top_logprobs = []
            for pos_lps in last_output_top_logprobs:
                if pos_lps:
                    top_logprobs.append({item[1]: item[0] for item in pos_lps})
                else:
                    top_logprobs.append({})

        entropy: Optional[List[float]] = None
        if return_entropy and last_output_token_entropy:
            entropy = last_output_token_entropy

        return GenerationResult(
            token_ids=token_ids,
            finish_reason=finish_reason,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            entropy=entropy,
            routing_indices=None,
            cached_tokens=None,
            usage=None,
            weight_version=weight_version,
        )

    def _decode_routing_indices(
        self,
        routed_experts_b64: str,
    ) -> np.ndarray:
        """Decode base64 routing indices.

        Returns np.ndarray of shape [T, L, K] dtype=uint8.

        sglang returns routing for positions [0, N-1) where
        N = prompt + completion.  The last token has no routing.
        Prefix-cached tokens are included in the returned array.
        """
        if self.num_layers is None or self.num_experts_per_tok is None:
            raise RuntimeError(
                "Cannot decode routing_indices: num_hidden_layers and num_experts_per_tok "
                "must be provided to SGLangBackend when return_routing_indices=True."
            )

        routing_flat = np.frombuffer(
            base64.b64decode(routed_experts_b64.encode("utf-8")),
            dtype=np.int32,
        )

        per_token = self.num_layers * self.num_experts_per_tok
        total_tokens = routing_flat.size // per_token
        if routing_flat.size != total_tokens * per_token:
            raise RuntimeError(
                f"Routing indices size {routing_flat.size} not divisible by "
                f"layers*topk={per_token}"
            )

        routing = routing_flat.reshape(total_tokens, self.num_layers, self.num_experts_per_tok)
        return routing.astype(np.uint8)

    @staticmethod
    def _normalize_finish_reason(finish_reason: Any) -> Optional[str]:
        if finish_reason is None:
            return None
        if isinstance(finish_reason, dict):
            return finish_reason.get("type")
        return str(finish_reason)
