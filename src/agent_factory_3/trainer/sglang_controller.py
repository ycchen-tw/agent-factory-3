"""SglangController — async HTTP controller for sglang weight sync.

Supports two sync modes:
  lora:   pause(abort) → unload_lora → load_lora → update_version → flush_cache → continue
  merged: pause → update_weights_from_disk(flush_cache, weight_version) → continue
"""

import asyncio
import logging

import aiohttp

logger = logging.getLogger(__name__)


class SglangController:
    """Controls sglang inference servers for RL weight sync."""

    def __init__(
        self,
        server_urls: list[str],
        *,
        mode: str = "lora",
        lora_name: str = "policy",
        flush_cache: bool = True,
    ):
        if mode not in ("lora", "merged"):
            raise ValueError(f"Unknown sync mode: {mode!r}, expected 'lora' or 'merged'")
        self.server_urls = server_urls
        self.mode = mode
        self.lora_name = lora_name
        self.flush_cache = flush_cache  # merged mode only; lora mode always flushes

    async def sync_weights(self, weight_path: str, weight_version: str, *, flush: bool = False) -> None:
        """Execute weight sync protocol on all servers based on configured mode.

        Args:
            flush: Per-call flush override. When True, forces retract+flush even if
                self.flush_cache is False. Ignored for lora mode (always abort+flush).
        """
        if self.mode == "lora":
            await self._sync_lora(weight_path, weight_version)
        else:
            await self._sync_merged(weight_path, weight_version, flush_override=flush)

    async def _sync_lora(self, lora_path: str, weight_version: str) -> None:
        """LoRA adapter sync: swap adapter on all servers.

        Each sglang API returns 200 when the command is accepted, but the actual
        processing (abort in-flight requests, release KV blocks, update model weights)
        happens asynchronously in the scheduler event loop. We sleep between steps
        to let the scheduler fully process each operation before proceeding.
        """
        await self._call_all("pause_generation", json={"mode": "abort"})
        await asyncio.sleep(5.0)  # let scheduler abort all in-flight requests & release KV
        await self._call_all("unload_lora_adapter", json={"lora_name": self.lora_name}, ignore_missing=True)
        await asyncio.sleep(3.0)  # let scheduler unload adapter weights
        await self._call_all("load_lora_adapter", json={"lora_name": self.lora_name, "lora_path": lora_path})
        await asyncio.sleep(2.0)  # let scheduler load new adapter weights
        await self._call_all("update_weight_version", json={"new_version": weight_version})
        await asyncio.sleep(1.0)  # let scheduler update version metadata
        await self._call_all("flush_cache")
        await asyncio.sleep(2.0)  # let scheduler flush KV cache
        await self._call_all("continue_generation", json={})
        logger.info(f"LoRA sync complete: {lora_path} → version {weight_version}")

    async def _sync_merged(self, model_path: str, weight_version: str, flush_override: bool = False) -> None:
        """Merged weight sync: update base model weights from disk on all servers.

        Uses sglang's update_weights_from_disk endpoint which atomically loads
        weight tensors from a safetensors checkpoint and updates matching parameters.

        Two modes:
        - do_flush=True (self.flush_cache or flush_override): abort + flush
        - do_flush=False: in_place, keep KV cache
        """
        do_flush = self.flush_cache or flush_override

        if do_flush:
            await self._call_all("pause_generation", json={"mode": "abort"})
            await asyncio.sleep(5.0)
        else:
            await self._call_all("pause_generation", json={"mode": "in_place"})
            await asyncio.sleep(2.0)

        await self._call_all("update_weights_from_disk", json={
            "model_path": model_path,
            "flush_cache": do_flush,
            "weight_version": weight_version,
        })
        await asyncio.sleep(3.0 if do_flush else 1.0)
        await self._call_all("continue_generation", json={})

        pause_mode = "abort" if do_flush else "in_place"
        logger.info(
            f"Merged sync complete: {model_path} → version {weight_version}"
            f" (pause={pause_mode}, flush={do_flush})"
        )

    async def _call_all(self, endpoint: str, json: dict | None = None, ignore_missing: bool = False) -> None:
        """Call endpoint on all servers in parallel."""
        async with aiohttp.ClientSession() as session:
            tasks = [self._call_one(session, url, endpoint, json, ignore_missing) for url in self.server_urls]
            await asyncio.gather(*tasks)

    async def _call_one(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        endpoint: str,
        json: dict | None,
        ignore_missing: bool = False,
    ) -> None:
        url = f"{base_url}/{endpoint}"
        async with session.post(url, json=json or {}) as resp:
            if resp.status != 200:
                text = await resp.text()
                if ignore_missing and "does not exist" in text:
                    logger.info(f"sglang {endpoint} on {base_url}: adapter not loaded yet, skipping unload")
                    return
                raise RuntimeError(f"sglang {endpoint} failed on {base_url}: {resp.status} {text}")
