"""GroupConfigFactory — turns a dataset into an infinite stream of GroupConfigs.

Each dataset item is a dict with at least a "prompt" field (or mapped via prompt_key).
Each item becomes one group: group_size rollouts with different seeds.

Dataset loading: accepts either a list[dict] directly or a file path (.json / .jsonl).
"""

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

from ..orchestrator.types import GroupConfig
from ..rollout.config import ConversationConfig, LoopConfig, RecordConfig
from ..rollout.parallel.config import RolloutConfig


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load dataset from .json or .jsonl file."""
    path = Path(path)
    assert path.exists(), f"Dataset file not found: {path}"

    if path.suffix == ".jsonl":
        data = []
        with open(path) as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        assert len(data) > 0, f"Empty dataset: {path}"
        return data
    elif path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list), f"Expected list in {path}, got {type(data)}"
        assert len(data) > 0, f"Empty dataset: {path}"
        return data
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix} (expected .json or .jsonl)")


class GroupConfigFactory:
    """Infinite iterator of GroupConfigs from a dataset.

    Shuffles each epoch with a deterministic seed.
    Wraps around to the next epoch when exhausted.
    """

    def __init__(
        self,
        *,
        dataset: list[dict[str, Any]] | str | Path,
        group_size: int,
        loop_config: LoopConfig,
        conv_config: ConversationConfig,
        record_config: RecordConfig,
        prompt_key: str = "prompt",
        mcp_config_key: str = "mcp_config",
        base_seed: int = 42,
        prompt_fn: Callable[[dict, str], tuple[str, dict]] | None = None,
        cache_salt_mode: Literal["none", "per_group", "per_rollout"] = "none",
        server_affinity: bool = False,
    ):
        if isinstance(dataset, (str, Path)):
            dataset = load_dataset(dataset)

        assert len(dataset) > 0, "dataset cannot be empty"
        assert group_size >= 2, f"group_size must be >= 2, got {group_size}"
        assert cache_salt_mode in ("none", "per_rollout", "per_group"), (
            f"cache_salt_mode must be 'none', 'per_rollout', or 'per_group', got {cache_salt_mode!r}"
        )

        self.dataset = dataset
        self.group_size = group_size
        self.loop_config = loop_config
        self.conv_config = conv_config
        self.record_config = record_config
        self.prompt_key = prompt_key
        self.mcp_config_key = mcp_config_key
        self.base_seed = base_seed
        self.prompt_fn = prompt_fn
        self.cache_salt_mode = cache_salt_mode
        self.server_affinity = server_affinity

        # Resumable state
        self.current_epoch = 0
        self.current_position = 0

    def __iter__(self) -> Iterator[GroupConfig]:
        while True:
            rng = random.Random(self.base_seed + self.current_epoch)
            indices = list(range(len(self.dataset)))
            rng.shuffle(indices)

            for idx in indices[self.current_position :]:
                item = self.dataset[idx]
                group_id = f"e{self.current_epoch}_i{idx}"
                yield self._make_group(group_id, item)
                self.current_position += 1

            self.current_epoch += 1
            self.current_position = 0

    def _make_group(
        self, group_id: str, item: dict[str, Any]
    ) -> GroupConfig:
        # Dynamic prompt via prompt_fn, or static prompt from item[prompt_key]
        if self.prompt_fn is not None:
            prompt, extra_meta = self.prompt_fn(item, group_id)
            # Strip _-prefixed keys (e.g. _summaries_correct) from metadata to
            # avoid serializing large payloads 40x per group through mp.Queue.
            # prompt_fn already consumed them; reward_fn doesn't need them.
            metadata = {k: v for k, v in item.items() if not k.startswith("_")}
            metadata.update(extra_meta)
        else:
            prompt = item[self.prompt_key]
            metadata = item

        mcp_config = item.get(self.mcp_config_key)
        loop_overrides = item.get("_loop_overrides") or {}
        configs: list[RolloutConfig] = []

        for j in range(self.group_size):
            rollout_id = f"{group_id}_r{j}"
            seed = _deterministic_seed(rollout_id)

            # Cache salt
            if self.cache_salt_mode == "per_group":
                salt = group_id
            elif self.cache_salt_mode == "per_rollout":
                salt = rollout_id
            else:
                salt = None

            configs.append(
                RolloutConfig(
                    rollout_id=rollout_id,
                    user_prompt=prompt,
                    loop_config=self.loop_config.model_copy(
                        update={
                            "sampling": self.loop_config.sampling.model_copy(
                                update={"seed": seed}
                            ),
                            "cache_salt": salt,
                            **loop_overrides,
                        }
                    ),
                    conv_config=self.conv_config,
                    record_config=self.record_config,
                    mcp_config=mcp_config,
                    metadata=metadata,
                )
            )

        return GroupConfig(
            group_id=group_id,
            rollout_configs=configs,
            metadata=metadata,
            server_affinity=self.server_affinity,
        )

    def state_dict(self) -> dict:
        return {
            "current_epoch": self.current_epoch,
            "current_position": self.current_position,
            "dataset_size": len(self.dataset),
        }

    def load_state_dict(self, state: dict) -> None:
        assert state["dataset_size"] == len(self.dataset), (
            f"Dataset size mismatch: checkpoint has {state['dataset_size']}, "
            f"current has {len(self.dataset)}"
        )
        self.current_epoch = state["current_epoch"]
        self.current_position = state["current_position"]


def _deterministic_seed(rollout_id: str) -> int:
    """Deterministic seed from rollout_id via md5."""
    h = hashlib.md5(rollout_id.encode()).hexdigest()
    return int(h[:8], 16) % (2**31 - 1)
