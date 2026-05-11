"""Checkpoint metadata utilities for save & resume."""

import json
import logging
import os
import re
import shutil
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

METADATA_FILENAME = "checkpoint_meta.json"


class CheckpointMetadata(BaseModel):
    step_count: int
    batch_idx: int
    factory_state: dict
    wandb_run_id: str | None = None
    timestamp: str
    training_mode: str | None = None  # None = legacy checkpoint (assumed LoRA)
    current_weight_version: int | None = None  # deprecated, kept for backward compat


def save_checkpoint_metadata(ckpt_dir: Path, meta: CheckpointMetadata) -> None:
    """Save checkpoint metadata atomically (write to .tmp then rename)."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    target = ckpt_dir / METADATA_FILENAME
    tmp = target.with_suffix(".tmp")
    tmp.write_text(meta.model_dump_json(indent=2))
    os.replace(tmp, target)


def load_checkpoint_metadata(ckpt_dir: Path) -> CheckpointMetadata:
    """Load checkpoint metadata from a checkpoint directory."""
    meta_path = Path(ckpt_dir) / METADATA_FILENAME
    if not meta_path.exists():
        raise FileNotFoundError(f"No {METADATA_FILENAME} in {ckpt_dir}")
    return CheckpointMetadata.model_validate_json(meta_path.read_text())


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Find the latest checkpoint directory with valid metadata.

    Prefers checkpoint_latest (always most recent) over checkpoint_step_*.
    Returns the path or None if no valid checkpoint exists.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    # checkpoint_latest is always the most recent if it exists
    latest_dir = checkpoint_dir / "checkpoint_latest"
    if latest_dir.is_dir() and (latest_dir / METADATA_FILENAME).exists():
        return latest_dir

    # Fall back to highest-numbered checkpoint_step_*
    pattern = re.compile(r"^checkpoint_step_(\d+)$")
    candidates: list[tuple[int, Path]] = []

    for entry in checkpoint_dir.iterdir():
        if not entry.is_dir():
            continue
        m = pattern.match(entry.name)
        if m and (entry / METADATA_FILENAME).exists():
            candidates.append((int(m.group(1)), entry))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def cleanup_old_checkpoints(checkpoint_dir: Path, keep: int) -> None:
    """Keep the latest `keep` checkpoints, remove the rest."""
    if keep <= 0:
        return

    checkpoint_dir = Path(checkpoint_dir)
    pattern = re.compile(r"^checkpoint_step_(\d+)$")
    candidates: list[tuple[int, Path]] = []

    for entry in checkpoint_dir.iterdir():
        if not entry.is_dir():
            continue
        m = pattern.match(entry.name)
        if m and (entry / METADATA_FILENAME).exists():
            candidates.append((int(m.group(1)), entry))

    candidates.sort(key=lambda x: x[0])

    to_remove = candidates[:-keep] if len(candidates) > keep else []
    for step, path in to_remove:
        logger.info(f"Removing old checkpoint: {path}")
        shutil.rmtree(path)
