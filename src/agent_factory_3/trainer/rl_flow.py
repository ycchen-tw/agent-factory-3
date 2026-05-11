"""RLFlow — main RL training pipeline with DDIS training and weight sync.

Architecture:
    Orchestrator Process (separate)     Training Process (this)
    ┌──────────────────────┐            ┌──────────────────────┐
    │ GroupConfigFactory   │            │ Rank 0:              │
    │ Orchestrator.run()   │            │   sample_queue.get() │
    │ SampleProcessor      │──queue──→  │   staleness filter   │
    │ SglangController     │            │   train_step()       │
    │ ← control signals    │←─queue──   │   weight sync        │
    └──────────────────────┘            │ Rank 1..N:           │
                                        │   rpc_worker_loop()  │
                                        └──────────────────────┘
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import queue
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

from ..orchestrator import GroupResult, Orchestrator
from ..orchestrator.types import RewardFn
from ..rollout.config import ConversationConfig, LoopConfig, RecordConfig
from ..rollout.parallel.config import RolloutResult
from .group_config_factory import GroupConfigFactory
from .model_trainer import ModelTrainer
from .rollout_saver import RolloutSaver
from .sample_processor import SampleProcessor
from .sglang_controller import SglangController
from .sglang_metrics import SglangMetricsDrainer
from .checkpoint_utils import (
    CheckpointMetadata,
    cleanup_old_checkpoints,
    find_latest_checkpoint,
    load_checkpoint_metadata,
    save_checkpoint_metadata,
)
from .training_config import TrainingConfig
from .types import ProcessStats, TrainingSample

logger = logging.getLogger(__name__)


# ---- Control signals (Rank 0 → Orchestrator) ----

@dataclass
class WeightsUpdated:
    path: str
    step: int
    flush: bool = False

@dataclass
class Shutdown:
    pass


# ---- Divisor assignment (same as v2) ----

_AGGREGATION_MODES = {
    "token-mean",
    "seq-mean-token-sum",
    "seq-mean-token-mean",
    "prompt-mean-token-mean",
}


def assign_divisors(samples: list[TrainingSample], mode: str) -> None:
    """Assign per-sample divisor based on aggregation mode.

    This determines how loss is normalized — identical to v2 logic.
    """
    if mode not in _AGGREGATION_MODES:
        raise ValueError(f"Unknown aggregation mode: {mode}")

    for s in samples:
        if mode == "token-mean":
            s.divisor = float(s.seq_token_count)
        elif mode == "seq-mean-token-sum":
            s.divisor = float(s.prompt_sequence_count)
        elif mode == "seq-mean-token-mean":
            s.divisor = float(s.prompt_sequence_count * s.seq_token_count)
        elif mode == "prompt-mean-token-mean":
            s.divisor = float(s.prompt_token_count)


# ---- Staleness filter ----

def filter_stale_samples(
    samples: list[TrainingSample],
    current_step: int,
    max_staleness: int,
) -> list[TrainingSample]:
    """Drop samples generated with weights too far behind current training step."""
    if max_staleness < 0:
        return samples

    threshold = current_step - max_staleness
    kept = [s for s in samples if s.oldest_weight_version >= threshold]
    dropped = len(samples) - len(kept)
    if dropped > 0:
        logger.info(
            f"Staleness filter: dropped {dropped}/{len(samples)} samples "
            f"(threshold={threshold}, current_step={current_step})"
        )
    return kept


def compute_staleness_metrics(
    samples: list[TrainingSample],
    current_step: int,
    pre_filter_count: int,
) -> tuple[dict[str, float], dict[int, float]]:
    """Compute staleness distribution metrics from kept samples.

    Returns:
        (metrics_dict, version_pct_dict)
        - metrics_dict: flat metrics for wandb scalar logging
        - version_pct_dict: {weight_version: token_pct} for table logging
    """
    drop_count = pre_filter_count - len(samples)
    metrics: dict[str, float] = {
        "flow/stale_drop_count": float(drop_count),
        "flow/stale_drop_rate": drop_count / max(pre_filter_count, 1),
    }
    version_pct: dict[int, float] = {}

    if not samples:
        metrics["flow/staleness_mean"] = 0.0
        metrics["flow/staleness_max"] = 0.0
        return metrics, version_pct

    # Aggregate token counts by weight version across all samples
    total_by_version: dict[int, int] = {}
    for s in samples:
        for v, count in s.weight_version_token_counts.items():
            total_by_version[v] = total_by_version.get(v, 0) + count

    total_tokens = sum(total_by_version.values())
    if total_tokens == 0:
        metrics["flow/staleness_mean"] = 0.0
        metrics["flow/staleness_max"] = 0.0
        return metrics, version_pct

    # Convert to staleness = current_step - version
    weighted_sum = 0.0
    max_staleness = 0
    for version, count in total_by_version.items():
        staleness = current_step - version
        pct = count / total_tokens
        metrics[f"flow/token_pct/staleness_{staleness}"] = pct
        version_pct[version] = pct
        weighted_sum += staleness * count
        max_staleness = max(max_staleness, staleness)

    metrics["flow/staleness_mean"] = weighted_sum / total_tokens
    metrics["flow/staleness_max"] = float(max_staleness)

    return metrics, version_pct


# ---- Orchestrator process entry point ----


def _orchestrator_main(
    sample_queue: mp.Queue,
    control_queue: mp.Queue,
    metrics_queue: mp.Queue | None,
    *,
    # Orchestrator config
    server_urls: list[str],
    num_workers: int,
    worker_concurrency: int,
    batch_size: int,
    reward_fn: RewardFn,
    # Data config
    dataset: list[dict[str, Any]] | str | Path,
    group_size: int,
    loop_config: LoopConfig,
    conv_config: ConversationConfig,
    record_config: RecordConfig,
    prompt_key: str,
    base_seed: int,
    # Sample processing
    normalize_advantages: bool,
    filter_all_failed: bool,
    filter_all_solved: bool,
    excluded_error_kinds: list[str] | None,
    # Rollout saving
    rollout_save_dir: str | None,
    save_routing_indices: bool = True,
    # Weight sync
    weight_sync_mode: str,
    lora_name: str,
    flush_cache_on_sync: bool,
    # Resume
    factory_initial_state: dict | None = None,
    # Logging
    rollout_log_path: str | None = None,
    # Dynamic prompt / prefix cache sharing
    prompt_fn: Any = None,
    cache_salt_mode: Literal["none", "per_group", "per_rollout"] = "none",
    server_affinity: bool = False,
) -> None:
    """Orchestrator process entry point. Runs in a separate process."""
    # Replace inherited handlers from parent process so that:
    #   1. [ORCH] prefix actually works (basicConfig was a no-op before)
    #   2. Rollout logs go to a dedicated file
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [ORCH] %(name)s %(levelname)s: %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if rollout_log_path:
        fh = logging.FileHandler(rollout_log_path)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    logger.info("Orchestrator process started")

    factory = GroupConfigFactory(
        dataset=dataset,
        group_size=group_size,
        loop_config=loop_config,
        conv_config=conv_config,
        record_config=record_config,
        prompt_key=prompt_key,
        base_seed=base_seed,
        prompt_fn=prompt_fn,
        cache_salt_mode=cache_salt_mode,
        server_affinity=server_affinity,
    )
    if factory_initial_state is not None:
        factory.load_state_dict(factory_initial_state)
        logger.info(
            f"Factory state restored: epoch={factory.current_epoch}, "
            f"pos={factory.current_position}"
        )

    processor = SampleProcessor(
        normalize_advantages=normalize_advantages,
        filter_all_failed=filter_all_failed,
        filter_all_solved=filter_all_solved,
        excluded_error_kinds=excluded_error_kinds,
    )

    orchestrator = Orchestrator(
        server_urls=server_urls,
        num_workers=num_workers,
        worker_concurrency=worker_concurrency,
        batch_size=batch_size,
        reward_fn=reward_fn,
        metrics_queue=metrics_queue,
    )

    saver = RolloutSaver(rollout_save_dir, save_routing_indices=save_routing_indices) if rollout_save_dir else None
    controller = SglangController(
        server_urls,
        mode=weight_sync_mode,
        lora_name=lora_name,
        flush_cache=flush_cache_on_sync,
    )

    try:
        asyncio.run(_orchestrator_loop(
            orchestrator=orchestrator,
            groups=factory,
            processor=processor,
            saver=saver,
            controller=controller,
            sample_queue=sample_queue,
            control_queue=control_queue,
        ))
    finally:
        # Flush the metrics_queue feeder thread before the process dies.
        # Without close()+join_thread() the producer-side feeder buffer
        # may still hold un-pushed items when the process exits, and the
        # consumer (rank 0) can never see them. Skip when wandb is
        # disabled — metrics_queue is None in that case.
        if metrics_queue is not None:
            try:
                metrics_queue.close()
                metrics_queue.join_thread()
            except Exception:
                logger.exception("metrics_queue feeder flush failed")

    logger.info("Orchestrator process exiting")


async def _orchestrator_loop(
    *,
    orchestrator: Orchestrator,
    groups: GroupConfigFactory,
    processor: SampleProcessor,
    saver: RolloutSaver | None,
    controller: SglangController,
    sample_queue: mp.Queue,
    control_queue: mp.Queue,
) -> None:
    """Main orchestrator async loop."""
    stop_event = asyncio.Event()
    batch_count = 0

    # Independent task to listen for control signals (weight sync, shutdown).
    # This ensures signals are processed immediately, not just at batch boundaries.
    listener = asyncio.create_task(
        _control_listener(control_queue, stop_event, controller)
    )

    try:
        async for group_results in orchestrator.run(groups, stop_event=stop_event):
            if stop_event.is_set():
                logger.info("Stop event set, draining remaining results")
                break

            batch_count += 1

            # Process into training samples (annotates group_results in-place)
            t0 = time.time()
            samples, stats = processor.process(group_results)
            process_time = time.time() - t0

            # Save rollouts to disk (after processing — has filter/advantage annotations)
            html_path: str | None = None
            if saver is not None:
                try:
                    _, hp = saver.save_batch(group_results, batch_count)
                    html_path = str(hp) if hp is not None else None
                    saver.save_training_data(group_results, batch_count)
                except Exception:
                    logger.exception(f"Failed to save batch {batch_count}")

            total_rollouts = sum(len(gr.results) for gr in group_results)
            logger.info(
                f"Batch {batch_count}: {len(group_results)} groups, "
                f"{total_rollouts} rollouts → {stats.total_samples} samples "
                f"(filtered: failed={stats.filtered_all_failed}, "
                f"solved={stats.filtered_all_solved}, "
                f"zero_loss={stats.filtered_zero_loss}) "
                f"[process={process_time:.2f}s]"
            )

            if not samples:
                logger.warning(f"Batch {batch_count}: all groups filtered, skipping")
                continue

            # Put into queue (blocks if full = back-pressure).
            factory_state = groups.state_dict()
            while not stop_event.is_set():
                try:
                    sample_queue.put((samples, stats, html_path, factory_state), timeout=5)
                    break
                except queue.Full:
                    continue
    finally:
        listener.cancel()
        try:
            await listener
        except asyncio.CancelledError:
            pass


async def _control_listener(
    control_queue: mp.Queue,
    stop_event: asyncio.Event,
    controller: SglangController,
) -> None:
    """Continuously listen for control signals and process them immediately.

    Runs as an independent asyncio task so weight sync is handled promptly,
    rather than waiting for a batch boundary.
    """
    loop = asyncio.get_event_loop()
    while not stop_event.is_set():
        try:
            signal = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: control_queue.get(timeout=1)),
                timeout=2,
            )
        except (asyncio.TimeoutError, queue.Empty):
            continue

        if isinstance(signal, Shutdown):
            logger.info("Received shutdown signal")
            stop_event.set()
        elif isinstance(signal, WeightsUpdated):
            flush_tag = " [flush]" if signal.flush else ""
            logger.info(f"Syncing weights: {signal.path} (step {signal.step}){flush_tag}")
            await controller.sync_weights(signal.path, str(signal.step), flush=signal.flush)
        else:
            logger.warning(f"Unknown control signal: {signal}")


def _drain_queue(q: mp.Queue) -> int:
    """Drain all items from a queue. Returns count of items drained."""
    count = 0
    while True:
        try:
            q.get_nowait()
            count += 1
        except queue.Empty:
            break
    return count


# ---- Wandb helpers ----


def _init_wandb(
    config: TrainingConfig,
    resume_run_id: str | None = None,
) -> tuple[bool, str | None]:
    """Initialize wandb. Returns (success, run_id)."""
    try:
        import wandb

        run_name = config.wandb_run_name
        if run_name is None:
            run_name = f"rl_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        wandb_dir = str(Path(config.run_dir).resolve())
        Path(wandb_dir).mkdir(parents=True, exist_ok=True)

        init_kwargs: dict[str, Any] = dict(
            project=config.wandb_project,
            name=run_name,
            tags=config.wandb_tags,
            notes=config.wandb_notes,
            config=config.model_dump(),
            dir=wandb_dir,
        )
        if resume_run_id:
            init_kwargs["id"] = resume_run_id
            init_kwargs["resume"] = "must"

        wandb.init(**init_kwargs)

        # Canonical wandb pattern for two independent metric streams in a
        # single run (training + sglang per-server). NO `step=` is passed
        # to wandb.log anywhere — wandb auto-increments its global _step,
        # which is always monotonic by construction so nothing is dropped.
        # The panel x axis comes from each metric's `step_metric`:
        #   - training metrics  → flow/step_count   (carried in payload)
        #   - sglang/* metrics  → sglang/_seq       (carried in payload)
        # Order: declare the step axes themselves first, then the wildcard
        # default, then the more-specific sglang override.
        wandb.define_metric("flow/step_count")
        wandb.define_metric("sglang/_seq")
        wandb.define_metric("*", step_metric="flow/step_count")
        wandb.define_metric("sglang/*", step_metric="sglang/_seq")

        if not resume_run_id:
            # Log source code as artifact for reproducibility (new runs only)
            wandb.run.log_code(
                root=str(Path(__file__).resolve().parent.parent.parent.parent),
                include_fn=lambda path, root: path.endswith(
                    (".py", ".sh", ".yaml", ".toml", ".md")
                ),
            )
        logger.info(f"Wandb initialized: {config.wandb_project}/{run_name}")
        return True, wandb.run.id
    except Exception:
        logger.exception("Failed to initialize wandb")
        return False, None


def _wandb_log_step(
    step_metrics: dict[str, float],
    stats: ProcessStats,
    step: int,
    html_path: str | None = None,
    staleness_table_rows: list[list] | None = None,
) -> None:
    """Log training step metrics + process stats + optional rollout HTML to wandb.

    `step` is carried as `flow/step_count` inside the payload — that key
    is registered as the x-axis step_metric for training panels in
    `_init_wandb`. We do NOT pass `step=` to `wandb.log`; see the
    canonical-pattern note in `_init_wandb`.
    """
    try:
        import wandb

        payload: dict = {**step_metrics, **stats.to_metrics_dict()}
        # Ensure flow/step_count is present even if caller forgot — this is
        # the x-axis for every training panel.
        payload.setdefault("flow/step_count", float(step))
        if html_path is not None:
            with open(html_path, encoding="utf-8") as f:
                payload["Rollout Visualizer"] = wandb.Html(f.read(), inject=False)
        if staleness_table_rows:
            table = wandb.Table(
                columns=["step", "weight_version", "staleness", "token_pct"],
                data=staleness_table_rows,
            )
            payload["Token Source Distribution"] = table
        wandb.log(payload, commit=True)  # no step= — wandb auto-advances
    except Exception:
        logger.exception("Failed to log to wandb")


# ---- RLFlow ----


class RLFlow:
    """Main RL training pipeline with DDIS training and async weight sync.

    Training loop (rank 0):
        1. Get samples from orchestrator queue
        2. Staleness filter
        3. Assign divisors
        4. train_step()
        5. Every K steps: save weights → signal orchestrator → weight sync
    """

    def __init__(
        self,
        *,
        # Training
        training_config: TrainingConfig,
        model_trainer: ModelTrainer,
        # Dataset
        dataset: list[dict[str, Any]] | str | Path,
        reward_fn: RewardFn,
        prompt_key: str = "prompt",
        # Orchestrator
        server_urls: list[str],
        num_workers: int = 8,
        worker_concurrency: int = 8,
        batch_size: int = 128,
        group_size: int = 8,
        # Rollout
        loop_config: LoopConfig,
        conv_config: ConversationConfig = ConversationConfig(),
        record_config: RecordConfig = RecordConfig(),
        # Sample processing
        normalize_advantages: bool = False,
        filter_all_failed: bool = True,
        filter_all_solved: bool = True,
        excluded_error_kinds: list[str] | None = None,
        # Flow
        max_batches: int | None = None,
        base_seed: int = 42,
        # Dynamic prompt / prefix cache sharing
        prompt_fn: Any = None,
        cache_salt_mode: Literal["none", "per_group", "per_rollout"] = "none",
        server_affinity: bool = False,
    ):
        self.training_config = training_config
        self.model_trainer = model_trainer
        self.dataset = dataset
        self.reward_fn = reward_fn
        self.prompt_key = prompt_key
        self.server_urls = server_urls
        self.num_workers = num_workers
        self.worker_concurrency = worker_concurrency
        self.batch_size = batch_size
        self.group_size = group_size
        self.loop_config = loop_config
        self.conv_config = conv_config
        self.record_config = record_config
        self.normalize_advantages = normalize_advantages
        self.filter_all_failed = filter_all_failed
        self.filter_all_solved = filter_all_solved
        self.excluded_error_kinds = excluded_error_kinds
        self.max_batches = max_batches
        self.base_seed = base_seed
        self.prompt_fn = prompt_fn
        self.cache_salt_mode = cache_salt_mode
        self.server_affinity = server_affinity

        # Cross-validate: full-param mode must not send lora_adapter_name to sglang
        if training_config.training_mode == "full" and loop_config.lora_adapter_name is not None:
            raise ValueError(
                f"training_mode='full' but loop_config.lora_adapter_name="
                f"{loop_config.lora_adapter_name!r}. "
                "Set lora_adapter_name=None for full-param training."
            )

        # Cross-validate: routing replay needs MoE shape on the rollout side
        # (otherwise SGLangBackend cannot reshape the [T, L, K] base64 payload).
        needs_routing_shape = (
            training_config.use_routing_replay or record_config.routing_indices
        )
        if needs_routing_shape and (
            loop_config.num_hidden_layers is None
            or loop_config.num_experts_per_tok is None
        ):
            raise ValueError(
                "use_routing_replay=True (or record_config.routing_indices=True) "
                "requires loop_config.num_hidden_layers and num_experts_per_tok to "
                "be set so SGLangBackend can reshape captured routing indices. "
                "For gpt-oss-120b: num_hidden_layers=36, num_experts_per_tok=4."
            )

        # Cross-validate: streaming mode does not return routed_experts
        if loop_config.use_streaming and record_config.routing_indices:
            raise ValueError(
                "loop_config.use_streaming=True is incompatible with "
                "record_config.routing_indices=True: sglang's streaming endpoint "
                "does not return routed_experts. "
                "Set use_streaming=False to capture routing indices."
            )

        # Cross-validate: segment_temperature uses two-phase generation, which
        # cannot return per-token logprobs or routing indices.
        if loop_config.segment_temperature is not None and (
            record_config.logprobs or record_config.routing_indices
        ):
            raise ValueError(
                "loop_config.segment_temperature is set, which uses two-phase "
                "generation that does not return logprobs or routing_indices. "
                "Either disable segment_temperature, or set "
                "record_config.logprobs=False and record_config.routing_indices=False."
            )

        # Cross-validate: batch_size must be a positive multiple of group_size
        # (DDIS computes group-relative advantage so partial groups are not allowed).
        if batch_size <= 0 or batch_size % group_size != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be a positive multiple of "
                f"group_size ({group_size})."
            )

    def run(
        self,
        on_step: Callable[[dict[str, float], int], None] | None = None,
    ) -> list[dict[str, float]]:
        """Run the full training pipeline.

        Non-rank-0 processes enter RPC worker loop.
        Rank 0 runs the training loop consuming from the orchestrator queue.

        Args:
            on_step: Optional callback after each train step.
                Signature: (step_metrics, step_index) -> None

        Returns:
            List of per-step metrics dicts.
        """
        # Non-rank-0: enter worker loop
        if not self.model_trainer.is_main:
            self.model_trainer.rpc_worker_loop()
            return []

        # Rank 0: start orchestrator and run training loop
        sample_queue: mp.Queue = mp.Queue(maxsize=2)
        control_queue: mp.Queue = mp.Queue()
        # Sglang per-server metrics → wandb. Allocation rules:
        #  - `use_wandb=False`: don't allocate at all → orchestrator's poller
        #    sees None and skips enqueue (no leak).
        #  - `use_wandb=True` but wandb init then fails: queue exists with
        #    no consumer; the maxsize cap bounds the leak — orchestrator's
        #    `put_nowait` will start raising Full once the cap hits, which
        #    its existing `except queue.Full: pass` already absorbs.
        # Cap sized for ~7h buffer at typical 4-server / 10s poll rate
        # (4 items / 10s = 1440/hr; 10000 / 1440 ≈ 7h).
        metrics_queue: mp.Queue | None = (
            mp.Queue(maxsize=10000) if self.training_config.use_wandb else None
        )

        # Create all output directories under run_dir
        for d in [
            self.training_config.checkpoint_dir,
            self.training_config.rollout_dir,
            self.training_config.log_dir,
            self.training_config.weight_sync_dir,
            self.training_config.weight_init_dir,
        ]:
            Path(d).mkdir(parents=True, exist_ok=True)

        # Set up per-process log files (training ↔ rollout separation)
        log_dir = Path(self.training_config.log_dir)
        log_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_log_path = log_dir / f"train_{log_ts}.log"
        rollout_log_path = log_dir / f"rollout_{log_ts}.log"

        train_fh = logging.FileHandler(train_log_path)
        train_fh.setFormatter(logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s: %(message)s",
        ))
        logging.getLogger().addHandler(train_fh)
        logger.info(f"Train log: {train_log_path}")
        logger.info(f"Rollout log: {rollout_log_path}")

        checkpoint_dir = Path(self.training_config.checkpoint_dir).resolve()
        sync_mode = self.training_config.weight_sync_mode

        # --- Resume from checkpoint ---
        resume_path: Path | None = None
        factory_initial_state: dict | None = None
        wandb_run_id: str | None = None

        resume_cfg = self.training_config.resume_from_checkpoint
        if resume_cfg:
            if resume_cfg == "latest":
                resume_path = find_latest_checkpoint(checkpoint_dir)
                if resume_path is None:
                    logger.warning(
                        "resume_from_checkpoint='latest' but no checkpoint found, "
                        "starting fresh"
                    )
            else:
                resume_path = Path(resume_cfg)
                if not resume_path.exists():
                    raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

        if resume_path is not None:
            logger.info(f"Resuming from checkpoint: {resume_path}")
            meta = load_checkpoint_metadata(resume_path)

            self.model_trainer.load_model(str(resume_path))

            ts_path = resume_path / "training_state.pt"
            if ts_path.exists():
                self.model_trainer.load_training_state(ts_path)

            step_count = meta.step_count
            batch_idx = meta.batch_idx
            factory_initial_state = meta.factory_state
            wandb_run_id = meta.wandb_run_id if self.training_config.wandb_resume else None
        else:
            step_count = 0
            batch_idx = 0

        # Initial weight sync: save weights and load into sglang before any rollout starts.
        # Always flush cache on init — stale KV cache from previous weights is invalid.
        if self.server_urls:
            controller = SglangController(
                self.server_urls,
                mode=sync_mode,
                lora_name=self.training_config.lora_adapter_name,
                flush_cache=True,
            )
            init_path = self.training_config.weight_init_dir
            if sync_mode == "lora":
                # LoRA mode: save adapter files; sglang loads via load_lora_adapter
                if resume_path is None:
                    self.model_trainer.save_model(init_path)
                else:
                    init_path = str(resume_path)
            else:
                # Merged mode (LoRA or full-param): always export merged weights.
                # On resume, the checkpoint may only have adapter/trainable files,
                # so we re-export to produce model.safetensors for sglang.
                self.model_trainer.save_merged_weights(init_path)
            asyncio.run(controller.sync_weights(
                init_path, str(batch_idx),
            ))
            logger.info(f"Initial {sync_mode} weights synced to sglang: {init_path}")

        last_sync_time = time.time()  # track for min_weight_sync_secs throttle

        orch_proc = mp.Process(
            target=_orchestrator_main,
            args=(sample_queue, control_queue, metrics_queue),
            kwargs=dict(
                server_urls=self.server_urls,
                num_workers=self.num_workers,
                worker_concurrency=self.worker_concurrency,
                batch_size=self.batch_size,
                reward_fn=self.reward_fn,
                dataset=self.dataset,
                group_size=self.group_size,
                loop_config=self.loop_config,
                conv_config=self.conv_config,
                record_config=self.record_config,
                prompt_key=self.prompt_key,
                base_seed=self.base_seed,
                normalize_advantages=self.normalize_advantages,
                filter_all_failed=self.filter_all_failed,
                filter_all_solved=self.filter_all_solved,
                excluded_error_kinds=self.excluded_error_kinds,
                rollout_save_dir=self.training_config.rollout_dir,
                save_routing_indices=self.training_config.save_routing_indices_resolved,
                weight_sync_mode=self.training_config.weight_sync_mode,
                lora_name=self.training_config.lora_adapter_name,
                flush_cache_on_sync=self.training_config.flush_cache_on_sync,
                factory_initial_state=factory_initial_state,
                rollout_log_path=str(rollout_log_path),
                prompt_fn=self.prompt_fn,
                cache_salt_mode=self.cache_salt_mode,
                server_affinity=self.server_affinity,
            ),
        )
        orch_proc.start()
        logger.info(f"Orchestrator process started (pid={orch_proc.pid})")

        # Initialize wandb
        wandb_active = False
        if self.training_config.use_wandb:
            wandb_active, new_run_id = _init_wandb(
                self.training_config, wandb_run_id,
            )
            wandb_run_id = wandb_run_id or new_run_id

        # Background drainer for sglang per-server metrics. It pulls from
        # metrics_queue at its own cadence (independent of batch ticks) so
        # the queue never falls behind under multi-server / long-batch loads.
        # `metrics_queue is not None` is guaranteed by `use_wandb=True` (see
        # allocation above) — guarding both makes the invariant explicit.
        sglang_drainer: SglangMetricsDrainer | None = None
        if wandb_active and metrics_queue is not None:
            sglang_drainer = SglangMetricsDrainer(metrics_queue)
            sglang_drainer.start()

        all_step_metrics: list[dict[str, float]] = []
        latest_factory_state: dict = {}
        staleness_table_rows: list[list] = []  # accumulated rows for wandb Table heatmap
        _STALENESS_TABLE_MAX_STEPS = 200
        t0 = time.time()

        try:
            while True:
                if self.max_batches is not None and batch_idx >= self.max_batches:
                    logger.info(f"Reached max_batches={self.max_batches}")
                    break

                # Get samples from orchestrator
                if not orch_proc.is_alive():
                    try:
                        result = sample_queue.get_nowait()
                    except queue.Empty:
                        logger.error("Orchestrator process died")
                        break
                else:
                    try:
                        result = sample_queue.get(timeout=30)
                    except queue.Empty:
                        if not orch_proc.is_alive():
                            logger.error("Orchestrator process died")
                            break
                        logger.warning("Waiting for orchestrator...")
                        continue

                samples, stats, html_path, latest_factory_state = result
                batch_idx += 1
                elapsed = time.time() - t0

                # Note: sglang metrics are streamed by SglangMetricsDrainer in
                # its own thread, not here — see drainer.start() above.

                # Staleness filter — uses batch_idx as reference so max_staleness
                # naturally means "N batches behind" regardless of num_inner_steps.
                num_inner = self.training_config.num_inner_steps
                pre_filter_count = len(samples)
                samples = filter_stale_samples(
                    samples,
                    batch_idx,
                    self.training_config.max_staleness,
                )
                if not samples:
                    logger.warning(f"Batch {batch_idx}: all samples stale, skipping")
                    continue

                # Compute staleness distribution (uses batch_idx so values are in batch units)
                staleness_metrics, version_pct = compute_staleness_metrics(
                    samples, batch_idx, pre_filter_count,
                )

                # ---- Split into K mini-batches ----
                # Shuffle before splitting so each mini-batch is a random subset.
                # When num_inner_steps=1, no shuffle/split — same as single train_step.
                if num_inner > 1:
                    random.shuffle(samples)
                n_samples = len(samples)
                chunks = [
                    samples[i * n_samples // num_inner : (i + 1) * n_samples // num_inner]
                    for i in range(num_inner)
                ]
                chunks = [c for c in chunks if c]  # drop empty chunks

                logger.info(
                    f"Training batch {batch_idx}: {n_samples} samples → "
                    f"{len(chunks)} mini-batches of ~{n_samples // max(num_inner, 1)}, "
                    f"{sum(s.num_loss_tokens() for s in samples)} loss tokens "
                    f"[{elapsed:.1f}s elapsed]"
                )

                for inner_idx, chunk in enumerate(chunks):
                    # Assign divisors per mini-batch — each chunk normalizes
                    # independently by its own num_groups.
                    assign_divisors(chunk, self.training_config.loss_aggregation_mode)
                    num_groups = len({s.prompt_id for s in chunk})
                    if num_groups > 1:
                        for s in chunk:
                            s.divisor *= num_groups

                    step_metrics = self.model_trainer.train_step(chunk)
                    step_count += 1
                    elapsed = time.time() - t0

                    # Flow-level metrics
                    step_metrics["flow/batch_idx"] = float(batch_idx)
                    step_metrics["flow/step_count"] = float(step_count)
                    step_metrics["flow/inner_step"] = float(inner_idx)
                    step_metrics["flow/num_inner_steps"] = float(len(chunks))
                    step_metrics["flow/num_samples"] = float(len(chunk))
                    step_metrics["flow/elapsed"] = elapsed

                    # Staleness metrics — same for all inner steps, only emit on first
                    if inner_idx == 0:
                        step_metrics.update(staleness_metrics)

                    all_step_metrics.append(step_metrics)

                    if on_step is not None:
                        on_step(step_metrics, step_count)

                    # Staleness table rows — only on first inner step
                    if inner_idx == 0:
                        for version, pct in version_pct.items():
                            staleness_table_rows.append([
                                step_count, version, batch_idx - version, pct,
                            ])
                        if len(staleness_table_rows) > _STALENESS_TABLE_MAX_STEPS * 10:
                            min_step = step_count - _STALENESS_TABLE_MAX_STEPS
                            staleness_table_rows = [
                                r for r in staleness_table_rows if r[0] > min_step
                            ]

                    # Wandb — every inner step; html + staleness table only on first
                    if wandb_active:
                        _wandb_log_step(
                            step_metrics, stats, step_count,
                            html_path if inner_idx == 0 else None,
                            staleness_table_rows if inner_idx == 0 else None,
                        )

                # ---- Post-batch: weight sync + checkpoint (once per batch) ----

                # Weight sync — interval based on batch_idx (not step_count)
                sync_batch_ok = batch_idx % self.training_config.weight_sync_interval == 0
                sync_time_ok = (time.time() - last_sync_time) >= self.training_config.min_weight_sync_secs
                if sync_batch_ok and sync_time_ok:
                    # Double-buffer: alternate directories to avoid overwriting
                    # a file that sglang may still be reading on NFS
                    buf = "a" if batch_idx % 2 == 0 else "b"
                    save_path = Path(self.training_config.weight_sync_dir + f"_{buf}")
                    if sync_mode == "lora":
                        self.model_trainer.save_model(str(save_path))
                    else:
                        self.model_trainer.save_merged_weights(str(save_path))

                    if self.training_config.reset_optimizer_on_sync:
                        self.model_trainer.reset_optimizer_state()

                    n = self.training_config.flush_cache_every_n_steps
                    do_flush = n > 0 and batch_idx % n == 0

                    control_queue.put(WeightsUpdated(
                        path=str(save_path),
                        step=batch_idx,
                        flush=do_flush,
                    ))
                    last_sync_time = time.time()
                    logger.info(
                        f"Weight sync triggered: step {step_count} "
                        f"saved to {save_path}"
                        f"{' [flush]' if do_flush else ''}"
                    )

                # Always-on latest checkpoint (overwrites every step)
                if self.training_config.save_latest_checkpoint:
                    latest_path = checkpoint_dir / "checkpoint_latest"
                    self.model_trainer.save_model(str(latest_path))
                    self.model_trainer.save_training_state(
                        latest_path / "training_state.pt"
                    )
                    save_checkpoint_metadata(
                        latest_path,
                        CheckpointMetadata(
                            step_count=step_count,
                            batch_idx=batch_idx,
                            factory_state=latest_factory_state,
                            wandb_run_id=wandb_run_id,
                            timestamp=datetime.now().isoformat(),
                            training_mode=self.training_config.training_mode,
                        ),
                    )
                    logger.info(f"Latest checkpoint updated: step {step_count}")

                # Periodic checkpoint — interval based on batch_idx
                if (
                    self.training_config.save_checkpoint_every > 0
                    and batch_idx % self.training_config.save_checkpoint_every == 0
                ):
                    ckpt_path = checkpoint_dir / f"checkpoint_step_{step_count}"
                    self.model_trainer.save_model(str(ckpt_path))
                    self.model_trainer.save_training_state(
                        ckpt_path / "training_state.pt"
                    )
                    save_checkpoint_metadata(
                        ckpt_path,
                        CheckpointMetadata(
                            step_count=step_count,
                            batch_idx=batch_idx,
                            factory_state=latest_factory_state,
                            wandb_run_id=wandb_run_id,
                            timestamp=datetime.now().isoformat(),
                            training_mode=self.training_config.training_mode,
                        ),
                    )
                    logger.info(f"Checkpoint saved: {ckpt_path}")

                    keep = self.training_config.max_checkpoints_to_keep
                    if keep > 0:
                        cleanup_old_checkpoints(checkpoint_dir, keep=keep)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            # Shutdown ordering matters for sglang metric completeness:
            #   1. Stop the drainer thread so it stops racing the producer
            #      (it may still have items it hasn't logged yet).
            #   2. Tell the orchestrator process to exit; drain sample_queue
            #      so any pending put() unblocks.
            #   3. Wait for the orchestrator to fully exit. mp.Queue's feeder
            #      thread (in the producer) only flushes its in-memory buffer
            #      to the OS pipe at process teardown — see _orchestrator_main.
            #   4. NOW we can pump_until_empty: every metric the orchestrator
            #      ever enqueued is now visible on the consumer side.
            #   5. Then wandb.finish() commits the final logs.
            if sglang_drainer is not None:
                sglang_drainer.stop(timeout=5.0)

            control_queue.put(Shutdown())
            _drain_queue(sample_queue)
            orch_proc.join(timeout=30)
            if orch_proc.is_alive():
                logger.warning("Orchestrator didn't exit, killing")
                orch_proc.kill()
                orch_proc.join(timeout=5)

            if sglang_drainer is not None:
                try:
                    sglang_drainer.pump_until_empty()
                except Exception:
                    logger.exception("Final sglang pump failed")

            # Shutdown RPC workers
            self.model_trainer.rpc_shutdown()

            if wandb_active:
                import wandb
                wandb.finish()

            # Clean up training file handler
            train_fh.close()
            logging.getLogger().removeHandler(train_fh)

        total_elapsed = time.time() - t0
        logger.info(
            f"RLFlow done: {step_count} steps, {batch_idx} batches "
            f"in {total_elapsed:.1f}s"
        )
        return all_step_metrics
