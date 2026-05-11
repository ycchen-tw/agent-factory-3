"""
ModelTrainer — single-step DDIS trainer with unpacked loss computation.

Ported from v2 ModelTrainerNew with the following changes:
- Removed: compute_old_logprobs, save_memory/resume_memory, _memory_saver_context
- Replaced: LossCalculatorNew → DDISLoss
- Added: reset_optimizer_state
"""

from __future__ import annotations

import contextlib
import gc
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from loguru import logger
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from torch.distributed.tensor import DTensor
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoModelForTextToWaveform,
    AutoProcessor,
)
from transformers.optimization import get_scheduler
from trl.models.utils import _ForwardRedirection

from .accelerate_rpc_mixin import AccelerateRPCMixin, rpc_method
from .data.data_collator_with_packing import DataCollatorWithPacking
from .data.rl_deterministic_packed_sampler import DeterministicPackedBatchSampler
from .losses.ddis_loss import DDISLoss
from .losses.logprobs_computer import LogprobsComputer
from .losses.metric_state import MeanMetricState
from .training_config import TrainingConfig
from .types import TrainingSample


class StepRLDataset(Dataset):
    """Minimal dataset adapter for a single training step."""

    def __init__(self, data: list[TrainingSample]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def get_sequence_lengths(self) -> list[int]:
        return [len(sample.input_ids) for sample in self.data]

    def __getitem__(self, idx: int) -> dict:
        sample = self.data[idx]
        item = {
            "input_ids": sample.input_ids,
            "completion_mask": sample.completion_mask,
            "advantages": sample.advantages,
            "gen_logprobs": sample.gen_logprobs,
            "sample_id": sample.sample_id,
            "idx": idx,
            "seq_token_count": sample.seq_token_count,
            "prompt_token_count": sample.prompt_token_count,
            "prompt_sequence_count": sample.prompt_sequence_count,
            "prompt_id": sample.prompt_id,
            "divisor": sample.divisor,
        }
        if sample.gen_entropy is not None:
            item["gen_entropy"] = sample.gen_entropy
        if sample.routing_indices is not None:
            item["routing_indices"] = sample.routing_indices
        return item


@dataclass
class StepAccumulator:
    loss_unscaled_sum: torch.Tensor | None = None
    metric_states: dict[str, MeanMetricState] = field(default_factory=dict)
    micro_batches: int = 0
    start_time: float = field(default_factory=time.perf_counter)

    def add(self, *, loss_unscaled: torch.Tensor, metric_states: dict[str, MeanMetricState]) -> None:
        self.micro_batches += 1
        loss_detached = loss_unscaled.detach()
        if self.loss_unscaled_sum is None:
            self.loss_unscaled_sum = loss_detached
        else:
            self.loss_unscaled_sum.add_(loss_detached)

        for k, st in metric_states.items():
            if k in self.metric_states:
                self.metric_states[k].merge_(st)
            else:
                self.metric_states[k] = st

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.start_time


class ModelTrainer(AccelerateRPCMixin):
    def __init__(
        self,
        config: TrainingConfig,
        accelerator: Accelerator,
        model: Any | None = None,
        processing_class: Any | None = None,
        optimizer: Any | tuple | None = None,
    ):
        self.config = config
        self.accelerator = accelerator
        self.accelerator.even_batches = False

        # Warn loud if caller built the model themselves but also set loader-only
        # TrainingConfig fields — those are not applied in this code path.
        if model is not None:
            loader_only = (
                "attn_implementation",
                "gradient_checkpointing",
                "use_reentrant",
                "model_use_cache",
                "model_init_kwargs",
            )
            ignored = {
                f: getattr(config, f)
                for f in loader_only
                if getattr(config, f) != TrainingConfig.model_fields[f].default
            }
            if ignored:
                logger.warning(
                    "ModelTrainer received an external `model`; these TrainingConfig "
                    "fields are loader-only and have NO effect on the passed model: "
                    f"{ignored}. Apply them where you call from_pretrained() or "
                    "model.gradient_checkpointing_enable()."
                )

        base_model = model or self._load_base_model()
        if self.config.training_mode == "lora":
            self.model = self._attach_lora(base_model)
        else:
            self.model = self._prepare_full_param(base_model)
        self.processing_class = processing_class or self._load_processing_class()
        self.optimizer, self.lr_scheduler = self._setup_optimizer(optimizer)
        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer,
        )

        self.logprobs_computer = LogprobsComputer(
            temperature=self.config.loss_temperature,
            backend=self.config.loss_backend,
        )
        self.loss_calculator = DDISLoss(
            eps_low=self.config.ddis_eps_low,
            eps_high=self.config.ddis_eps_high,
            use_opsm=self.config.use_opsm,
            opsm_delta=self.config.opsm_delta,
            entropy_top_pct=self.config.entropy_top_pct,
        )
        self._forward_redirection = _ForwardRedirection()

    # ==================== Model Init ====================

    def _load_base_model(self):
        """Load base model from config. Skipped when model is passed to __init__."""
        model_config = AutoConfig.from_pretrained(self.config.model_id)

        if type(model_config) in AutoModelForImageTextToText._model_mapping.keys():
            load_class = AutoModelForImageTextToText
        elif type(model_config) in AutoModelForSeq2SeqLM._model_mapping.keys():
            load_class = AutoModelForSeq2SeqLM
        elif type(model_config) in AutoModelForTextToWaveform._model_mapping.keys():
            load_class = AutoModelForTextToWaveform
        else:
            load_class = AutoModelForCausalLM

        model_init_kwargs = self.config.model_init_kwargs or {}
        if self._is_fsdp:
            # FSDP manages device placement — load to CPU, let FSDP shard + distribute.
            # With fsdp_cpu_ram_efficient_loading, only rank 0 loads weights.
            pass
        else:
            model_init_kwargs["device_map"] = f"cuda:{self.accelerator.process_index}"
        model = load_class.from_pretrained(
            self.config.model_id,
            attn_implementation=self.config.attn_implementation,
            **model_init_kwargs,
        )

        model.config.use_cache = self.config.model_use_cache
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": self.config.use_reentrant}
            )

        return model

    def _attach_lora(self, model):
        """Attach LoRA adapter. Called when training_mode='lora'."""
        lora_kwargs = self.config.lora_kwargs or {}
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            **lora_kwargs,
        )

        model = get_peft_model(
            model,
            lora_config,
            autocast_adapter_dtype=self.config.autocast_adapter_dtype,
        )
        model.enable_input_require_grads()

        if self.accelerator.is_main_process:
            logger.info(f"model: {model}")
            model.print_trainable_parameters()

        return model

    def _prepare_full_param(self, model):
        """Prepare model for full-parameter or selective module training.

        Called when training_mode='full'.
        - trainable_modules=None  → all parameters trainable
        - trainable_modules=[...] → only matching modules trainable (full-rank)
        """
        modules = self.config.trainable_modules
        if modules is not None:
            for param in model.parameters():
                param.requires_grad = False
            for name, param in model.named_parameters():
                if any(m in name for m in modules):
                    param.requires_grad = True

        # Always call for gradient checkpointing safety — ensures embedding output
        # has requires_grad=True even when early layers are frozen or when the
        # embedding itself isn't in trainable_modules.
        model.enable_input_require_grads()

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if trainable == 0:
            raise RuntimeError(
                f"Full-param mode: 0/{total:,} params are trainable. "
                f"trainable_modules={modules} matched nothing. "
                "Check your trainable_modules patterns."
            )

        if self.accelerator.is_main_process:
            logger.info(
                f"Full-param mode: {trainable:,}/{total:,} params trainable "
                f"({100 * trainable / total:.1f}%)"
            )

        return model

    def _load_processing_class(self):
        return AutoProcessor.from_pretrained(pretrained_model_name_or_path=self.config.model_id)

    def _setup_optimizer(self, optimizer) -> tuple:
        if isinstance(optimizer, tuple):
            opt, sched = optimizer
        else:
            opt = optimizer or self._create_optimizer()
            sched = None

        if sched is None:
            sched = get_scheduler(
                name=self.config.lr_scheduler_type,
                optimizer=opt,
                num_warmup_steps=self.config.lr_warmup_steps,
                num_training_steps=100_0000,
            )
        return opt, sched

    def _create_optimizer(self):
        if self.config.optimizer_type != "AdamW":
            raise ValueError(f"Unsupported optimizer type: {self.config.optimizer_type}")
        # Filter to trainable params only — avoids wasting Adam state memory on frozen params.
        # Safe for both LoRA (PEFT already freezes base) and full-param modes.
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            params=trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )

    # ==================== Dataloader ====================

    def _create_step_dataloader(
        self,
        step_samples: list[TrainingSample],
        *,
        max_capacity: int,
        pad_to_multiple_of: int,
    ) -> torch.utils.data.DataLoader:
        rl_dataset = StepRLDataset(step_samples)
        collate_fn = DataCollatorWithPacking(
            pad_token_id=self.processing_class.pad_token_id,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        batch_sampler = DeterministicPackedBatchSampler(
            dataset=rl_dataset,
            max_capacity=max_capacity,
            world_size=self.accelerator.num_processes,
            balance_ranks=True,
            verbose=False,
        )
        return torch.utils.data.DataLoader(
            rl_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            pin_memory=True,
            num_workers=self.config.dataloader_num_workers,
        )

    def _prepare_dataloader_iter(self, dataloader, desc: str):
        if hasattr(dataloader.batch_sampler, "set_epoch"):
            dataloader.batch_sampler.set_epoch(0)
        return tqdm(dataloader, desc=desc, leave=False, disable=not self.accelerator.is_main_process)

    # ==================== Forward ====================

    def _forward_batch(self, batch: dict, unwrapped_model) -> dict:
        logprobs_input = {
            "input_ids": batch["input_ids"],
            "position_ids": batch["position_ids"],
        }
        if self.config.use_routing_replay:
            if "routing_indices" not in batch:
                raise RuntimeError(
                    "config.use_routing_replay=True but 'routing_indices' not in batch. "
                    f"Available keys: {list(batch.keys())}"
                )
            logprobs_input["routing_indices"] = batch["routing_indices"]

        return self._forward_redirection(
            self.model,
            unwrapped_model,
            partial(self.logprobs_computer.compute_logprobs, return_entropy=True),
            unwrapped_model,
            logprobs_input,
        )

    @staticmethod
    def _unpack_packed_tensor(packed: torch.Tensor, cu_seqlens: torch.Tensor) -> list[torch.Tensor]:
        return LogprobsComputer.unpack_packed_tensor(packed, cu_seqlens)

    def _unpack_batch_for_samples(self, batch: dict, logps: torch.Tensor, num_samples: int) -> dict[str, list[torch.Tensor]]:
        cu = batch["cu_seqlens"]
        return {
            "logps": self._unpack_packed_tensor(logps, cu)[:num_samples],
            "gen_logprobs": self._unpack_packed_tensor(batch["gen_logprobs"], cu)[:num_samples],
            "completion_mask": self._unpack_packed_tensor(batch["completion_mask"], cu)[:num_samples],
            "advantages": self._unpack_packed_tensor(batch["advantages"], cu)[:num_samples],
        }

    # ==================== Loss ====================

    def _compute_microbatch(self, *, batch: dict, unwrapped_model) -> tuple[torch.Tensor, dict[str, MeanMetricState]]:
        sample_ids: list[str] = batch["sample_id"]
        num_samples = len(sample_ids)
        if num_samples == 0:
            raise RuntimeError("Empty microbatch: no samples")

        if "divisor" not in batch:
            raise RuntimeError("ModelTrainer requires precomputed 'divisor' per sample in the batch.")

        divisors = torch.tensor(batch["divisor"], device=self.accelerator.device, dtype=torch.float32)
        if divisors.numel() != num_samples:
            raise RuntimeError(f"divisor length mismatch: got {divisors.numel()} expected {num_samples}")

        logprobs_dict = self._forward_batch(batch, unwrapped_model)
        logps = logprobs_dict["logps"]
        entropies = logprobs_dict.get("entropies")  # [1, T_packed] or None

        unpacked = self._unpack_batch_for_samples(batch, logps, num_samples)

        # Unpack entropies for entropy token masking
        unpacked_entropies = None
        if entropies is not None:
            cu = batch["cu_seqlens"]
            unpacked_entropies = self._unpack_packed_tensor(entropies, cu)[:num_samples]

        outputs = self.loss_calculator.compute_unpacked(
            log_probs=unpacked["logps"],
            completion_mask=unpacked["completion_mask"],
            advantages=unpacked["advantages"],
            divisors=divisors,
            gen_log_probs=unpacked["gen_logprobs"],
            entropies=unpacked_entropies,
        )
        metric_states = outputs.metric_states

        # Token-weighted mean entropy over completion tokens
        mask = batch["completion_mask"].float()
        if entropies is not None:
            metric_states["algo/response_entropy"] = MeanMetricState(
                sum=(entropies.detach() * mask).sum(),
                count=mask.sum().to(torch.float32),
                weighting="token",
            )

        # Generation-time entropy (from sglang)
        if "gen_entropy" in batch:
            gen_ent = batch["gen_entropy"]
            metric_states["algo/gen_entropy"] = MeanMetricState(
                sum=(gen_ent.detach() * mask).sum(),
                count=mask.sum().to(torch.float32),
                weighting="token",
            )

        return outputs.loss_unscaled, metric_states

    # ==================== Optimizer Step ====================

    def _perform_optimizer_step(self, accumulator: StepAccumulator, step_count: int) -> dict[str, float]:
        grad_norm = None
        if self.config.max_grad_norm > 0:
            grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        lr = float(self.optimizer.param_groups[0]["lr"])

        loss_sum = accumulator.loss_unscaled_sum
        if loss_sum is None:
            loss_sum = torch.zeros((), device=self.accelerator.device, dtype=torch.float32)

        # Reduce loss + metrics across ranks
        loss_sum_detached = loss_sum.detach()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(loss_sum_detached, op=torch.distributed.ReduceOp.SUM)
            for st in accumulator.metric_states.values():
                st.all_reduce_()

        step_metrics: dict[str, float] = {
            "loss": float(loss_sum_detached.item()),
            "learning_rate": lr,
            "step_time": accumulator.elapsed,
            "num_batches": float(accumulator.micro_batches),
        }
        if grad_norm is not None:
            step_metrics["grad_norm"] = float(grad_norm.item()) if hasattr(grad_norm, "item") else float(grad_norm)

        for k, st in accumulator.metric_states.items():
            step_metrics[k] = float(st.mean().item())

        if self.accelerator.is_main_process:
            logger.info(
                f"  Step {step_count}: loss={step_metrics['loss']:.4f}"
                + (f", grad_norm={step_metrics['grad_norm']:.4f}" if "grad_norm" in step_metrics else "")
                + f", lr={lr:.2e}"
            )
        return step_metrics

    def _training_cleanup(self):
        if hasattr(self.accelerator, "_dataloaders"):
            self.accelerator._dataloaders.clear()
        gc.collect()
        torch.cuda.empty_cache()

    # ==================== RPC Methods ====================

    @rpc_method(gather=False)
    def train_step(self, step_samples: list[TrainingSample]) -> dict[str, float]:
        if not step_samples:
            raise ValueError("train_step received empty step_samples")

        if any(s.divisor is None or s.divisor == 0 for s in step_samples):
            raise ValueError("train_step requires all samples to have a non-null, non-zero divisor.")

        dataloader = self._create_step_dataloader(
            step_samples,
            max_capacity=self.config.max_capacity,
            pad_to_multiple_of=self.config.pad_to_multiple_of_training,
        )
        dataloader = self.accelerator.prepare(dataloader)

        self.model.train()
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        accumulator = StepAccumulator()

        # DDP/FSDP averages gradients across ranks, so scale loss by world_size
        factor = float(self.accelerator.num_processes)

        dataloader_iter = self._prepare_dataloader_iter(dataloader, "Training (step)")
        num_microbatches = len(dataloader)

        for mb_idx, batch in enumerate(dataloader_iter):
            sync = mb_idx == (num_microbatches - 1)
            sync_ctx = contextlib.nullcontext() if sync else self.accelerator.no_sync(self.model)

            with sync_ctx:
                loss_unscaled, metric_states = self._compute_microbatch(batch=batch, unwrapped_model=unwrapped_model)
                self.accelerator.backward(loss_unscaled * factor)

            accumulator.add(loss_unscaled=loss_unscaled, metric_states=metric_states)

        step_metrics = self._perform_optimizer_step(accumulator, step_count=0)

        seq_lens = [len(s.input_ids) for s in step_samples]
        comp_lens = [s.num_loss_tokens() for s in step_samples]
        total_tokens = sum(seq_lens)
        step_metrics["train/tokens_per_sec"] = total_tokens / step_metrics["step_time"]
        step_metrics["train/seq_len_mean"] = total_tokens / len(seq_lens)
        step_metrics["train/seq_len_max"] = float(max(seq_lens))
        step_metrics["train/completion_len_mean"] = sum(comp_lens) / len(comp_lens)
        step_metrics["train/completion_len_max"] = float(max(comp_lens))

        self.accelerator.wait_for_everyone()
        self._training_cleanup()
        return step_metrics

    @rpc_method(gather=False)
    def reset_optimizer_state(self):
        """Reset Adam momentum states (DDIS: on weight sync)."""
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                state = self.optimizer.state.get(p)
                if state:
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in state:
                            state[key].zero_()
        self.accelerator.wait_for_everyone()

    # ==================== Save / Load ====================

    @rpc_method(broadcast_args=True, gather=False)
    def save_model(self, save_dir: str | Path):
        save_dir = Path(save_dir)
        if self.accelerator.is_main_process:
            save_dir.mkdir(parents=True, exist_ok=True)

        if self.config.training_mode == "lora":
            if self._is_fsdp:
                self._save_fsdp_lora(self.model, save_dir)
            else:
                self._save_standard_lora(self.model, save_dir)
        else:
            if self._is_fsdp:
                self._save_fsdp_full(self.model, save_dir)
            else:
                self._save_standard_full(self.model, save_dir)

    @property
    def _is_fsdp(self) -> bool:
        from accelerate.state import DistributedType
        return self.accelerator.state.distributed_type == DistributedType.FSDP

    def _save_standard_lora(self, model, save_dir: Path, adapter_name: str = "default"):
        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(save_dir)
        self.accelerator.wait_for_everyone()

    def _save_fsdp_lora(self, model, save_dir: Path, adapter_name: str = "default"):
        unwrapped = self.accelerator.unwrap_model(model)
        state_dict = get_peft_model_state_dict(unwrapped, adapter_name=adapter_name)
        # full_tensor() is a collective op — all ranks must call it
        gathered = {}
        for key, value in state_dict.items():
            full_val = value.full_tensor() if isinstance(value, DTensor) else value
            if self.accelerator.is_main_process:
                gathered[key] = full_val.cpu()
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            torch.save(gathered, save_dir / "adapter_model.bin")
            unwrapped.peft_config[adapter_name].save_pretrained(save_dir)

    def _save_standard_full(self, model, save_dir: Path):
        """Save trainable parameters for full-param mode (single-GPU / DDP)."""
        if self.accelerator.is_main_process:
            from safetensors.torch import save_file

            unwrapped = self.accelerator.unwrap_model(model)
            trainable_keys = {n for n, p in unwrapped.named_parameters() if p.requires_grad}
            sd = {k: v.cpu() for k, v in unwrapped.state_dict().items() if k in trainable_keys}
            save_file(sd, save_dir / "trainable_params.safetensors")
            logger.info(f"Saved {len(sd)} trainable params to {save_dir}")
        self.accelerator.wait_for_everyone()

    def _save_fsdp_full(self, model, save_dir: Path):
        """Save trainable parameters for full-param mode (FSDP2)."""
        from safetensors.torch import save_file

        gathered = self._gather_trainable_state_dict(model)
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            save_file(gathered, save_dir / "trainable_params.safetensors")
            logger.info(f"Saved {len(gathered)} trainable params to {save_dir} (FSDP)")

    def _gather_trainable_state_dict(self, model) -> dict[str, torch.Tensor]:
        """Gather trainable parameters from all FSDP ranks onto rank 0.

        All ranks must participate (full_tensor() is a collective op).
        Only rank 0 gets populated dict; other ranks get empty dict.
        Also works for non-FSDP (DTensor check is a no-op).
        """
        unwrapped = self.accelerator.unwrap_model(model)
        trainable_keys = {n for n, p in unwrapped.named_parameters() if p.requires_grad}
        full_sd = unwrapped.state_dict()

        gathered: dict[str, torch.Tensor] = {}
        for key, value in full_sd.items():
            # All ranks must call full_tensor() to keep FSDP collectives in sync
            full_val = value.full_tensor() if isinstance(value, DTensor) else value
            if self.accelerator.is_main_process and key in trainable_keys:
                gathered[key] = full_val.cpu()

        return gathered

    @rpc_method(broadcast_args=True, gather=False)
    def save_merged_weights(self, save_dir: str | Path) -> None:
        """Save trainable weights for sglang weight sync (model.safetensors).

        For LoRA: merges base + LoRA delta into original HF key names.
        For full-param: saves trainable parameters directly (keys are already HF format).

        Compatible with sglang's update_weights_from_disk endpoint.
        """
        save_dir = Path(save_dir)
        if self.accelerator.is_main_process:
            save_dir.mkdir(parents=True, exist_ok=True)

        if self.config.training_mode == "lora":
            merged_sd = self._build_merged_state_dict()
        else:
            merged_sd = self._build_trainable_state_dict()

        if self.accelerator.is_main_process:
            from safetensors.torch import save_file

            save_file(merged_sd, save_dir / "model.safetensors")
            logger.info(f"Merged weights saved: {save_dir} ({len(merged_sd)} tensors)")

        self.accelerator.wait_for_everyone()

    def _build_merged_state_dict(self) -> dict[str, torch.Tensor]:
        """Build state dict with LoRA merged into base weights (new tensors only).

        Only includes the layers that LoRA modifies. The merge is computed in fp32
        to avoid bf16 precision loss, then cast back to the original dtype.

        Returns:
            Dict mapping original HF model keys to merged weight tensors (on CPU).
            Only populated on rank 0; other ranks return empty dict.
        """
        unwrapped = self.accelerator.unwrap_model(self.model)
        full_sd = unwrapped.state_dict()
        scaling = self.config.lora_alpha / self.config.lora_r

        # Get the actual PEFT adapter name (typically "default" from get_peft_model)
        adapter_name = unwrapped.active_adapter
        if isinstance(adapter_name, list):
            adapter_name = adapter_name[0]

        # Pass 1: gather all LoRA tensors (DTensor.full_tensor() is a collective op,
        # so all ranks must participate, but only rank 0 stores the result)
        lora_a_suffix = f".lora_A.{adapter_name}.weight"
        lora_b_suffix = f".lora_B.{adapter_name}.weight"

        lora_tensors: dict[str, torch.Tensor] = {}
        for key, value in full_sd.items():
            if not (key.endswith(lora_a_suffix) or key.endswith(lora_b_suffix)):
                continue
            gathered = value.full_tensor() if isinstance(value, DTensor) else value
            if self.accelerator.is_main_process:
                lora_tensors[key] = gathered.cpu()

        # Pass 2: for each base_layer.weight, compute merged weight.
        # All ranks must iterate in the same order so FSDP collectives stay in sync.
        merged: dict[str, torch.Tensor] = {}
        for key, value in full_sd.items():
            if ".base_layer.weight" not in key:
                continue

            # Gather base weight (collective op for FSDP)
            base_w = value.full_tensor() if isinstance(value, DTensor) else value

            if self.accelerator.is_main_process:
                prefix = key.rsplit(".base_layer.weight", 1)[0]
                lora_a_key = f"{prefix}{lora_a_suffix}"
                lora_b_key = f"{prefix}{lora_b_suffix}"

                if lora_a_key not in lora_tensors or lora_b_key not in lora_tensors:
                    raise RuntimeError(
                        f"Missing LoRA tensors for {prefix}: expected adapter '{adapter_name}', "
                        f"but {lora_a_key} or {lora_b_key} not found in state_dict"
                    )

                lora_a = lora_tensors[lora_a_key]
                lora_b = lora_tensors[lora_b_key]

                # fp32 merge on CPU for precision (and to avoid GPU memory pressure)
                base_fp32 = base_w.cpu().float()
                delta = (lora_b.float() @ lora_a.float()) * scaling
                merged_w = (base_fp32 + delta).to(base_w.dtype)

                # Restore original HF key: strip PEFT prefixes
                # "base_model.model.model.layers.0...base_layer.weight"
                # → "model.layers.0...weight"
                original_key = key.replace(".base_layer.", ".")
                original_key = original_key.removeprefix("base_model.model.")
                merged[original_key] = merged_w

        return merged

    def _build_trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Build state dict of trainable parameters for full-param mode.

        For non-PEFT models, state_dict keys are already in HF format —
        no prefix stripping needed. Compatible with sglang's update_weights_from_disk.

        Returns:
            Dict mapping HF model keys to weight tensors (on CPU).
            Only populated on rank 0; other ranks return empty dict.
        """
        return self._gather_trainable_state_dict(self.model)

    @rpc_method(broadcast_args=True, gather=False)
    def load_model(self, load_dir: str | Path):
        load_dir = Path(load_dir)
        if not load_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {load_dir}")

        if self.config.training_mode == "lora":
            if self._is_fsdp:
                self._load_fsdp_lora(load_dir)
            else:
                self._load_standard_lora(load_dir)
        else:
            if self._is_fsdp:
                self._load_fsdp_full(load_dir)
            else:
                self._load_standard_full(load_dir)

    def _load_standard_lora(self, load_dir: Path):
        weight_file = self._find_adapter_file(load_dir)
        state_dict = self._load_adapter_file(weight_file)
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        set_peft_model_state_dict(unwrapped_model, state_dict)
        self.accelerator.wait_for_everyone()

    def _load_fsdp_lora(self, load_dir: Path):
        from torch.distributed.checkpoint.state_dict import (
            set_model_state_dict, StateDictOptions,
        )
        # Only rank 0 loads; DCP broadcasts + reshards to all ranks
        state_dict = {}
        if self.accelerator.is_main_process:
            weight_file = self._find_adapter_file(load_dir)
            raw_sd = self._load_adapter_file(weight_file)
            # get_peft_model_state_dict saves keys WITHOUT adapter name:
            #   ...lora_A.weight
            # but model.state_dict() uses keys WITH adapter name:
            #   ...lora_A.default.weight
            # set_model_state_dict expects the model key format, so we convert.
            unwrapped = self.accelerator.unwrap_model(self.model)
            adapter_name = unwrapped.active_adapter
            if isinstance(adapter_name, list):
                adapter_name = adapter_name[0]
            state_dict = {
                k.replace(".lora_A.", f".lora_A.{adapter_name}.")
                 .replace(".lora_B.", f".lora_B.{adapter_name}.")
                 .replace(".lora_embedding_A.", f".lora_embedding_A.{adapter_name}.")
                 .replace(".lora_embedding_B.", f".lora_embedding_B.{adapter_name}."): v
                for k, v in raw_sd.items()
            }
        set_model_state_dict(
            model=self.model,
            model_state_dict=state_dict,
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
                strict=False,
            ),
        )
        self.accelerator.wait_for_everyone()

    @staticmethod
    def _find_adapter_file(load_dir: Path) -> Path:
        for name in ("adapter_model.bin", "adapter_model.safetensors"):
            f = load_dir / name
            if f.exists():
                return f
        raise FileNotFoundError(f"No adapter weights found in {load_dir}")

    @staticmethod
    def _load_adapter_file(weight_file: Path) -> dict:
        if weight_file.suffix == ".safetensors":
            from safetensors.torch import load_file
            return load_file(str(weight_file))
        return torch.load(weight_file, map_location="cpu", weights_only=False)

    def _load_standard_full(self, load_dir: Path):
        """Load trainable parameters for full-param mode (single-GPU / DDP)."""
        from safetensors.torch import load_file

        weight_file = load_dir / "trainable_params.safetensors"
        if not weight_file.exists():
            raise FileNotFoundError(f"No trainable_params.safetensors in {load_dir}")
        state_dict = load_file(str(weight_file))
        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.load_state_dict(state_dict, strict=False)
        self.accelerator.wait_for_everyone()

    def _load_fsdp_full(self, load_dir: Path):
        """Load trainable parameters for full-param mode (FSDP2).

        Rank 0 loads the file; DCP broadcasts + reshards to all ranks.
        strict=False because we only saved the trainable subset.
        """
        from torch.distributed.checkpoint.state_dict import (
            set_model_state_dict, StateDictOptions,
        )

        state_dict = {}
        if self.accelerator.is_main_process:
            from safetensors.torch import load_file

            weight_file = load_dir / "trainable_params.safetensors"
            if not weight_file.exists():
                raise FileNotFoundError(f"No trainable_params.safetensors in {load_dir}")
            state_dict = load_file(str(weight_file))

        set_model_state_dict(
            model=self.model,
            model_state_dict=state_dict,
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
                strict=False,
            ),
        )
        self.accelerator.wait_for_everyone()

    @rpc_method(broadcast_args=True, gather=False)
    def save_training_state(self, save_path: Path):
        if self._is_fsdp:
            self._save_fsdp_training_state(save_path)
        else:
            if self.accelerator.is_main_process:
                training_state = {"optimizer": self.optimizer.state_dict()}
                if self.lr_scheduler is not None:
                    training_state["lr_scheduler"] = self.lr_scheduler.state_dict()
                torch.save(training_state, save_path)
            self.accelerator.wait_for_everyone()

    def _save_fsdp_training_state(self, save_path: Path):
        from torch.distributed.checkpoint.state_dict import (
            get_optimizer_state_dict, StateDictOptions,
        )
        optim_state = get_optimizer_state_dict(
            self.model, self.optimizer,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        if self.accelerator.is_main_process:
            training_state = {"optimizer": optim_state}
            if self.lr_scheduler is not None:
                training_state["lr_scheduler"] = self.lr_scheduler.state_dict()
            torch.save(training_state, save_path)
        self.accelerator.wait_for_everyone()

    @rpc_method(broadcast_args=True, gather=False)
    def load_training_state(self, load_path: Path):
        if self._is_fsdp:
            self._load_fsdp_training_state(load_path)
        else:
            training_state = torch.load(load_path, map_location="cpu", weights_only=False)
            self.optimizer.load_state_dict(training_state["optimizer"])
            if "lr_scheduler" in training_state and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(training_state["lr_scheduler"])
            self.accelerator.wait_for_everyone()

    def _load_fsdp_training_state(self, load_path: Path):
        from torch.distributed.checkpoint.state_dict import (
            set_optimizer_state_dict, StateDictOptions,
        )
        import torch.distributed as dist

        # Only rank 0 loads; broadcast_from_rank0 distributes to all ranks
        optim_state = {}
        lr_state = None
        if self.accelerator.is_main_process:
            training_state = torch.load(load_path, map_location="cpu", weights_only=False)
            optim_state = training_state["optimizer"]
            lr_state = training_state.get("lr_scheduler")

        # Patch missing optimizer keys: checkpoint may lack state for params
        # added after the checkpoint was saved (e.g. sinks).  Fill them with
        # empty dicts so set_optimizer_state_dict doesn't KeyError.
        if self.accelerator.is_main_process and optim_state.get("state"):
            model_keys = {n for n, p in self.model.named_parameters() if p.requires_grad}
            for k in model_keys:
                if k not in optim_state["state"]:
                    optim_state["state"][k] = {}
                    logger.info(f"Optimizer resume: added empty state for missing key '{k}'")

        set_optimizer_state_dict(
            self.model, self.optimizer,
            optim_state_dict=optim_state,
            options=StateDictOptions(
                full_state_dict=True,
                broadcast_from_rank0=True,
            ),
        )

        if self.lr_scheduler is not None:
            lr_state_list = [lr_state]
            dist.broadcast_object_list(lr_state_list, src=0)
            if lr_state_list[0] is not None:
                self.lr_scheduler.load_state_dict(lr_state_list[0])

        self.accelerator.wait_for_everyone()
