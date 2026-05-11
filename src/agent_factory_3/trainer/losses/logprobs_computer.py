"""
LogprobsComputer - Model forward and per-token logprobs/entropy computation.

This module is responsible for:
1. Model forward pass to get hidden states
2. Computing per-token log probabilities and entropy

All outputs are aligned with token position:
- logprobs[i] = log P(token_i | context_<i)
- position 0 is padded with 0 (no "predict token 0")
"""

import torch
from typing import Any

from .linear_cross_entropy import linear_cross_entropy
from .torch_functional import FusedLinearForPPO


class LogprobsComputer:
    """
    Compute per-token log probabilities and entropy from model forward pass.

    All outputs are aligned with token position:
    - logprobs[i] = log P(token_i | context_<i)
    - position 0 is padded with 0 (since there's no "predict token 0")

    The gradient behavior is controlled by the caller:
    - Call under torch.no_grad() for inference (no gradients)
    - Call normally for training (gradients preserved)
    """

    def __init__(
        self,
        temperature: float = 1.0,
        backend: str = "triton",  # "triton" | "torch" | "cce" | "liger"
    ):
        self.temperature = float(temperature)
        self.backend = backend.lower()

        if self.backend not in ["triton", "torch", "cce", "liger"]:
            raise ValueError(f"backend must be 'triton', 'torch', 'cce', or 'liger', got '{backend}'")

    # --------- Model navigation helpers ---------

    def _unwrap_model(self, model) -> Any:
        """
        Extract the actual backbone model from wrappers (PEFT, Accelerate, etc.).

        Handles common wrapping patterns:
        - model.base_model.model.model (triple nested)
        - model.base_model.model (double nested)
        - model.model (single nested)
        """
        if (
            hasattr(model, "base_model")
            and hasattr(model.base_model, "model")
            and hasattr(model.base_model.model, "model")
        ):
            return model.base_model.model.model
        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            return model.base_model.model
        if hasattr(model, "model"):
            return model.model
        return model

    def _find_lm_head_owner(self, model) -> Any:
        """
        Find the module that owns lm_head.weight in the model hierarchy.

        Searches through common locations where lm_head might be defined.
        Raises RuntimeError if lm_head cannot be found.
        """
        candidates = [model]

        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            candidates.append(model.base_model.model)
        if hasattr(model, "model"):
            candidates.append(model.model)

        for m in candidates:
            if hasattr(m, "lm_head") and hasattr(m.lm_head, "weight"):
                return m

        raise RuntimeError("Could not find lm_head with weight in provided model.")

    # --------- Forward pass ---------

    def get_hidden_states(
        self,
        model,
        inputs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Execute model forward pass and return last hidden states.

        Args:
            model: The language model (can be wrapped by PEFT/Accelerate)
            inputs: Dict containing at least 'input_ids', optionally:
                - position_ids
                - attention_mask
                - routing_indices (for MoE models)
                - pixel_values, image_grid_thw (for vision models)

        Returns:
            last_hidden_state: Tensor of shape [B, T, D]

        Raises:
            RuntimeError: If model outputs don't contain hidden states
        """
        core = self._unwrap_model(model)

        valid_keys = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "pixel_values",
            "image_grid_thw",
            "cu_seq_lens_q",
            "cu_seq_lens_k",
            "max_length_k",
            "max_length_q",
            "routing_indices",
        ]
        model_inputs = {k: inputs[k] for k in valid_keys if k in inputs}
        model_inputs["use_cache"] = False

        outputs = core(**model_inputs)

        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        if hasattr(outputs, "hidden_states"):
            return outputs.hidden_states[-1]

        raise RuntimeError("Model outputs must contain last_hidden_state or hidden_states.")

    # --------- Core computation ---------

    def _compute_with_cce(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute logprobs using cut-cross-entropy backend.

        Temperature is supported via pre-scaling hidden states:
        (hidden / T) @ weight.T == logits / T.
        Entropy is handled separately by _compute_entropy_chunked.
        """
        try:
            from cut_cross_entropy import linear_cross_entropy as cce_fn
        except ImportError:
            raise ImportError(
                "cut_cross_entropy not installed. Install with: pip install cut-cross-entropy"
            )

        if self.temperature != 1.0:
            hidden_states = hidden_states / self.temperature

        # CCE returns loss = -log P(token)
        loss = cce_fn(hidden_states, weight, labels, reduction="none")
        log_probs = -loss  # Convert to logprobs
        return log_probs

    def _compute_with_liger(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        return_entropy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute logprobs (and optionally entropy) using liger-kernel backend.

        Liger expects 2D input [BT, H], so we flatten and restore shape.
        Temperature is supported via pre-scaling hidden states.
        Entropy is computed natively inside Liger's chunked forward loop
        (requires ycchen-tw/Liger-Kernel fork with return_entropy support).
        """
        try:
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
        except ImportError:
            raise ImportError(
                "liger-kernel not installed. Install with: pip install liger-kernel"
            )

        if self.temperature != 1.0:
            hidden_states = hidden_states / self.temperature

        # Liger expects 2D input [BT, H], flatten batch and sequence dims
        orig_shape = hidden_states.shape[:-1]  # [B, T]
        hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])  # [BT, H]
        labels_flat = labels.reshape(-1)  # [BT]

        # Liger's forward signature is (weight, input, target)
        liger_fn = LigerFusedLinearCrossEntropyLoss(reduction="none", return_entropy=return_entropy)
        result = liger_fn(weight, hidden_flat, labels_flat)

        if return_entropy:
            # Fork returns CrossEntropyOutput(loss=..., entropy=...)
            log_probs = (-result.loss).reshape(orig_shape)
            entropy = result.entropy.reshape(orig_shape)
        else:
            # Stock returns plain loss tensor
            log_probs = (-result).reshape(orig_shape)
            entropy = None

        return log_probs, entropy

    @torch.no_grad()
    def _compute_entropy_chunked(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        chunk_size: int = 512,
    ) -> torch.Tensor:
        """Compute per-token entropy via chunked matmul. For logging only — no gradients.

        H = logsumexp(logits) - sum(softmax(logits) * logits)
        """
        orig_shape = hidden_states.shape[:-1]  # [B, T]
        hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])  # [BT, D]
        T = hidden_flat.shape[0]
        entropy = torch.zeros(T, dtype=torch.float32, device=hidden_flat.device)

        for start in range(0, T, chunk_size):
            end = min(start + chunk_size, T)
            logits = (hidden_flat[start:end] @ weight.t()) / self.temperature
            logits = logits.float()
            lse = torch.logsumexp(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            entropy[start:end] = lse - (probs * logits).sum(-1)

        return entropy.reshape(orig_shape)

    def _compute_logprobs_from_hidden(
        self,
        model,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        return_entropy: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Compute per-token log probabilities and entropy from hidden states.

        Args:
            model: The language model (needed to access lm_head)
            hidden_states: Tensor of shape [B, T, D]
            labels: Target token IDs of shape [B, T]
            return_entropy: Whether to compute and return entropy

        Returns:
            log_probs: Per-token log probabilities, shape matches hidden_states batch dims
            entropy: Per-token entropy (or None if return_entropy=False)
        """
        head_owner = self._find_lm_head_owner(model)
        weight = head_owner.lm_head.weight

        if self.backend == "triton":
            hidden = hidden_states
            hidden_size = hidden.shape[-1]

            # Triton kernel requires hidden_size to be divisible by 32
            if hidden_size % 32 != 0:
                padded_size = ((hidden_size + 31) // 32) * 32
                pad_amount = padded_size - hidden_size
                hidden = torch.nn.functional.pad(hidden, (0, pad_amount), value=0.0)
                weight = torch.nn.functional.pad(weight, (0, pad_amount), value=0.0)

            log_probs, entropy = linear_cross_entropy(
                hidden,
                weight,
                labels,
                self.temperature,
                "none",
            )

        elif self.backend == "torch":
            fused_linear_for_ppo = FusedLinearForPPO()
            log_probs, entropy = fused_linear_for_ppo.forward(
                hidden_states=hidden_states,
                vocab_weights=weight,
                input_ids=labels.long(),
                temperature=self.temperature,
            )

        elif self.backend == "cce":
            log_probs = self._compute_with_cce(hidden_states, weight, labels)
            entropy = self._compute_entropy_chunked(hidden_states, weight) if return_entropy else None

        elif self.backend == "liger":
            log_probs, entropy = self._compute_with_liger(hidden_states, weight, labels, return_entropy)

        if not return_entropy:
            entropy = None

        entropy_out = entropy.to(torch.float32) if entropy is not None else None
        return log_probs.to(torch.float32), entropy_out

    # --------- Public API ---------

    def compute_logprobs(
        self,
        model,
        inputs: dict[str, torch.Tensor],
        return_entropy: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Compute per-token log probabilities and entropy aligned with input_ids.

        This is the main entry point. It performs:
        1. Model forward pass to get hidden states
        2. Compute logprobs using hidden[:-1] to predict input_ids[1:]
        3. Pad position 0 with zeros for alignment

        Args:
            model: The language model
            inputs: Dict containing at least 'input_ids', optionally 'position_ids', etc.
            return_entropy: Whether to compute and return entropy. Default True.
                For cce/liger backends, entropy is computed via a separate chunked pass.

        Returns:
            {
                'logps': Tensor[B, T] - log P(token_i | context_<i), position 0 = 0
                'entropies': Tensor[B, T] - entropy at each position, position 0 = 0
                    (only if return_entropy=True)
            }

        Raises:
            ValueError: If 'input_ids' is not in inputs
        """
        if "input_ids" not in inputs:
            raise ValueError("inputs must contain 'input_ids'.")

        batch_size, seq_len = inputs["input_ids"].shape

        # Get hidden states for all positions
        last_hidden_state = self.get_hidden_states(model, inputs)

        # hidden[i] predicts token[i+1], so:
        # - Use hidden[0:T-1] to predict tokens[1:T]
        hidden_for_pred = last_hidden_state[:, :-1, :]  # [B, T-1, D]
        labels = inputs["input_ids"][:, 1:]              # [B, T-1]

        # Compute logprobs and entropy
        log_probs, entropies = self._compute_logprobs_from_hidden(
            model=model,
            hidden_states=hidden_for_pred,
            labels=labels,
            return_entropy=return_entropy,
        )

        # Reshape to [B, T-1]
        log_probs = log_probs.reshape(batch_size, seq_len - 1)

        # Pad position 0 with zeros for alignment with input_ids
        # After padding: logprobs[i] = log P(token_i | context_<i)
        device = log_probs.device
        dtype = log_probs.dtype
        pad = torch.zeros(batch_size, 1, dtype=dtype, device=device)

        log_probs = torch.cat([pad, log_probs], dim=1)   # [B, T]

        result = {"logps": log_probs}

        if return_entropy and entropies is not None:
            entropies = entropies.reshape(batch_size, seq_len - 1)
            entropies = torch.cat([pad, entropies], dim=1)   # [B, T]
            result["entropies"] = entropies

        return result

    # --------- Utility methods ---------

    @staticmethod
    def unpack_packed_tensor(
        packed: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Unpack a packed tensor into a list of per-sample tensors.

        This is used to convert packed sequences (where multiple samples are
        concatenated) back into individual sample tensors.

        Args:
            packed: Packed tensor of shape [T] or [1, T]
            cu_seqlens: Cumulative sequence lengths of shape [N+1],
                        where N is the number of samples.
                        e.g., [0, L1, L1+L2, L1+L2+L3] for 3 samples

        Returns:
            List of N tensors, each of shape [Li] where Li is the length
            of sample i. Tensors stay on the same device as input.
        """
        # Ensure 1D [T]
        if packed.dim() == 2:
            packed = packed.squeeze(0)

        unpacked: list[torch.Tensor] = []
        num_samples = len(cu_seqlens) - 1

        for i in range(num_samples):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            unpacked.append(packed[start:end])

        return unpacked
