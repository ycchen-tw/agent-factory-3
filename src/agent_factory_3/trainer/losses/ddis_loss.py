"""
DDISLoss — Double-sided hard-mask policy gradient loss.

DDIS vs PPO/CISPO:
- Anchor is always gen_logprobs (no old_logprobs, no extra forward pass)
- Out-of-range tokens get gradient = 0 (hard mask, both sides, regardless of advantage sign)
- Divisor-based aggregation is identical to v2 LossCalculatorNew

The DDIS mask is a per-token modifier (same level as PPO clip), not a loss reduction change.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence

from .metric_state import MeanMetricState


@dataclass(frozen=True)
class DDISLossOutputs:
    loss_unscaled: torch.Tensor
    metric_states: dict[str, MeanMetricState]


def _mean_state_from_samples(
    values_per_sample: torch.Tensor,
    valid_samples: torch.Tensor,
) -> MeanMetricState:
    values = values_per_sample.detach()
    return MeanMetricState(
        sum=values[valid_samples].sum(),
        count=valid_samples.sum().to(dtype=torch.float32),
        weighting="sample",
    )


class DDISLoss:
    """Double-sided hard-mask policy gradient loss.

    Args:
        eps_low:  lower epsilon — tokens with ratio < 1 - eps_low are masked.
        eps_high: upper epsilon — tokens with ratio > 1 + eps_high are masked.
        use_opsm: enable Off-Policy Sequence Masking (negative seqs with high KL → loss zeroed).
        opsm_delta: KL threshold for OPSM. Only used when use_opsm=True.
    """

    def __init__(self, *, eps_low: float = 0.2, eps_high: float = 0.2,
                 use_opsm: bool = False, opsm_delta: float = 1e-4,
                 entropy_top_pct: float | None = None):
        if eps_low < 0:
            raise ValueError(f"eps_low must be >= 0, got {eps_low}")
        if eps_high < 0:
            raise ValueError(f"eps_high must be >= 0, got {eps_high}")
        self.eps_low = eps_low
        self.eps_high = eps_high
        self.use_opsm = use_opsm
        self.opsm_delta = opsm_delta
        self.entropy_top_pct = entropy_top_pct

    def compute_unpacked(
        self,
        *,
        log_probs: list[torch.Tensor],       # [T_i] current policy logprobs
        completion_mask: list[torch.Tensor],  # [T_i] 0/1
        advantages: list[torch.Tensor],       # [T_i]
        divisors: torch.Tensor,               # [B]
        gen_log_probs: list[torch.Tensor],    # [T_i] rollout-time logprobs (constant)
        entropies: list[torch.Tensor] | None = None,  # [T_i] train-time entropy (for token masking)
    ) -> DDISLossOutputs:
        """Compute DDIS loss from unpacked per-sample sequences.

        Metrics are emitted as mergeable (sum, count) states; caller must all-reduce.
        """
        bsz = len(log_probs)
        if len(completion_mask) != bsz or len(advantages) != bsz or len(gen_log_probs) != bsz:
            raise ValueError("Mismatched list lengths in compute_unpacked inputs")
        if divisors.numel() != bsz:
            raise ValueError(f"divisors must have shape [B], got numel={divisors.numel()} B={bsz}")
        if bsz == 0:
            raise ValueError("compute_unpacked received empty log_probs list")

        # Validate per-sample tensor lengths are consistent
        for i in range(bsz):
            n = log_probs[i].shape[0]
            assert completion_mask[i].shape[0] == n, f"Sample {i}: completion_mask len {completion_mask[i].shape[0]} != log_probs len {n}"
            assert advantages[i].shape[0] == n, f"Sample {i}: advantages len {advantages[i].shape[0]} != log_probs len {n}"
            assert gen_log_probs[i].shape[0] == n, f"Sample {i}: gen_log_probs len {gen_log_probs[i].shape[0]} != log_probs len {n}"

        device = log_probs[0].device

        # Pad to [B, T_max]
        logp = pad_sequence(log_probs, batch_first=True, padding_value=0.0).to(dtype=torch.float32)
        comp = pad_sequence(completion_mask, batch_first=True, padding_value=0.0).to(dtype=torch.float32)
        adv = pad_sequence(advantages, batch_first=True, padding_value=0.0).to(dtype=torch.float32)
        gen = pad_sequence(gen_log_probs, batch_first=True, padding_value=0.0).to(dtype=torch.float32).detach()
        ent = None
        if entropies is not None:
            if len(entropies) != bsz:
                raise ValueError(f"entropies list length {len(entropies)} != bsz {bsz}")
            for i in range(bsz):
                n = log_probs[i].shape[0]
                assert entropies[i].shape[0] == n, f"Sample {i}: entropies len {entropies[i].shape[0]} != log_probs len {n}"
            ent = pad_sequence(entropies, batch_first=True, padding_value=0.0).to(dtype=torch.float32).detach()
        elif self.entropy_top_pct is not None:
            raise RuntimeError(
                "entropy_top_pct is set but entropies not provided. "
                "Ensure the loss backend supports return_entropy=True (triton/torch/liger fork)."
            )

        completion_bool = comp > 0
        completion_count = comp.sum(dim=1)  # [B]
        has_completion = completion_count > 0

        # Token-level log ratio (only meaningful on completion tokens)
        log_ratio = (logp - gen).clamp(-20, 20)
        safe_log_ratio = torch.where(completion_bool, log_ratio, torch.zeros_like(log_ratio))
        ratio = torch.exp(safe_log_ratio)

        # DDIS double-sided hard mask
        in_range = (ratio > 1.0 - self.eps_low) & (ratio < 1.0 + self.eps_high)
        ddis_mask = in_range.float()

        # Effective mask = completion AND in-range
        effective_mask = ddis_mask * comp
        effective_count = effective_mask.sum(dim=1)
        has_effective = effective_count > 0

        # Entropy token masking: keep only top x% highest-entropy completion tokens
        entropy_mask_applied = None
        entropy_scale = None
        if self.entropy_top_pct is not None and ent is not None:
            masked_ent = torch.where(comp > 0, ent, torch.full_like(ent, -float('inf')))
            sorted_ent, _ = masked_ent.sort(dim=1, descending=True)
            k = (completion_count * self.entropy_top_pct / 100.0).clamp(min=1).long()
            threshold = sorted_ent.gather(1, (k - 1).unsqueeze(1)).squeeze(1)  # [B]
            entropy_token_mask = (ent >= threshold.unsqueeze(1)).float() * comp
            effective_mask = effective_mask * entropy_token_mask  # stays binary 0/1
            # Per-sample scale to preserve loss magnitude
            kept_count = entropy_token_mask.sum(dim=1)
            entropy_scale = completion_count / kept_count.clamp_min(1.0)  # [B]
            entropy_mask_applied = entropy_token_mask

        # Loss: -(ratio.detach() * adv * logp * mask), divided by per-sample divisor
        per_token_loss = -(ratio.detach() * adv * logp * effective_mask)
        per_sample_loss_sum = per_token_loss.sum(dim=1)  # [B]

        # Entropy scale compensation (per-sample)
        if entropy_scale is not None:
            per_sample_loss_sum = per_sample_loss_sum * entropy_scale

        # OPSM: mask out negative-advantage sequences with high KL
        opsm_mask = None
        if self.use_opsm:
            mean_adv = (adv * comp).sum(dim=1) / completion_count.clamp_min(1.0)
            kl = ((gen - logp).detach() * comp).sum(dim=1) / completion_count.clamp_min(1.0)
            opsm_mask = (mean_adv < 0) & (kl > self.opsm_delta)
            per_sample_loss_sum = per_sample_loss_sum * (~opsm_mask).float()

        divisors_f = divisors.to(device=device, dtype=torch.float32)
        assert (divisors_f > 0).all(), f"divisors must be positive, got {divisors_f}"
        loss_unscaled = (per_sample_loss_sum / divisors_f).sum()

        # ---- Metrics ----
        metric_states: dict[str, MeanMetricState] = {}

        # DDIS mask rate: fraction of completion tokens clipped by DDIS (pure DDIS, no entropy)
        ddis_masked_count = (comp * (1.0 - ddis_mask)).sum(dim=1)
        ddis_mask_rate = ddis_masked_count / completion_count.clamp_min(1.0)
        metric_states["algo/ddis_mask_rate"] = _mean_state_from_samples(ddis_mask_rate, has_completion)

        # Mask rate split by advantage sign
        pos_comp = (adv > 0).float() * comp
        pos_count = pos_comp.sum(dim=1)
        has_pos = pos_count > 0
        pos_masked = (pos_comp * (1.0 - ddis_mask)).sum(dim=1)
        mask_rate_pos = pos_masked / pos_count.clamp_min(1.0)
        metric_states["algo/mask_rate_pos"] = _mean_state_from_samples(mask_rate_pos, has_pos)

        neg_comp = (adv < 0).float() * comp
        neg_count = neg_comp.sum(dim=1)
        has_neg = neg_count > 0
        neg_masked = (neg_comp * (1.0 - ddis_mask)).sum(dim=1)
        mask_rate_neg = neg_masked / neg_count.clamp_min(1.0)
        metric_states["algo/mask_rate_neg"] = _mean_state_from_samples(mask_rate_neg, has_neg)

        # Mask rate by direction (low = ratio dropped below trust region, high = above)
        ratio_detached = ratio.detach()
        mask_rate_low = ((ratio_detached <= 1.0 - self.eps_low).float() * comp).sum(dim=1) / completion_count.clamp_min(1.0)
        metric_states["algo/mask_rate_low"] = _mean_state_from_samples(mask_rate_low, has_completion)
        mask_rate_high = ((ratio_detached >= 1.0 + self.eps_high).float() * comp).sum(dim=1) / completion_count.clamp_min(1.0)
        metric_states["algo/mask_rate_high"] = _mean_state_from_samples(mask_rate_high, has_completion)

        # Ratio stats over completion tokens
        ratio_sum = (ratio_detached * comp).sum(dim=1)
        ratio_mean = ratio_sum / completion_count.clamp_min(1.0)
        ratio_for_max = torch.where(completion_bool, ratio_detached, torch.full_like(ratio_detached, -torch.inf))
        ratio_for_min = torch.where(completion_bool, ratio_detached, torch.full_like(ratio_detached, torch.inf))
        ratio_max = ratio_for_max.max(dim=1).values
        ratio_min = ratio_for_min.min(dim=1).values

        ratio_sq_sum = ((ratio_detached ** 2) * comp).sum(dim=1)
        ratio_second = ratio_sq_sum / completion_count.clamp_min(1.0)
        ratio_var = (ratio_second - ratio_mean ** 2).clamp(min=0.0)
        ratio_std = torch.sqrt(ratio_var)

        metric_states["algo/ratio_mean"] = _mean_state_from_samples(ratio_mean, has_completion)
        metric_states["algo/ratio_max"] = _mean_state_from_samples(ratio_max, has_completion)
        metric_states["algo/ratio_min"] = _mean_state_from_samples(ratio_min, has_completion)
        metric_states["algo/ratio_std"] = _mean_state_from_samples(ratio_std, has_completion)

        # Logprobs mean over completion tokens
        logprobs_sum = (logp * comp).sum(dim=1)
        logprobs_mean = logprobs_sum / completion_count.clamp_min(1.0)
        metric_states["algo/logprobs_mean"] = _mean_state_from_samples(logprobs_mean, has_completion)

        logprobs_pos = (logp * pos_comp).sum(dim=1) / pos_count.clamp_min(1.0)
        metric_states["algo/logprobs_mean_pos"] = _mean_state_from_samples(logprobs_pos, has_pos)
        logprobs_neg = (logp * neg_comp).sum(dim=1) / neg_count.clamp_min(1.0)
        metric_states["algo/logprobs_mean_neg"] = _mean_state_from_samples(logprobs_neg, has_neg)

        gen_logprobs_sum = (gen * comp).sum(dim=1)
        gen_logprobs_mean = gen_logprobs_sum / completion_count.clamp_min(1.0)
        metric_states["algo/gen_logprobs_mean"] = _mean_state_from_samples(gen_logprobs_mean, has_completion)

        # KL(π_gen ‖ π_train) ≈ mean(gen_logp - train_logp) over completion tokens
        kl_per_sample = ((gen - logp) * comp).sum(dim=1) / completion_count.clamp_min(1.0)
        metric_states["algo/kl_gen_train"] = _mean_state_from_samples(kl_per_sample, has_completion)

        # OPSM metrics
        if self.use_opsm and opsm_mask is not None:
            metric_states["opsm/drop_rate"] = _mean_state_from_samples(opsm_mask.float(), has_completion)
            metric_states["opsm/drop_count"] = MeanMetricState(
                sum=opsm_mask.float().sum().detach(),
                count=torch.tensor(1.0, device=device),
                weighting="token",
            )

        # Token rates after all masks (DDIS + entropy combined)
        token_remain_count = effective_mask.sum(dim=1)
        token_remain_rate = token_remain_count / completion_count.clamp_min(1.0)
        metric_states["loss/token_remain_rate"] = _mean_state_from_samples(token_remain_rate, has_completion)
        metric_states["loss/token_mask_rate"] = _mean_state_from_samples(1.0 - token_remain_rate, has_completion)

        return DDISLossOutputs(loss_unscaled=loss_unscaled, metric_states=metric_states)
