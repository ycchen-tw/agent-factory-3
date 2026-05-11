"""Patch torch._foreach_copy_ to preserve bf16 NaN bit patterns.

torch._foreach_copy_ canonicalizes bf16 NaN to 0x7FFF during same-dtype copy.
This corrupts packed uint8 data stored in bf16 containers (used by FSDP2).

Fix: for bf16 same-dtype copies, fall back to individual copy_() which preserves bits.

Usage: import this module before any FSDP2 operations.

    import patch_fsdp2_foreach_copy  # noqa: F401
"""
import torch

_orig_foreach_copy_ = torch._foreach_copy_


def _safe_foreach_copy_(dsts, srcs):
    # Only need special handling for bf16 same-dtype copy (where NaN gets canonicalized)
    needs_safe = any(
        d.dtype == torch.bfloat16 and s.dtype == torch.bfloat16
        for d, s in zip(dsts, srcs)
    )
    if needs_safe:
        for d, s in zip(dsts, srcs):
            d.copy_(s)
    else:
        _orig_foreach_copy_(dsts, srcs)


torch._foreach_copy_ = _safe_foreach_copy_
