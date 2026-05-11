# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MXFP4 MoE with dynamic dequantization — matmul_ogs API version.

Supports two dequant modes (configurable via dequant_dtype):
- "bf16": MXFP4 → BF16 dequant, BF16 matmul (default)
- "fp8":  MXFP4 → FP8 E5M2 dequant, FP8 activation with per-token scale

Uses matmul_ogs (GatherIndx/ScatterIndx/RoutingData) instead of raw matmul.
Key advantages:
- No _convert_routing() — routing objects passed through directly
- matmul_ogs scatter handles topk reduction automatically — no manual .view().sum()
- Cleaner interface: GatherIndx/ScatterIndx dataclasses instead of raw int tensors
"""

from transformers.utils import is_torch_available, logging

if is_torch_available():
    import torch
    from torch import nn
from contextlib import contextmanager

from transformers.core_model_loading import ConversionOps
from transformers.quantizers.quantizers_utils import get_module_from_name, should_convert_module

# ---------------------------------------------------------------------------
# Direct imports from triton_kernels (matmul_ogs API)
# ---------------------------------------------------------------------------
from triton_kernels.matmul_ogs import matmul_ogs, PrecisionConfig, FlexCtx
from triton_kernels.numerics import InFlexData
from triton_kernels.routing import (
    GatherIndx, ScatterIndx, RoutingData,
    routing as triton_routing,
    compute_expt_data_torch,
)

# ---------------------------------------------------------------------------
# Local Triton kernels (dequant, quantize, SwiGLU)
# ---------------------------------------------------------------------------
from .kernels import (
    mxfp4_to_bf16_triton,
    mxfp4_to_fp8_e5m2_triton,
    mxfp4_to_fp8_e5m2_colmajor,
    dynamic_quantize_per_token,
    _swiglu_forward_triton,
    _swiglu_backward,
)

logger = logging.get_logger(__name__)


@contextmanager
def on_device(dev):
    if is_torch_available():
        import torch
        if isinstance(dev, torch.Tensor):
            dev = dev.device
        elif isinstance(dev, str):
            dev = torch.device(dev)
        dev_type = getattr(dev, "type", None)
        if dev_type == "cuda":
            with torch.cuda.device(dev):
                yield
                return
        if dev_type == "xpu" and hasattr(torch, "xpu"):
            with torch.xpu.device(dev):
                yield
                return
    yield


# ---------------------------------------------------------------------------
# MXFP4 → BF16 dequantization (LUT + ldexp, based on transformers.integrations.mxfp4)
# ---------------------------------------------------------------------------

def mxfp4_to_bf16(blocks, scales):
    """Dequant MXFP4 to BF16 — delegates to optimized Triton kernel.

    Args:
        blocks: [E, N, G, 16] uint8 — packed FP4 (2 E2M1 per byte)
        scales: [E, N, G] uint8 — E8M0 block scales

    Returns:
        bf16_tensor: [E, N, G*32] bfloat16
    """
    return mxfp4_to_bf16_triton(blocks, scales)


def _mxfp4_to_bf16_pytorch(blocks, scales):
    """Reference: MXFP4→BF16 via PyTorch LUT + ldexp (slow, may produce NaN under FSDP).

    Kept for correctness testing only. Use mxfp4_to_bf16_triton for production.
    """
    lut = torch.tensor(FP4_VALUES, dtype=torch.bfloat16, device=blocks.device)
    prefix_shape = blocks.shape[:-2]
    G, B = blocks.shape[-2], blocks.shape[-1]  # B=16

    blk = blocks.to(torch.uint8)
    exp = (scales.to(torch.int32) - 127).unsqueeze(-1)  # [E, N, G, 1]

    idx_lo = (blk & 0x0F).to(torch.long)
    idx_hi = (blk >> 4).to(torch.long)
    out = torch.empty(*prefix_shape, G, B * 2, dtype=torch.bfloat16, device=blocks.device)
    out[..., 0::2] = lut[idx_lo]
    out[..., 1::2] = lut[idx_hi]
    del idx_lo, idx_hi

    torch.ldexp(out, exp.expand_as(out), out=out)
    return out.reshape(*prefix_shape, G * B * 2)




# ---------------------------------------------------------------------------
# ConversionOps for transformers weight loading
# ---------------------------------------------------------------------------

FP4_VALUES = [
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def _convert_moe_packed_tensors(blocks, scales, *, dtype=torch.bfloat16, rows_per_chunk=32768 * 1024):
    """Dequantize MXFP4 to bf16 (for dequantize fallback)."""
    import math
    blocks = blocks.to(torch.uint8)
    scales = scales.to(torch.int32) - 127
    assert blocks.shape[:-1] == scales.shape
    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)
    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G
    blocks = blocks.reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1)
    out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)
    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)
        blk = blocks[r0:r1]
        exp = scales[r0:r1]
        sub = out[r0:r1]
        idx_lo = (blk & 0x0F).to(torch.int)
        sub[:, 0::2] = lut[idx_lo]
        del idx_lo
        idx_hi = (blk >> 4).to(torch.int)
        sub[:, 1::2] = lut[idx_hi]
        del idx_hi
        torch.ldexp(sub, exp, out=sub)
        del blk, exp, sub
    out = out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)
    return out.transpose(1, 2).contiguous()


def convert_moe_packed_tensors(blocks, scales, *, dtype=torch.bfloat16, rows_per_chunk=32768 * 1024):
    try:
        return _convert_moe_packed_tensors(blocks, scales, dtype=dtype, rows_per_chunk=rows_per_chunk)
    except torch.OutOfMemoryError:
        blocks = blocks.to("cpu")
        scales = scales.to("cpu")
        return _convert_moe_packed_tensors(blocks, scales, dtype=dtype, rows_per_chunk=rows_per_chunk)


class Mxfp4Fp8DequantQuantize(ConversionOps):
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict, model=None, missing_keys=None, full_layer_name=None, **kwargs):
        return {}


class Mxfp4Fp8DequantDequantize(ConversionOps):
    """Dequantize MXFP4 to bf16 (fallback path)."""
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict, model=None, full_layer_name=None, missing_keys=None, **kwargs):
        if "_blocks" in input_dict.keys():
            blocks = input_dict["_blocks"][0] if isinstance(input_dict["_blocks"], list) else input_dict["_blocks"]
        if "_scales" in input_dict.keys():
            scales = input_dict["_scales"][0] if isinstance(input_dict["_scales"], list) else input_dict["_scales"]
        dequantized = convert_moe_packed_tensors(blocks, scales)
        return {full_layer_name: dequantized}


class Mxfp4Fp8DequantDeserialize(ConversionOps):
    """Load MXFP4 checkpoint weights and store as-is (no conversion)."""

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(self, input_dict, model=None, full_layer_name=None, missing_keys=None, **kwargs):
        param_data = {}
        if "_blocks" in input_dict.keys():
            param_data["_blocks"] = input_dict["_blocks"][0] if isinstance(input_dict["_blocks"], list) else input_dict["_blocks"]
        if "_scales" in input_dict.keys():
            param_data["_scales"] = input_dict["_scales"][0] if isinstance(input_dict["_scales"], list) else input_dict["_scales"]

        module, _ = get_module_from_name(model, full_layer_name)
        proj = "gate_up_proj" if "gate_up_proj" in full_layer_name else "down_proj"
        _store_mxfp4_weights(
            param_data["_blocks"],
            param_data["_scales"],
            module,
            proj,
            param_data["_blocks"].device,
        )
        missing_keys.discard(f"{full_layer_name}")
        module._is_hf_initialized = True
        return {}


# ---------------------------------------------------------------------------
# FSDP-compatible MXFP4 weight packing
# ---------------------------------------------------------------------------
# FSDP requires parameters to be floating-point. We pack uint8 bytes into
# bfloat16 containers (2 bytes per bf16 element) so FSDP can shard them.
# The bit patterns are preserved through FSDP's all-gather/reshard.
# ---------------------------------------------------------------------------

def _pack_uint8_as_bf16(tensor):
    """Reinterpret uint8 tensor as bfloat16 (2 uint8 → 1 bf16 element)."""
    flat = tensor.contiguous().reshape(-1)
    if flat.numel() % 2:
        flat = torch.cat([flat, flat.new_zeros(1)])
    return flat.view(torch.bfloat16)


def _unpack_bf16_as_uint8(packed, shape):
    """Recover uint8 tensor from bfloat16 packed representation."""
    import math
    numel = math.prod(shape)
    return packed.reshape(-1).view(torch.uint8)[:numel].reshape(shape)


# ---------------------------------------------------------------------------
# MXFP4 weight storage — packed as bf16 Parameters for FSDP compatibility
# ---------------------------------------------------------------------------

def _store_mxfp4_weights(blocks, scales, module, proj, target_device):
    """Store MXFP4 blocks+scales as bf16 Parameters (FSDP-compatible).

    Weights stay in MXFP4 format, packed into bfloat16 containers so FSDP
    can shard/all-gather them. Dequantized to BF16/FP8 on-the-fly in forward.

    Args:
        blocks: [n_experts, N, K//32, 16] uint8 — packed FP4
        scales: [n_experts, N, K//32] uint8 — E8M0 block scales
        module: target nn.Module
        proj: "gate_up_proj" or "down_proj"
        target_device: device to store on
    """
    if getattr(target_device, "type", target_device) == "cpu":
        target_device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"

    blocks = blocks.to(target_device).contiguous()
    scales = scales.to(target_device).contiguous()

    blocks_attr = f"{proj}_blocks"
    scales_attr = f"{proj}_scales"

    # Remove old nn.Parameter placeholders if they exist
    for attr in [blocks_attr, scales_attr, proj]:
        if attr in module._parameters:
            del module._parameters[attr]

    # Store original shapes for unpacking in forward
    setattr(module, f"_{proj}_shapes", (tuple(blocks.shape), tuple(scales.shape)))

    # Pack as bfloat16 Parameters so FSDP can manage them
    module.register_parameter(
        blocks_attr, nn.Parameter(_pack_uint8_as_bf16(blocks), requires_grad=False)
    )
    module.register_parameter(
        scales_attr, nn.Parameter(_pack_uint8_as_bf16(scales), requires_grad=False)
    )


# ---------------------------------------------------------------------------
# Autograd Functions: composable dequant-matmul + SwiGLU (BNB MatMul4Bit style)
# ---------------------------------------------------------------------------

class DequantMoEMatmul(torch.autograd.Function):
    """MXFP4 dequant + matmul_ogs.

    Weights are NOT saved for backward — instead, the module reference and
    projection name are stored so backward can re-derive weights from the
    (re-all-gathered) FSDP parameters.  This avoids cloning the full weight
    tensors, which would negate FSDP's memory savings.

    Supports dequant modes: "bf16" (BF16 matmul) or "fp8" (FP8 E5M2 + per-token scale).
    """

    @staticmethod
    def forward(
        ctx,
        x,                         # activation input
        blocks,                    # [E, N, G, 16] uint8 — MXFP4 packed weights
        scales,                    # [E, N, G] uint8 — E8M0 block scales
        bias,                      # [E, N] bfloat16 or None
        routing_data,              # RoutingData object
        fwd_gather_indx,           # GatherIndx for forward (or None)
        fwd_scatter_indx,          # ScatterIndx for forward (or None)
        fwd_gammas,                # gammas for forward (or None)
        bwd_gather_indx,           # GatherIndx for backward (or None)
        bwd_scatter_indx,          # ScatterIndx for backward (or None)
        bwd_gammas,                # gammas for backward (or None)
        dequant_dtype="bf16",      # "bf16" or "fp8"
        n_expts_act=1,             # top_k for gather scale reorder
        experts_module=None,       # Mxfp4Bf16DequantGptOssExperts instance (for backward re-derive)
        proj_name=None,            # "gate_up_proj" or "down_proj"
    ):
        with on_device(x.device):
            matmul_kwargs = dict(
                routing_data=routing_data,
                gather_indx=fwd_gather_indx,
                scatter_indx=fwd_scatter_indx,
                gammas=fwd_gammas,
            )

            if dequant_dtype == "fp8":
                # ① MXFP4 → FP8 E5M2 (row-major)
                w_fp8 = mxfp4_to_fp8_e5m2_triton(blocks, scales)
                w_fp8_t = w_fp8.transpose(-1, -2)

                # ② Activation → FP8 E4M3FN (per-token scale)
                x_fp8, per_token_scale = dynamic_quantize_per_token(x)

                # ③ Per-token scale reorder for gather
                if fwd_gather_indx is not None:
                    act_scale = per_token_scale[fwd_gather_indx.src_indx // n_expts_act]
                else:
                    act_scale = per_token_scale

                # ④ PrecisionConfig with per-token scale
                matmul_kwargs["precision_config"] = PrecisionConfig(
                    flex_ctx=FlexCtx(
                        lhs_data=InFlexData(scale=act_scale, scale_mode="per_token"),
                        rhs_data=InFlexData(),
                    ),
                    out_dtype=torch.bfloat16,
                )

                # ⑤ FP8 matmul_ogs
                out = matmul_ogs(x_fp8, w_fp8_t, bias, **matmul_kwargs)
                del w_fp8, w_fp8_t
            else:
                # BF16 path
                w_bf16 = mxfp4_to_bf16(blocks, scales)
                out = matmul_ogs(x, w_bf16.transpose(-1, -2), bias, **matmul_kwargs)
                del w_bf16

        # Don't save blocks/scales — re-derive in backward from module params.
        # This avoids cloning full weights, which would negate FSDP memory savings.
        ctx.experts_module = experts_module
        ctx.proj_name = proj_name
        ctx.routing_data = routing_data
        ctx.bwd_gather_indx = bwd_gather_indx
        ctx.bwd_scatter_indx = bwd_scatter_indx
        ctx.bwd_gammas = bwd_gammas
        ctx.dequant_dtype = dequant_dtype
        ctx.n_expts_act = n_expts_act
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # Re-derive weights from module (FSDP has re-all-gathered params for backward)
        blocks, scales = ctx.experts_module._get_uint8_weights(ctx.proj_name)

        with on_device(grad_output.device):
            if ctx.dequant_dtype == "fp8":
                w_fp8 = mxfp4_to_fp8_e5m2_colmajor(blocks, scales)
                w_fp8 = w_fp8.transpose(-1, -2)

                grad_fp8, grad_scale = dynamic_quantize_per_token(grad_output)

                if ctx.bwd_gather_indx is not None:
                    act_scale = grad_scale[ctx.bwd_gather_indx.src_indx // ctx.n_expts_act]
                else:
                    act_scale = grad_scale

                pc = PrecisionConfig(
                    flex_ctx=FlexCtx(
                        lhs_data=InFlexData(scale=act_scale, scale_mode="per_token"),
                        rhs_data=InFlexData(),
                    ),
                    out_dtype=torch.bfloat16,
                )
                grad_x = matmul_ogs(
                    grad_fp8, w_fp8, bias=None,
                    routing_data=ctx.routing_data,
                    gather_indx=ctx.bwd_gather_indx,
                    scatter_indx=ctx.bwd_scatter_indx,
                    gammas=ctx.bwd_gammas,
                    precision_config=pc,
                )
                del w_fp8
            else:
                w_bf16 = mxfp4_to_bf16(blocks, scales)
                grad_x = matmul_ogs(
                    grad_output, w_bf16, bias=None,
                    routing_data=ctx.routing_data,
                    gather_indx=ctx.bwd_gather_indx,
                    scatter_indx=ctx.bwd_scatter_indx,
                    gammas=ctx.bwd_gammas,
                )
                del w_bf16

        # 15 inputs (excl ctx) → 15 gradient returns
        return grad_x, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class SwiGLUFunction(torch.autograd.Function):
    """Fused SwiGLU: split interleaved gate/up, clamp, activate.

    Forward uses Triton fused kernel (handles >2GB tensors via int64 offsets).
    Backward uses unfused PyTorch ops (not on inference critical path).
    Saved tensors = gate + up (mathematical minimum for SwiGLU backward).
    """

    @staticmethod
    def forward(ctx, gate_up_raw, alpha, limit):
        out, gate, up = _swiglu_forward_triton(gate_up_raw, alpha, limit, save_for_bwd=True)
        del gate_up_raw
        ctx.save_for_backward(gate, up)
        ctx.alpha = alpha
        ctx.limit = limit
        return out

    @staticmethod
    def backward(ctx, grad_output):
        gate, up = ctx.saved_tensors
        return _swiglu_backward(grad_output, gate, up, ctx.alpha, ctx.limit), None, None


# ---------------------------------------------------------------------------
# BF16 MoE Expert Module with Dynamic Dequantization
# ---------------------------------------------------------------------------

class Mxfp4Bf16DequantGptOssExperts(nn.Module):
    """MoE expert module with dynamic MXFP4 dequantization.

    Supports two modes:
    - dequant_dtype="bf16": MXFP4 → BF16 dequant (default)
    - dequant_dtype="fp8":  MXFP4 → FP8 E5M2 + per-token activation scale

    Uses matmul_ogs API with GatherIndx/ScatterIndx for MoE routing.
    Scatter automatically handles topk reduction.
    """

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size

        # Placeholder parameters — will be replaced by uint8 tensors during loading
        self.gate_up_proj = nn.Parameter(
            torch.zeros(self.num_experts, 2 * self.intermediate_size, self.hidden_size // 32, 16, dtype=torch.uint8),
            requires_grad=False,
        )
        self.gate_up_proj_bias = nn.Parameter(
            torch.zeros(self.num_experts, 2 * self.intermediate_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.down_proj = nn.Parameter(
            torch.zeros(self.num_experts, self.hidden_size, self.intermediate_size // 32, 16, dtype=torch.uint8),
            requires_grad=False,
        )
        self.down_proj_bias = nn.Parameter(
            torch.zeros(self.num_experts, self.hidden_size, dtype=torch.bfloat16),
            requires_grad=False,
        )
        self.alpha = 1.702
        self.limit = getattr(config, "swiglu_limit", 7.0)
        self.dequant_dtype = getattr(config, "dequant_dtype", "bf16")

    def _get_uint8_weights(self, proj):
        """Unpack bf16 Parameters back to uint8 blocks and scales.

        Returns views (no clone) — caller must use them before FSDP reshards.
        Safe because both forward and backward run within FSDP's all-gather context.
        """
        shapes = getattr(self, f"_{proj}_shapes", None)
        blocks_param = getattr(self, f"{proj}_blocks")
        scales_param = getattr(self, f"{proj}_scales")
        if shapes is not None:
            blocks_shape, scales_shape = shapes
            blocks = _unpack_bf16_as_uint8(blocks_param, blocks_shape)
            scales = _unpack_bf16_as_uint8(scales_param, scales_shape)
        else:
            blocks = blocks_param
            scales = scales_param
        return blocks, scales

    def forward(self, hidden_states: torch.Tensor, routing_data, gather_idx, scatter_idx) -> torch.Tensor:
        # matmul_ogs accepts routing objects directly — no conversion needed
        n_expts_act = routing_data.n_expts_act

        # Unpack MXFP4 weights from bf16 FSDP-compatible storage
        gu_blocks, gu_scales = self._get_uint8_weights("gate_up_proj")
        dn_blocks, dn_scales = self._get_uint8_weights("down_proj")

        # ① Gate-up projection (dequant + gather matmul)
        gate_up_raw = DequantMoEMatmul.apply(
            hidden_states,
            gu_blocks, gu_scales,
            self.gate_up_proj_bias.float(),
            routing_data,
            gather_idx, None, None,                    # fwd: gather
            None, scatter_idx, None,                   # bwd: scatter (auto-reduce)
            self.dequant_dtype, n_expts_act,           # dequant mode
            self, "gate_up_proj",                      # module ref for backward re-derive
        )

        # ② SwiGLU activation
        intermediate = SwiGLUFunction.apply(gate_up_raw, self.alpha, self.limit)

        # ③ Down projection (dequant + scatter matmul)
        #    scatter + gammas in forward → gather + gammas in backward
        output = DequantMoEMatmul.apply(
            intermediate,
            dn_blocks, dn_scales,
            self.down_proj_bias.float(),
            routing_data,
            None, scatter_idx, routing_data.gate_scal,  # fwd: scatter + gammas
            gather_idx, None, routing_data.gate_scal,   # bwd: gather + gammas
            self.dequant_dtype, n_expts_act,             # dequant mode
            self, "down_proj",                           # module ref for backward re-derive
        )

        # matmul_ogs scatter handles topk reduction — no manual .view().sum()
        return output


# ---------------------------------------------------------------------------
# Routing (same as mxfp4_fp8)
# ---------------------------------------------------------------------------

def routing_torch_dist(logits, n_expts_act):
    import os
    with on_device(logits.device):
        world_size = torch.distributed.get_world_size()
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        replace_value = -1
        n_tokens = logits.shape[0]
        n_expts_tot = logits.shape[1]
        n_local_experts = n_expts_tot // world_size
        local_expert_start = rank * n_local_experts
        local_expert_end = (rank + 1) * n_local_experts
        n_gates_pad = n_tokens * n_expts_act

        def topk(vals, k):
            tk_indx = torch.argsort(-vals, dim=1, stable=True)[:, :k]
            tk_indx = tk_indx.long()
            tk_val = torch.take_along_dim(vals, tk_indx, dim=1)
            return tk_val, tk_indx.int()

        expt_scal, expt_indx = topk(logits, n_expts_act)
        expt_scal = torch.softmax(expt_scal, dim=-1)
        expt_indx, sort_indices = torch.sort(expt_indx, dim=1)
        expt_scal = torch.gather(expt_scal, 1, sort_indices)
        expt_scal = expt_scal.reshape(-1)
        hist = torch.histc(expt_indx, bins=n_expts_tot, max=n_expts_tot - 1)[local_expert_start:local_expert_end]
        expt_indx = expt_indx.view(-1).to(torch.int32)
        var = 1000
        expt_indx = torch.where(expt_indx < local_expert_start, var, expt_indx)
        topk_indx = torch.argsort(expt_indx, stable=True).to(torch.int32)
        gate_indx = torch.argsort(topk_indx).to(torch.int32)
        expt_indx = torch.where(expt_indx < local_expert_end, expt_indx, replace_value)
        expt_indx = torch.where(local_expert_start <= expt_indx, expt_indx, replace_value)
        gate_indx = torch.where(expt_indx == replace_value, replace_value, gate_indx)
        gate_scal = expt_scal[topk_indx]
        topk_indx = torch.where(gate_indx[topk_indx] == replace_value, replace_value, topk_indx)
        gather_indx = GatherIndx(src_indx=topk_indx.int(), dst_indx=gate_indx.int())
        scatter_indx = ScatterIndx(src_indx=gate_indx.int(), dst_indx=topk_indx.int())
        expt_data = compute_expt_data_torch(hist, n_local_experts, n_gates_pad)
        hit_experts = n_expts_act
    return RoutingData(gate_scal, hist, n_local_experts, hit_experts, expt_data), gather_indx, scatter_indx


def routing_replay(logits, expt_indx, n_expts_act):
    """Build RoutingData from pre-determined expert indices (for routing replay).

    Instead of running top-k, uses the provided expert assignments to construct
    the same gather/scatter/routing structures that triton_routing produces.

    Args:
        logits: [T, E] router logits (used for softmax over selected experts)
        expt_indx: [T, K] int32 expert indices per token
        n_expts_act: K (top-k)

    Returns:
        (RoutingData, GatherIndx, ScatterIndx) — same format as triton_routing
    """
    T, E = logits.shape
    device = logits.device

    expt_indx_long = expt_indx.to(torch.long)

    # Softmax only over the K selected experts (not all E)
    logits_topk = torch.take_along_dim(logits, expt_indx_long, dim=-1)  # [T, K]
    expt_scal = torch.softmax(logits_topk, dim=-1).reshape(-1)  # [T*K]
    expt_indx_flat = expt_indx_long.reshape(-1)  # [T*K]

    # Sort by expert id
    sorted_expts, topk_indx = torch.sort(expt_indx_flat, stable=True)  # [T*K]

    # Inverse permutation (avoid argsort)
    N = topk_indx.numel()
    gate_indx = torch.empty_like(topk_indx)
    gate_indx[topk_indx] = torch.arange(N, device=device, dtype=topk_indx.dtype)

    gate_scal = expt_scal[topk_indx]
    hist = torch.bincount(sorted_expts, minlength=E).to(torch.int32)
    expt_data = compute_expt_data_torch(hist, E, N)

    topk_i32 = topk_indx.to(torch.int32)
    gate_i32 = gate_indx.to(torch.int32)

    return (
        RoutingData(gate_scal, hist, E, n_expts_act, expt_data),
        GatherIndx(src_indx=topk_i32, dst_indx=gate_i32),
        ScatterIndx(src_indx=gate_i32, dst_indx=topk_i32),
    )


def mlp_forward(self, hidden_states, router_indices=None):
    """MoE MLP forward with optional routing replay.

    Args:
        hidden_states: [B, S, H]
        router_indices: [B, S, K] int — pre-determined expert indices for replay.
            If None, uses normal top-k routing.
    """
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized() and hasattr(self, "_is_hooked"):
        routing = routing_torch_dist
    else:
        routing = triton_routing
    batch_size = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(-1, self.router.hidden_dim)
    router_logits = nn.functional.linear(hidden_states, self.router.weight, self.router.bias)
    with on_device(router_logits.device):
        if router_indices is not None:
            expt_indx = router_indices.reshape(-1, self.router.top_k).to(
                device=router_logits.device, dtype=torch.int32,
            )
            routing_data, gather_idx, scatter_idx = routing_replay(
                router_logits, expt_indx, self.router.top_k,
            )
        else:
            routing_data, gather_idx, scatter_idx = routing(router_logits, self.router.top_k)
    routed_out = self.experts(hidden_states, routing_data, gather_idx, scatter_idx=scatter_idx)
    routed_out = routed_out.reshape(batch_size, -1, self.router.hidden_dim)

    return routed_out, router_logits


# ---------------------------------------------------------------------------
# Module replacement
# ---------------------------------------------------------------------------

def replace_with_dequant_experts(model, quantization_config=None, modules_to_not_convert=None):
    """Replace GptOssExperts with Mxfp4Bf16DequantGptOssExperts."""
    if quantization_config.dequantize:
        return model

    # Pass config to model.config so Expert module can read it
    dequant_dtype = getattr(quantization_config, "dequant_dtype", "bf16")
    model.config.dequant_dtype = dequant_dtype

    has_been_replaced = False
    for module_name, module in model.named_modules():
        if not should_convert_module(module_name, modules_to_not_convert):
            continue
        if module.__class__.__name__ == "GptOssExperts" and not quantization_config.dequantize:
            with torch.device("meta"):
                model.set_submodule(module_name, Mxfp4Bf16DequantGptOssExperts(model.config))
                has_been_replaced = True
        if module.__class__.__name__ == "GptOssMLP" and not quantization_config.dequantize:
            from types import MethodType
            module.forward = MethodType(mlp_forward, module)

    if not has_been_replaced:
        logger.warning("No expert modules found for FP8 dequant conversion.")

    return model
