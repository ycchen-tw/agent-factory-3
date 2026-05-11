"""
Local Triton kernels for MXFP4 dequantization, FP8 quantization, and SwiGLU.

Optimized kernels with:
- uint16/uint32 packed stores to avoid write amplification
- int64 pointer offsets for >2GB tensors
- Fast approximate exp via PTX ex2.approx.ftz.f32
- Column-major dequant output for Hopper FP8 backward
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# MXFP4 → FP8 E5M2 dequantization (Triton bitwise conversion)
# ---------------------------------------------------------------------------

@triton.jit
def _convert_nibble_to_fp8(nibble, scale_adj):
    """Convert E2M1 nibble to FP8 E5M2 with broadcasted scale_adj.

    Works with any shape — element-wise on int32 tensors.
    scale_adj = E8M0_scale - 113 (bias correction: E2M1_bias + E8M0_bias - E5M2_bias).
    """
    sign = (nibble >> 3) & 1
    ee = (nibble >> 1) & 3
    m = nibble & 1
    n_abs = nibble & 7
    exp = ee + scale_adj
    mant = tl.where(ee > 0, m << 1, tl.zeros_like(m))
    result = (sign << 7) | (exp << 2) | mant
    result = tl.where(n_abs == 0, sign << 7, result)
    result = tl.where((n_abs > 0) & (exp < 1), sign << 7, result)
    return result


# --- Row-major output (for forward) ---

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_G": 256}, num_warps=4),
        triton.Config({"BLOCK_G": 256}, num_warps=8),
        triton.Config({"BLOCK_G": 512}, num_warps=4),
        triton.Config({"BLOCK_G": 512}, num_warps=8),
        triton.Config({"BLOCK_G": 1024}, num_warps=8),
    ],
    key=["n_groups"],
)
@triton.jit
def _mxfp4_to_fp8_e5m2_kernel(
    blocks_ptr,   # [n_groups * 16] uint8 — packed FP4
    scales_ptr,   # [n_groups] uint8 — E8M0 scales
    out_ptr,      # [n_groups * 16] uint16 — packed FP8 E5M2 pairs
    n_groups,
    BLOCK_G: tl.constexpr,
):
    """MXFP4→FP8 E5M2 kernel with packed uint16 stores.

    Loads [BLOCK_G, 16] bytes, converts each byte's lo/hi nibbles to
    two FP8 values, packs them into uint16, and stores [BLOCK_G, 16]
    contiguous uint16 (= 32 output bytes per group).
    """
    pid = tl.program_id(0)
    g_offs = pid * BLOCK_G + tl.arange(0, BLOCK_G)
    g_mask = g_offs < n_groups
    byte_offs = tl.arange(0, 16)

    scale_adj = tl.load(scales_ptr + g_offs, mask=g_mask, other=0).to(tl.int32) - 113

    block_addrs = g_offs[:, None] * 16 + byte_offs[None, :]
    packed = tl.load(blocks_ptr + block_addrs, mask=g_mask[:, None], other=0).to(tl.int32)

    lo_result = _convert_nibble_to_fp8(packed & 0x0F, scale_adj[:, None])
    hi_result = _convert_nibble_to_fp8(packed >> 4, scale_adj[:, None])

    combined = lo_result | (hi_result << 8)
    out_addrs = g_offs[:, None] * 16 + byte_offs[None, :]
    tl.store(out_ptr + out_addrs, combined.to(tl.uint16), mask=g_mask[:, None])


def mxfp4_to_fp8_e5m2_triton(blocks, scales):
    """Convert MXFP4 packed weights to FP8 E5M2 (row-major output).

    Args:
        blocks: [..., G, 16] uint8 — packed FP4 (2 E2M1 values per byte)
        scales: [..., G] uint8 — E8M0 block scales

    Returns:
        fp8_tensor: [..., G*32] float8_e5m2
    """
    prefix_shape = blocks.shape[:-2]
    G = blocks.shape[-2]
    n_groups = blocks.numel() // 16

    blocks_flat = blocks.reshape(n_groups, 16).contiguous()
    scales_flat = scales.reshape(n_groups).contiguous()
    out_u16 = torch.empty(n_groups, 16, dtype=torch.uint16, device=blocks.device)

    grid = lambda meta: ((n_groups + meta["BLOCK_G"] - 1) // meta["BLOCK_G"],)
    _mxfp4_to_fp8_e5m2_kernel[grid](
        blocks_flat, scales_flat, out_u16,
        n_groups,
    )

    out_flat = out_u16.view(torch.uint8).reshape(n_groups, 32)
    out = out_flat.reshape(*prefix_shape, G * 32)
    return out.view(torch.float8_e5m2)


# ---------------------------------------------------------------------------
# MXFP4 → BF16 dequantization (Triton bitwise conversion)
# ---------------------------------------------------------------------------
# Direct integer bit manipulation: E2M1 nibble + E8M0 scale → BF16 bits.
# No floating-point intermediate ops — avoids NaN propagation from packed
# bf16 FSDP parameters. Uses packed uint32 stores (2 bf16 per uint32).
# ---------------------------------------------------------------------------

@triton.jit
def _convert_nibble_to_bf16(nibble, scale):
    """Convert E2M1 nibble to BF16 bit pattern using E8M0 scale (pure integer ops).

    Math:
      Normal E2M1 (ee>0): value = 2^(ee-1) * (1+m/2) * 2^(S-127)
        → BF16 exp = ee-1+S, mantissa bit6 = m  (BF16 bias 127 cancels E8M0 bias)
      Subnormal E2M1 (ee=0, m=1): value = 0.5 * 2^(S-127) = 2^(S-128)
        → BF16 exp = S-1, mantissa = 0

    Args:
        nibble: int32 — E2M1 packed (4 bits: 1 sign + 2 exp + 1 mantissa)
        scale: int32 — raw E8M0 byte value [0..255]

    Returns:
        int32 with lower 16 bits = BF16 bit pattern
    """
    sign = (nibble >> 3) & 1
    ee = (nibble >> 1) & 3
    m = nibble & 1
    n_abs = nibble & 7

    # BF16 exponent (before overflow/underflow check)
    exp = tl.where(ee > 0, ee - 1 + scale, scale - 1)

    # BF16 mantissa: only bit 6 used, only for normal E2M1
    mant = tl.where(ee > 0, m << 6, tl.zeros_like(m))

    # Assemble BF16 bits
    result = (sign << 15) | (exp << 7) | mant

    # ±zero (n_abs == 0)
    result = tl.where(n_abs == 0, sign << 15, result)
    # Overflow → ±zero (BF16 exp 255 = inf/NaN)
    result = tl.where((n_abs > 0) & (exp > 254), sign << 15, result)
    # Underflow → ±zero (flush denormals)
    result = tl.where((n_abs > 0) & (exp < 1), sign << 15, result)

    return result


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_G": 256}, num_warps=4),
        triton.Config({"BLOCK_G": 256}, num_warps=8),
        triton.Config({"BLOCK_G": 512}, num_warps=4),
        triton.Config({"BLOCK_G": 512}, num_warps=8),
        triton.Config({"BLOCK_G": 1024}, num_warps=8),
    ],
    key=["n_groups"],
)
@triton.jit
def _mxfp4_to_bf16_kernel(
    blocks_ptr,   # [n_groups * 16] uint8 — packed FP4
    scales_ptr,   # [n_groups] uint8 — E8M0 scales
    out_ptr,      # [n_groups * 16] uint32 — packed BF16 pairs
    n_groups,
    BLOCK_G: tl.constexpr,
):
    """MXFP4→BF16 kernel with packed uint32 stores.

    Each uint8 byte → 2 BF16 values (lo/hi nibbles).
    Pack (lo_bf16, hi_bf16) as uint32 → 16 contiguous uint32 per group.
    """
    pid = tl.program_id(0)
    g_offs = pid * BLOCK_G + tl.arange(0, BLOCK_G)
    g_mask = g_offs < n_groups
    byte_offs = tl.arange(0, 16)

    scale = tl.load(scales_ptr + g_offs, mask=g_mask, other=0).to(tl.int32)

    block_addrs = g_offs[:, None] * 16 + byte_offs[None, :]
    packed = tl.load(blocks_ptr + block_addrs, mask=g_mask[:, None], other=0).to(tl.int32)

    lo_bf16 = _convert_nibble_to_bf16(packed & 0x0F, scale[:, None])
    hi_bf16 = _convert_nibble_to_bf16(packed >> 4, scale[:, None])

    # Pack (lo, hi) bf16 pair as uint32: lo in low 16 bits, hi in high 16 bits
    # In memory (little-endian): [lo_byte0, lo_byte1, hi_byte0, hi_byte1]
    # Viewed as bf16: [lo_value, hi_value] — matches interleaved layout
    combined = (lo_bf16 & 0xFFFF).to(tl.uint32) | ((hi_bf16 & 0xFFFF).to(tl.uint32) << 16)

    out_addrs = g_offs[:, None] * 16 + byte_offs[None, :]
    tl.store(out_ptr + out_addrs, combined, mask=g_mask[:, None])


def mxfp4_to_bf16_triton(blocks, scales):
    """Convert MXFP4 packed weights to BF16 (optimized Triton kernel).

    Pure integer bitwise conversion — no floating-point intermediate ops.
    Avoids NaN propagation issues with torch.ldexp on FSDP-packed parameters.

    Args:
        blocks: [..., G, 16] uint8 — packed FP4 (2 E2M1 values per byte)
        scales: [..., G] uint8 — E8M0 block scales

    Returns:
        bf16_tensor: [..., G*32] bfloat16
    """
    prefix_shape = blocks.shape[:-2]
    G = blocks.shape[-2]
    n_groups = blocks.numel() // 16

    blocks_flat = blocks.reshape(n_groups, 16).contiguous()
    scales_flat = scales.reshape(n_groups).contiguous()
    out_u32 = torch.empty(n_groups, 16, dtype=torch.uint32, device=blocks.device)

    grid = lambda meta: ((n_groups + meta["BLOCK_G"] - 1) // meta["BLOCK_G"],)
    _mxfp4_to_bf16_kernel[grid](
        blocks_flat, scales_flat, out_u32,
        n_groups,
    )

    # Reinterpret uint32 → bfloat16: each uint32 contains 2 bf16 values
    out = out_u32.view(torch.bfloat16).reshape(n_groups, 32)
    return out.reshape(*prefix_shape, G * 32)


# --- Column-major output (for backward on Hopper) ---

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 64}, num_warps=4),
        triton.Config({"BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_N": 128}, num_warps=8),
        triton.Config({"BLOCK_N": 256}, num_warps=8),
    ],
    key=["N", "G"],
)
@triton.jit
def _mxfp4_to_fp8_colmajor_kernel(
    blocks_ptr,   # [E, N, G, 16] uint8
    scales_ptr,   # [E, N, G] uint8
    out_ptr,      # [E, K, N] uint8 — column-major (N contiguous)
    N,            # rows per expert
    G,            # groups per row (K = G * 32)
    NG16,         # N * G * 16 — expert stride for blocks
    NG,           # N * G — expert stride for scales
    KN,           # K * N — expert stride for output (may need int64)
    BLOCK_N: tl.constexpr,
):
    """Fused MXFP4 → FP8 E5M2 dequant with column-major output.

    Grid: (cdiv(N, BLOCK_N), G, E)
    Each program handles BLOCK_N rows for one group of one expert.

    Read: per-byte loop, L2 cache absorbs repeated access to same group.
    Write: BLOCK_N contiguous uint8 stores along N → fully coalesced.
    """
    pid_n = tl.program_id(0)
    g = tl.program_id(1)
    expert = tl.program_id(2)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    blk_base = expert * NG16
    scl_base = expert * NG
    out_base = tl.cast(expert, tl.int64) * KN

    scl_addr = scl_base + n_offs * G + g
    scale_adj = tl.load(scales_ptr + scl_addr, mask=n_mask, other=0).to(tl.int32) - 113

    for b in tl.static_range(16):
        blk_addr = blk_base + n_offs * (G * 16) + g * 16 + b
        packed = tl.load(blocks_ptr + blk_addr, mask=n_mask, other=0).to(tl.int32)

        lo = packed & 0x0F
        hi = packed >> 4
        lo_fp8 = _convert_nibble_to_fp8(lo, scale_adj)
        hi_fp8 = _convert_nibble_to_fp8(hi, scale_adj)

        k_lo = g * 32 + b * 2
        k_hi = k_lo + 1

        lo_addr = out_base + tl.cast(k_lo, tl.int64) * N + n_offs.to(tl.int64)
        hi_addr = out_base + tl.cast(k_hi, tl.int64) * N + n_offs.to(tl.int64)
        tl.store(out_ptr + lo_addr, lo_fp8.to(tl.uint8), mask=n_mask)
        tl.store(out_ptr + hi_addr, hi_fp8.to(tl.uint8), mask=n_mask)


def mxfp4_to_fp8_e5m2_colmajor(blocks, scales):
    """Convert MXFP4 → FP8 E5M2 with column-major output [E, K, N].

    Output has N contiguous → after .transpose(-1,-2) gives [E, N, K]
    with stride(-2)==1, as required by Hopper FP8 matmul_ogs backward.

    Args:
        blocks: [E, N, G, 16] uint8 — packed FP4
        scales: [E, N, G] uint8 — E8M0 block scales

    Returns:
        fp8_tensor: [E, K, N] float8_e5m2 (K = G*32, N contiguous)
    """
    assert blocks.dim() == 4
    E, N, G, B = blocks.shape
    assert B == 16
    K = G * 32

    out = torch.empty(E, K, N, dtype=torch.uint8, device=blocks.device)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), G, E)
    _mxfp4_to_fp8_colmajor_kernel[grid](
        blocks.contiguous(), scales.contiguous(), out,
        N, G,
        N * G * 16, N * G, K * N,
    )

    return out.view(torch.float8_e5m2)


# ---------------------------------------------------------------------------
# Per-token dynamic FP8 quantization
# ---------------------------------------------------------------------------

@triton.jit
def _dynamic_quantize_per_token_kernel(
    x_ptr, out_ptr, scale_ptr,
    M, K,
    stride_x_m, stride_x_k,
    stride_out_m, stride_out_k,
    EPS: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused per-token quantization: BF16 → FP8 E4M3FN with per-row scale.

    Each program handles one row. Two passes:
      Pass 1: row-wise amax (data stays in L2)
      Pass 2: scale and cast to FP8 (L2 hit)
    """
    row = tl.program_id(0).to(tl.int64)
    x_base = x_ptr + row * stride_x_m
    out_base = out_ptr + row * stride_out_m

    amax = tl.zeros([], dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        mask = k_offs < K
        x = tl.load(x_base + k_offs * stride_x_k, mask=mask, other=0.0)
        amax = tl.maximum(amax, tl.max(tl.abs(x.to(tl.float32))))

    scale = tl.maximum(amax, EPS) / FP8_MAX
    inv_scale = 1.0 / scale
    tl.store(scale_ptr + row, scale)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        mask = k_offs < K
        x = tl.load(x_base + k_offs * stride_x_k, mask=mask, other=0.0)
        x_fp8 = (x.to(tl.float32) * inv_scale).to(tl.float8e4nv)
        tl.store(out_base + k_offs * stride_out_k, x_fp8, mask=mask)


def dynamic_quantize_per_token(x):
    """Quantize activation to FP8 E4M3FN with per-token (per-row) scale."""
    M, K = x.shape
    out = torch.empty(M, K, dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty(M, dtype=torch.float32, device=x.device)
    BLOCK_K = triton.next_power_of_2(K)
    _dynamic_quantize_per_token_kernel[(M,)](
        x, out, scale,
        M, K,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        EPS=1e-12, FP8_MAX=448.0, BLOCK_K=BLOCK_K,
    )
    return out, scale


# ---------------------------------------------------------------------------
# Fused SwiGLU (forward + backward)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 512}, num_warps=8),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 1024}, num_warps=16),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 256}, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.jit
def _swiglu_fwd_kernel(
    inp_ptr,   # [M, 2*N] bf16 — interleaved [gate, up, gate, up, ...]
    out_ptr,   # [M, N] bf16 — SwiGLU output
    gate_ptr,  # [M, N] bf16 — clamped gate (saved for backward), or None
    up_ptr,    # [M, N] bf16 — clamped up (saved for backward), or None
    M, N,
    stride_inp_m,
    stride_out_m,
    alpha,
    limit,
    SAVE_FOR_BWD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offs < M
    n_mask = n_offs < N
    mask = m_mask[:, None] & n_mask[None, :]

    inp_base = m_offs[:, None].to(tl.int64) * stride_inp_m + n_offs[None, :].to(tl.int64) * 2
    gate_raw = tl.load(inp_ptr + inp_base, mask=mask, other=0.0).to(tl.float32)
    up_raw = tl.load(inp_ptr + inp_base + 1, mask=mask, other=0.0).to(tl.float32)

    gate = tl.minimum(gate_raw, limit)
    up = tl.clamp(up_raw, -limit, limit)

    neg_alpha_gate = -alpha * gate
    log2_e: tl.constexpr = 1.4426950408889634
    exp_val = tl.inline_asm_elementwise(
        "ex2.approx.ftz.f32 $0, $1;", "=r, r",
        [neg_alpha_gate * log2_e], dtype=tl.float32, is_pure=True, pack=1,
    )
    sig = 1.0 / (1.0 + exp_val)
    result = gate * sig * (up + 1.0)

    out_base = m_offs[:, None].to(tl.int64) * stride_out_m + n_offs[None, :].to(tl.int64)
    tl.store(out_ptr + out_base, result.to(out_ptr.dtype.element_ty), mask=mask)

    if SAVE_FOR_BWD:
        tl.store(gate_ptr + out_base, gate.to(gate_ptr.dtype.element_ty), mask=mask)
        tl.store(up_ptr + out_base, up.to(up_ptr.dtype.element_ty), mask=mask)


def _swiglu_forward_triton(gate_up_raw, alpha, limit, save_for_bwd=False):
    """Fused SwiGLU forward using Triton kernel.

    Args:
        gate_up_raw: [..., 2*N] tensor with interleaved gate/up values
        alpha: sigmoid scaling factor
        limit: clamp limit
        save_for_bwd: if True, also returns (gate, up) for backward

    Returns:
        out: [..., N] SwiGLU output
        gate: [..., N] clamped gate (only if save_for_bwd)
        up: [..., N] clamped up (only if save_for_bwd)
    """
    assert gate_up_raw.shape[-1] % 2 == 0
    assert gate_up_raw.stride(-1) == 1
    prefix_shape = gate_up_raw.shape[:-1]
    N = gate_up_raw.shape[-1] // 2
    M = gate_up_raw.numel() // gate_up_raw.shape[-1]

    out = torch.empty(*prefix_shape, N, dtype=gate_up_raw.dtype, device=gate_up_raw.device)
    if save_for_bwd:
        gate_save = torch.empty(*prefix_shape, N, dtype=gate_up_raw.dtype, device=gate_up_raw.device)
        up_save = torch.empty(*prefix_shape, N, dtype=gate_up_raw.dtype, device=gate_up_raw.device)
    else:
        gate_save = None
        up_save = None

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    _swiglu_fwd_kernel[grid](
        gate_up_raw, out, gate_save, up_save,
        M, N,
        gate_up_raw.stride(-2), out.stride(-2) if out.dim() > 1 else N,
        alpha, limit,
        SAVE_FOR_BWD=save_for_bwd,
    )

    if save_for_bwd:
        return out, gate_save, up_save
    return out


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 512}, num_warps=8),
        triton.Config({"BLOCK_M": 4, "BLOCK_N": 1024}, num_warps=16),
        triton.Config({"BLOCK_M": 8, "BLOCK_N": 256}, num_warps=8),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=8),
    ],
    key=["M", "N"],
)
@triton.jit
def _swiglu_bwd_kernel(
    grad_out_ptr,   # [M, N] bf16
    gate_ptr,       # [M, N] bf16
    up_ptr,         # [M, N] bf16
    grad_gu_ptr,    # [M, N] uint32 — packed (grad_gate_bf16, grad_up_bf16)
    M, N,
    stride_grad_m,
    stride_gate_m,
    stride_out_m,
    alpha,
    limit,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = m_offs < M
    n_mask = n_offs < N
    mask = m_mask[:, None] & n_mask[None, :]

    grad_base = m_offs[:, None].to(tl.int64) * stride_grad_m + n_offs[None, :].to(tl.int64)
    gate_base = m_offs[:, None].to(tl.int64) * stride_gate_m + n_offs[None, :].to(tl.int64)

    grad = tl.load(grad_out_ptr + grad_base, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(gate_ptr + gate_base, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr + gate_base, mask=mask, other=0.0).to(tl.float32)

    neg_alpha_gate = -alpha * gate
    log2_e: tl.constexpr = 1.4426950408889634
    exp_val = tl.inline_asm_elementwise(
        "ex2.approx.ftz.f32 $0, $1;", "=r, r",
        [neg_alpha_gate * log2_e], dtype=tl.float32, is_pure=True, pack=1,
    )
    sig = 1.0 / (1.0 + exp_val)

    grad_gate = grad * (up + 1.0) * sig * (1.0 + alpha * gate * (1.0 - sig))
    grad_gate = tl.where(gate < limit, grad_gate, 0.0)

    grad_up = grad * gate * sig
    grad_up = tl.where((up > -limit) & (up < limit), grad_up, 0.0)

    # Pack (grad_gate, grad_up) as uint32 for contiguous store
    gate_bits = grad_gate.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
    up_bits = grad_up.to(tl.bfloat16).to(tl.uint16, bitcast=True).to(tl.uint32)
    combined = gate_bits | (up_bits << 16)

    out_base = m_offs[:, None].to(tl.int64) * stride_out_m + n_offs[None, :].to(tl.int64)
    tl.store(grad_gu_ptr + out_base, combined, mask=mask)


def _swiglu_backward(grad_output, gate, up, alpha, limit):
    """Fused SwiGLU backward using Triton kernel.

    Reads grad_output, gate, up once → computes in registers → packs as
    uint32 → writes contiguously → reinterprets as interleaved bf16.

    Returns grad_gate_up with interleaved [gate, up] layout.
    """
    assert grad_output.stride(-1) == 1
    prefix_shape = grad_output.shape[:-1]
    N = grad_output.shape[-1]
    M = grad_output.numel() // N

    out_u32 = torch.empty(M, N, dtype=torch.uint32, device=grad_output.device)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    _swiglu_bwd_kernel[grid](
        grad_output, gate, up, out_u32,
        M, N,
        grad_output.stride(-2) if grad_output.dim() > 1 else N,
        gate.stride(-2) if gate.dim() > 1 else N,
        N,
        alpha, limit,
    )

    grad_gate_up = out_u32.view(torch.bfloat16).reshape(*prefix_shape, 2 * N)
    return grad_gate_up
