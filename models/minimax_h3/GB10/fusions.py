"""The four fused Triton kernels, and the activation quantisers they replace.

Sibling of `optimized/fusions.py`, which fuses the same sites for the 8xGB200 line. Every
kernel here is required to be *bit-identical* to the eager path it replaces, not merely close,
because the whole point of the lossless tier is that its speedup can be taken without an
accuracy argument. Two of the corrections that buys are unobvious enough to restate:

* `tl.math.div_rn`, not `/`. Triton's default float division is an approximate `div.full.f32`
  good to about 2 ulp, and 1 ulp is the entire story: it moves a value sitting exactly on an
  E4M3 midpoint just past it, and the downcast then rounds the other way. Real activations put
  0.94% of elements on such a midpoint.
* Rounding in the integer domain. `x.to(tl.bfloat16).to(tl.float32)` is folded away as an
  identity by the compiler, so a fused chain would keep FP32 precision throughout and be *more*
  accurate than eager — and therefore not bit-identical. `_round_bf16` does it with integer
  arithmetic the compiler cannot elide.
"""

from __future__ import annotations

import torch

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0

_FUSION_NAMES = ("quantize", "modulate", "gate_add", "rope", "swiglu")
_FUSION_HITS = {name: 0 for name in _FUSION_NAMES}


def _record_fusion_hit(name: str) -> None:
    _FUSION_HITS[name] += 1


def reset_fusion_stats() -> None:
    for name in _FUSION_NAMES:
        _FUSION_HITS[name] = 0


def read_fusion_stats() -> dict[str, int]:
    return dict(_FUSION_HITS)

def _quantize_eager(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Activation -> FP8 E4M3, the obvious way: four kernels and an FP32 round trip.

    At H3's shapes this reads and writes about 2.5 GB per call for 205 MB of output, and the
    profile charges it 3.7 s of `div` + `clamp_` alone across the 250 quantised linears.
    """
    return (x.float() / scale).clamp_(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)


# Same arithmetic, one kernel — but *not* the same result. Inductor's FP32->FP8 downcast breaks
# ties away from zero where eager breaks them to even, and E4M3's 3-bit mantissa puts exact
# midpoints all over real activations: on block 39's `to_out` input, 0.94% of elements land on
# one (42.0 between 40 and 44, 5.25 between 5.0 and 5.5, ...). That is 4.0 s faster and 3.9%
# different, so it is kept only as a measurement, not as an option.
_quantize_compiled = torch.compile(_quantize_eager, dynamic=False)


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _quantize_triton_kernel(x_ptr, scale_ptr, out_ptr, n, LIMIT: tl.constexpr,
                                BLOCK: tl.constexpr):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        # `div_rn`, not `/`: Triton's default float division is an approximate `div.full.f32`
        # good to ~2 ulp, and 1 ulp is the whole story here — it moves a value that sits exactly
        # on an E4M3 midpoint just above it, so the downcast rounds the other way. With
        # correctly-rounded division the midpoints stay midpoints and `rtne` reproduces eager.
        y = tl.math.div_rn(x, tl.load(scale_ptr))
        y = tl.minimum(tl.maximum(y, -LIMIT), LIMIT)
        # `rtne` is the rule eager's cast uses; inductor's unstated default rounds ties away.
        tl.store(out_ptr + offsets, y.to(tl.float8e4nv, fp_downcast_rounding="rtne"), mask=mask)

    def _quantize_triton(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """One kernel, BF16 in and FP8 out, with eager's tie-breaking."""
        _record_fusion_hit("quantize")
        x = x.contiguous()
        out = torch.empty(x.shape, dtype=FP8_DTYPE, device=x.device)
        n = x.numel()
        _quantize_triton_kernel[(triton.cdiv(n, 4096),)](
            x, scale, out, n, LIMIT=FP8_MAX, BLOCK=4096
        )
        return out

except ImportError:  # pragma: no cover - triton ships with the torch builds used here
    _quantize_triton = None


# `triton` is the one to use: it is the only fused variant that reproduces eager bit for bit on
# real activations, across all 150 quantised linears. `compiled` is kept so the 3.9% it costs
# stays reproducible rather than anecdotal.
QUANTIZERS = {
    "eager": _quantize_eager,
    "compiled": _quantize_compiled,
    "triton": _quantize_triton,
}


# Each block spends six `index_select`s and four elementwise passes on modulation:
#
#     h = norm(x) * (1 + scale[idx]) + shift[idx]        (twice, msa and mlp)
#     x = residual + gate[idx] * branch                  (twice)
#
# Every `index_select` materialises a full `(38247, 5376)` tensor — 411 MB — to read a table
# of at most nine rows, and each `*` and `+` reads and writes another. The profile charges
# 0.66 s of `gather` and about 2 s of the `mul`/`add` total to this. Gathering inside the
# kernel instead turns roughly 3.3 GB of traffic per site into 0.8 GB.
#
# Both kernels round after every arithmetic step, because that is what the eager ops do: a
# BF16 `a * b` computes in FP32 and rounds once on store, so a fused version that keeps full
# precision throughout would be *more* accurate and therefore not bit-identical.

if _quantize_triton is not None:

    @triton.jit
    def _round_bf16(x):
        """Round an FP32 value to BF16 precision, in FP32, without the compiler eliding it.

        Writing this as `x.to(tl.bfloat16).to(tl.float32)` does not survive: Triton folds the
        truncate/extend pair away as an identity, so the intermediate roundings vanish and the
        kernel silently computes `x * (1 + scale) + shift` at full FP32 precision — one BF16 ulp
        away from the three separately-rounded aten ops on 27% of elements. Doing the round in
        the integer domain is opaque to that fold. Round-to-nearest-even: add half an ulp plus
        the low bit of the truncated result, then mask the mantissa.
        """
        bits = x.to(tl.int32, bitcast=True)
        bits = bits + 0x7FFF + ((bits >> 16) & 1)
        return (bits & -65536).to(tl.float32, bitcast=True)  # -65536 == 0xFFFF0000 as int32

    @triton.jit
    def _modulate_kernel(x_ptr, scale_ptr, shift_ptr, idx_ptr, out_ptr, n_cols,
                         BLOCK: tl.constexpr):
        row = tl.program_id(0)
        table_row = tl.load(idx_ptr + row)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + table_row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + table_row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        out = _round_bf16(_round_bf16(x * _round_bf16(1.0 + scale)) + shift)
        tl.store(out_ptr + row * n_cols + cols, out.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _modulate_batched_kernel(x_ptr, scale_ptr, shift_ptr, idx_ptr, out_ptr, n_cols,
                                 n_index, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        # `adaln_indices` describes the packed sequence once and is broadcast over the batch.
        table_row = tl.load(idx_ptr + (row % n_index))
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(x_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr + table_row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + table_row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        out = _round_bf16(_round_bf16(x * _round_bf16(1.0 + scale)) + shift)
        tl.store(out_ptr + row * n_cols + cols, out.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _gate_add_kernel(residual_ptr, gate_ptr, branch_ptr, idx_ptr, out_ptr, n_cols,
                         BLOCK: tl.constexpr):
        row = tl.program_id(0)
        table_row = tl.load(idx_ptr + row)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        residual = tl.load(residual_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        branch = tl.load(branch_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + table_row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        out = _round_bf16(residual + _round_bf16(gate * branch))
        tl.store(out_ptr + row * n_cols + cols, out.to(tl.bfloat16), mask=mask)

    @triton.jit
    def _gate_add_batched_kernel(residual_ptr, gate_ptr, branch_ptr, idx_ptr, out_ptr, n_cols,
                                 n_index, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        # See `_modulate_batched_kernel`: table rows are indexed per sequence position.
        table_row = tl.load(idx_ptr + (row % n_index))
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        residual = tl.load(residual_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        branch = tl.load(branch_ptr + row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + table_row * n_cols + cols, mask=mask, other=0.0).to(tl.float32)
        out = _round_bf16(residual + _round_bf16(gate * branch))
        tl.store(out_ptr + row * n_cols + cols, out.to(tl.bfloat16), mask=mask)

    def _launch(kernel, first, *rest):
        rows, cols = first.shape[-2], first.shape[-1]
        out = torch.empty_like(first)
        block = triton.next_power_of_2(cols)
        kernel[(rows,)](first, *rest, out, cols, BLOCK=block, num_warps=8)
        return out

    def _launch_batched(kernel, first, *rest, index_count):
        rows, cols = first.shape[-2], first.shape[-1]
        out = torch.empty_like(first)
        block = triton.next_power_of_2(cols)
        kernel[(rows,)](
            first, *rest, out, cols, index_count, BLOCK=block, num_warps=8
        )
        return out

    # The modulation tables arrive as `chunk(6, dim=-1)` views of one `(T, 6 * hidden)` buffer,
    # so their row stride is `6 * hidden`, not `hidden`. The kernels index them as dense rows;
    # feeding the views straight in reads the wrong row for every modality but the first, which
    # is why video came out only slightly wrong (tag 0 -> row 0) and audio came out destroyed
    # (tag 2 -> row 2). They are at most nine rows, so compacting them costs nothing.
    def fused_modulate(x, scale, shift, indices):
        """`x * (1 + scale[indices]) + shift[indices]`, one kernel."""
        _record_fusion_hit("modulate")
        if x.numel() == 0:
            return x.clone()
        if x.ndim == 3 and x.shape[0] == 1:
            # Keep the established single-request path unchanged: no flatten or modulo in the
            # kernel, which is the shape used by the Thor CLI.
            return _launch(
                _modulate_kernel,
                x.squeeze(0).contiguous(),
                scale.contiguous(),
                shift.contiguous(),
                indices.contiguous(),
            ).unsqueeze(0)
        if x.ndim < 2:
            raise ValueError(f"fused_modulate expects at least 2 dimensions, got {list(x.shape)}")
        sequence_length = x.shape[-2]
        if indices.ndim != 1 or indices.numel() != sequence_length:
            raise ValueError(
                "fused_modulate expects one index per sequence row, got "
                f"indices={list(indices.shape)} for sequence_length={sequence_length}"
            )
        indices = indices.contiguous()
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        out = _launch_batched(
            _modulate_batched_kernel,
            flat,
            scale.contiguous(),
            shift.contiguous(),
            indices,
            index_count=indices.numel(),
        )
        return out.reshape_as(x)

    def fused_gate_add(residual, gate, branch, indices):
        """`residual + gate[indices] * branch`, one kernel."""
        _record_fusion_hit("gate_add")
        if residual.numel() == 0:
            return residual.clone()
        if residual.ndim == 3 and residual.shape[0] == 1:
            # Keep the established single-request path unchanged; see `fused_modulate`.
            return _launch(
                _gate_add_kernel,
                residual.squeeze(0).contiguous(),
                gate.contiguous(),
                branch.squeeze(0).contiguous(),
                indices.contiguous(),
            ).unsqueeze(0)
        if residual.ndim < 2 or branch.shape != residual.shape:
            raise ValueError(
                "fused_gate_add expects residual and branch with the same shape, got "
                f"{list(residual.shape)} and {list(branch.shape)}"
            )
        sequence_length = residual.shape[-2]
        if indices.ndim != 1 or indices.numel() != sequence_length:
            raise ValueError(
                "fused_gate_add expects one index per sequence row, got "
                f"indices={list(indices.shape)} for sequence_length={sequence_length}"
            )
        indices = indices.contiguous()
        residual_flat = residual.reshape(-1, residual.shape[-1]).contiguous()
        branch_flat = branch.reshape(-1, branch.shape[-1]).contiguous()
        out = _launch_batched(
            _gate_add_batched_kernel,
            residual_flat,
            gate.contiguous(),
            branch_flat,
            indices,
            index_count=indices.numel(),
        )
        return out.reshape_as(residual)

else:  # pragma: no cover
    fused_modulate = fused_gate_add = None


# ---------------------------------------------------------------------------------------
# Rotary embedding
# ---------------------------------------------------------------------------------------
#
# `_apply_rotary_emb` rotates 96 of every head's 128 channels and passes the other 32 through.
# Upstream spells that as `neg`, `cat`, two `mul`s, an `add`, a second `cat` and a
# `contiguous` — seven kernels moving about 4.5 GB per call, twice per block. One kernel that
# reads its partner channel directly instead of materialising the rotated copy moves 1.1 GB.

if _quantize_triton is not None:

    @triton.jit
    def _rope_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, width, head_dim: tl.constexpr,
                     rotary_dim: tl.constexpr, XS: tl.constexpr, XH: tl.constexpr,
                     XD: tl.constexpr, BLOCK: tl.constexpr):
        # Tiled across the row rather than one program per row: a full row is
        # `heads * head_dim` = 7168 wide, and holding that twice (the value and its rotation
        # partner) spills registers badly enough to make the fused kernel slower than the six
        # eager ops it replaces.
        row = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < width
        channel = cols % head_dim
        half = rotary_dim // 2
        rotating = channel < rotary_dim
        low = channel < half

        base = row.to(tl.int64) * XS + (cols // head_dim) * XH
        x = tl.load(x_ptr + base + channel * XD, mask=mask, other=0.0).to(tl.float32)
        # `rotate_half` reads the partner channel: `cat((-x2, x1))` is `-x[c + half]` on the
        # low half and `x[c - half]` on the high half.
        partner = tl.where(low, channel + half, channel - half)
        paired = tl.load(x_ptr + base + partner * XD, mask=mask & rotating, other=0.0)
        rotated = tl.where(low, -paired.to(tl.float32), paired.to(tl.float32))

        # `cos`/`sin` are FP32 buffers that upstream casts to the activation dtype *before* the
        # multiply, so the cast has to happen here too or the products differ by an ulp.
        cos = _round_bf16(tl.load(cos_ptr + row * rotary_dim + channel,
                                  mask=mask & rotating, other=0.0))
        sin = _round_bf16(tl.load(sin_ptr + row * rotary_dim + channel,
                                  mask=mask & rotating, other=0.0))
        y = _round_bf16(_round_bf16(x * cos) + _round_bf16(rotated * sin))
        tl.store(out_ptr + row * width + cols, tl.where(rotating, y, x).to(tl.bfloat16), mask=mask)

    @triton.jit
    def _rope_batched_kernel(x_ptr, cos_ptr, sin_ptr, out_ptr, width, seq_len,
                             head_dim: tl.constexpr, rotary_dim: tl.constexpr,
                             XB: tl.constexpr, XS: tl.constexpr, XH: tl.constexpr,
                             XD: tl.constexpr,
                             BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < width
        channel = cols % head_dim
        half = rotary_dim // 2
        rotating = channel < rotary_dim
        low = channel < half

        base = ((row // seq_len).to(tl.int64) * XB
                + (row % seq_len).to(tl.int64) * XS + (cols // head_dim) * XH)
        x = tl.load(x_ptr + base + channel * XD, mask=mask, other=0.0).to(tl.float32)
        partner = tl.where(low, channel + half, channel - half)
        paired = tl.load(x_ptr + base + partner * XD, mask=mask & rotating, other=0.0)
        rotated = tl.where(low, -paired.to(tl.float32), paired.to(tl.float32))

        # The rotary tables have one row per sequence position and are broadcast over batches.
        token = row % seq_len
        cos = _round_bf16(tl.load(cos_ptr + token * rotary_dim + channel,
                                  mask=mask & rotating, other=0.0))
        sin = _round_bf16(tl.load(sin_ptr + token * rotary_dim + channel,
                                  mask=mask & rotating, other=0.0))
        y = _round_bf16(_round_bf16(x * cos) + _round_bf16(rotated * sin))
        tl.store(out_ptr + row * width + cols, tl.where(rotating, y, x).to(tl.bfloat16), mask=mask)

    def fused_apply_rotary_emb(hidden_states, cos, sin):
        """Drop-in for `_apply_rotary_emb`, `(batch, seq, heads, head_dim)` in and out."""
        _record_fusion_hit("rope")
        batch, seq_len, heads, head_dim = hidden_states.shape
        rotary_dim = cos.shape[-1]
        if hidden_states.numel() == 0:
            return hidden_states.clone()
        if cos.ndim != 2 or cos.shape[0] != seq_len or sin.shape != cos.shape:
            raise ValueError(
                "fused_apply_rotary_emb expects cos/sin shaped (sequence_length, rotary_dim), "
                f"got {list(cos.shape)} and {list(sin.shape)} for sequence_length={seq_len}"
            )
        cos = cos.contiguous()
        sin = sin.contiguous()
        # Read packed Q/K views directly, as the Spark attention path does: a
        # contiguous staging tensor would copy every token before rotating it.
        out = torch.empty(hidden_states.shape, dtype=hidden_states.dtype, device=hidden_states.device)
        xb, xs, xh, xd = hidden_states.stride()
        width = heads * head_dim
        block = 1024
        grid = (batch * seq_len, triton.cdiv(width, block))
        if batch == 1:
            # Avoid batch indexing for the common single-request case.
            _rope_kernel[grid](
                hidden_states, cos, sin, out, width, head_dim=head_dim, rotary_dim=rotary_dim,
                XS=xs, XH=xh, XD=xd,
                BLOCK=block, num_warps=4,
            )
        else:
            _rope_batched_kernel[grid](
                hidden_states, cos, sin, out, width, seq_len, XB=xb, XS=xs, XH=xh, XD=xd,
                head_dim=head_dim, rotary_dim=rotary_dim, BLOCK=block, num_warps=4,
            )
        return out.reshape(batch, seq_len, heads, head_dim)

else:  # pragma: no cover
    fused_apply_rotary_emb = None


# ---------------------------------------------------------------------------------------
# SwiGLU
# ---------------------------------------------------------------------------------------
#
# `fc1` emits `(seq, 2 * ffn_dim)` — 2.19 GB at the published cell — which upstream then reads
# twice more: once for `silu(gate)` and once for `value * that`. Reading it once and writing
# only the product cuts the post-GEMM traffic from about 5.5 GB to 3.3 GB per block.

if _quantize_triton is not None:

    @triton.jit
    def _swiglu_kernel(x_ptr, out_ptr, half, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < half
        value = tl.load(x_ptr + row * 2 * half + cols, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(x_ptr + row * 2 * half + half + cols, mask=mask, other=0.0).to(tl.float32)
        activated = _round_bf16(gate * tl.sigmoid(gate))
        tl.store(out_ptr + row * half + cols,
                 _round_bf16(value * activated).to(tl.bfloat16), mask=mask)

    def fused_swiglu(x):
        """`value * silu(gate)` over a fused `[value; gate]` row, one kernel."""
        _record_fusion_hit("swiglu")
        *lead, width = x.shape
        half = width // 2
        flat = x.reshape(-1, width).contiguous()
        out = torch.empty(flat.shape[0], half, dtype=x.dtype, device=x.device)
        block = 1024
        _swiglu_kernel[(flat.shape[0], triton.cdiv(half, block))](
            flat, out, half, BLOCK=block, num_warps=4
        )
        return out.reshape(*lead, half)

else:  # pragma: no cover
    fused_swiglu = None
