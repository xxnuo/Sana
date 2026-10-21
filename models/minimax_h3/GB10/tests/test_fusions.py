from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

GB10_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GB10_ROOT))

import fusions

CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available() or fusions._quantize_triton is None,
    reason="Triton fusion tests require CUDA and Triton",
)


def test_fusion_stats_reset_and_copy() -> None:
    fusions.reset_fusion_stats()
    assert fusions.read_fusion_stats() == {
        "quantize": 0,
        "modulate": 0,
        "gate_add": 0,
        "rope": 0,
        "swiglu": 0,
    }

    fusions._record_fusion_hit("rope")
    snapshot = fusions.read_fusion_stats()
    snapshot["rope"] = 99
    assert fusions.read_fusion_stats()["rope"] == 1
    fusions.reset_fusion_stats()


@CUDA_REQUIRED
@pytest.mark.parametrize("batch", [1, 2])
def test_fused_modulate_broadcasts_indices_over_batch(batch: int) -> None:
    torch.manual_seed(11)
    device = torch.device("cuda")
    sequence_length, hidden_size, table_rows = 7, 32, 9
    table = torch.randn(
        (table_rows, hidden_size * 3), device=device, dtype=torch.bfloat16
    )
    scale, shift, _ = table.chunk(3, dim=-1)
    indices = torch.tensor([0, 8, 2, 5, 1, 7, 3], device=device, dtype=torch.int64)
    hidden = torch.randn(
        (batch, sequence_length, hidden_size), device=device, dtype=torch.bfloat16
    )

    actual = fusions.fused_modulate(hidden, scale, shift, indices)
    expected = hidden * (1.0 + scale.index_select(0, indices))
    expected = expected + shift.index_select(0, indices)

    assert actual.shape == hidden.shape
    assert torch.equal(actual, expected)


@CUDA_REQUIRED
@pytest.mark.parametrize("batch", [1, 2])
def test_fused_gate_add_broadcasts_indices_over_batch(batch: int) -> None:
    torch.manual_seed(12)
    device = torch.device("cuda")
    sequence_length, hidden_size, table_rows = 7, 32, 9
    table = torch.randn(
        (table_rows, hidden_size * 3), device=device, dtype=torch.bfloat16
    )
    _, _, gate = table.chunk(3, dim=-1)
    indices = torch.tensor([0, 8, 2, 5, 1, 7, 3], device=device, dtype=torch.int64)
    residual = torch.randn(
        (batch, sequence_length, hidden_size), device=device, dtype=torch.bfloat16
    )
    branch = torch.randn_like(residual)

    actual = fusions.fused_gate_add(residual, gate, branch, indices)
    expected = residual + gate.index_select(0, indices) * branch

    assert actual.shape == residual.shape
    assert torch.equal(actual, expected)


@CUDA_REQUIRED
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("layout", ["contiguous", "packed_qkv", "transposed", "sliced", "expanded"])
@pytest.mark.parametrize("heads,head_dim,rotary_dim", [(2, 8, 4), (56, 128, 96)])
def test_fused_rope_reuses_position_rows_over_batch(
    batch: int, layout: str, heads: int, head_dim: int, rotary_dim: int
) -> None:
    torch.manual_seed(13)
    device = torch.device("cuda")
    sequence_length = 7
    hidden_states = torch.randn(
        (batch, sequence_length, heads, head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    if layout == "packed_qkv":
        hidden_states = torch.cat([hidden_states] * 3, dim=2).chunk(3, dim=2)[1]
    elif layout == "transposed":
        hidden_states = hidden_states.transpose(1, 2).contiguous().transpose(1, 2)
    elif layout == "sliced":
        hidden_states = torch.repeat_interleave(hidden_states, 2, dim=-1)[..., ::2]
    elif layout == "expanded":
        hidden_states = hidden_states[:1].expand(batch, -1, -1, -1)
    cos = torch.randn(
        (sequence_length, rotary_dim), device=device, dtype=torch.float32
    )
    sin = torch.randn_like(cos)

    actual = fusions.fused_apply_rotary_emb(hidden_states, cos, sin)
    rotary, passthrough = hidden_states[..., :rotary_dim], hidden_states[..., rotary_dim:]
    first, second = rotary.chunk(2, dim=-1)
    rotated = rotary * cos.to(hidden_states.dtype)[None, :, None, :]
    rotated = rotated + torch.cat((-second, first), dim=-1) * sin.to(hidden_states.dtype)[
        None, :, None, :
    ]
    expected = torch.cat((rotated, passthrough), dim=-1).contiguous()

    assert actual.shape == hidden_states.shape
    assert actual.is_contiguous()
    assert torch.equal(actual, expected)

    # Only the small rotary tables may need staging; activation copies would
    # undo the bandwidth saving for packed Q/K and strided input views.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        fusions.fused_apply_rotary_emb(hidden_states, cos, sin)
    assert not {"aten::clone", "aten::contiguous"} & {event.key for event in profile.key_averages()}


@CUDA_REQUIRED
def test_single_batch_fast_paths_accept_strided_views() -> None:
    torch.manual_seed(14)
    device = torch.device("cuda")
    sequence_length, hidden_size, table_rows = 7, 32, 9
    table = torch.randn(
        (table_rows, hidden_size * 3), device=device, dtype=torch.bfloat16
    )
    scale, shift, gate = table.chunk(3, dim=-1)
    indices = torch.randint(
        0, table_rows, (sequence_length * 2,), device=device, dtype=torch.int64
    )[::2]
    hidden = torch.randn(
        (1, sequence_length, hidden_size * 2), device=device, dtype=torch.bfloat16
    )[..., ::2]
    branch = torch.randn(
        (1, sequence_length, hidden_size * 2), device=device, dtype=torch.bfloat16
    )[..., ::2]

    actual_modulate = fusions.fused_modulate(hidden, scale, shift, indices)
    expected_modulate = hidden * (1.0 + scale.index_select(0, indices))
    expected_modulate = expected_modulate + shift.index_select(0, indices)
    assert torch.equal(actual_modulate, expected_modulate)

    actual_gate = fusions.fused_gate_add(hidden, gate, branch, indices)
    expected_gate = hidden + gate.index_select(0, indices) * branch
    assert torch.equal(actual_gate, expected_gate)

    heads, head_dim, rotary_dim = 2, 8, 4
    rope_input = torch.randn(
        (1, sequence_length, heads, head_dim * 2),
        device=device,
        dtype=torch.bfloat16,
    )[..., ::2]
    cos = torch.randn(
        (sequence_length, rotary_dim * 2), device=device, dtype=torch.float32
    )[:, ::2]
    sin = torch.randn(
        (sequence_length, rotary_dim * 2), device=device, dtype=torch.float32
    )[:, ::2]
    actual_rope = fusions.fused_apply_rotary_emb(rope_input, cos, sin)
    rotary, passthrough = rope_input[..., :rotary_dim], rope_input[..., rotary_dim:]
    first, second = rotary.chunk(2, dim=-1)
    expected_rope = rotary * cos.to(rope_input.dtype)[None, :, None, :]
    expected_rope = expected_rope + torch.cat((-second, first), dim=-1) * sin.to(
        rope_input.dtype
    )[None, :, None, :]
    expected_rope = torch.cat((expected_rope, passthrough), dim=-1).contiguous()
    assert torch.equal(actual_rope, expected_rope)
