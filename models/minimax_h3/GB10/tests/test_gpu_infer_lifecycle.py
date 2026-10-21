from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import os
import sys
import tempfile
from types import SimpleNamespace
from pathlib import Path


GB10_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GB10_ROOT))

# ``gpu_infer`` resolves model paths at import time.  Only its small config lookup is needed for
# these CPU-only lifecycle tests; no model is loaded.
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="h3-lifecycle-"))
(_TEST_ROOT / "qwen").mkdir()
(_TEST_ROOT / "model").mkdir()
(_TEST_ROOT / "comfy" / "diffusion_models").mkdir(parents=True)
(_TEST_ROOT / "qwen" / "config.json").write_text(
    json.dumps({"quantization_config": {"ignored_layers": []}}), encoding="utf-8"
)
os.environ.setdefault("H3_MODEL_ROOT", str(_TEST_ROOT / "model"))
os.environ.setdefault("H3_QWEN_FP8", str(_TEST_ROOT / "qwen"))
os.environ.setdefault("H3_COMFY_ROOT", str(_TEST_ROOT / "comfy"))

import gpu_infer
import torch
import pytest


@pytest.mark.parametrize("frames", [1, 16, 17, 124])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
)])
def test_video_chunks_preserve_pixels_and_bound_conversion(frames, dtype, device):
    # Include clipping and rounding boundaries, with a non-contiguous FCHW view.
    video = torch.linspace(-0.1, 1.1, frames * 3 * 8 * 10, dtype=dtype, device=device)
    video = video.reshape(frames, 3, 8, 10)
    video = video[..., ::2]
    chunks = gpu_infer.video_uint8_chunks(video)
    first = next(chunks)
    assert first.shape == (min(frames, 16), 8, 5, 3)
    if frames > 16:
        video[16:] = 0.5  # Later frames must not have been converted eagerly.
    parts = [first, *chunks]
    assert all(part.shape[0] <= 16 for part in parts)
    expected = video.permute(0, 2, 3, 1).mul(255).round().clamp(0, 255).to(torch.uint8)
    assert torch.equal(torch.cat(parts), expected)


def test_chunked_export_preserves_encoded_video_and_audio(tmp_path):
    av = pytest.importorskip("av")
    from diffusers.utils.export_utils import encode_video
    import numpy as np

    video = torch.rand(17, 3, 16, 24, generator=torch.Generator().manual_seed(42))
    audio = torch.sin(torch.arange(22666).float() * 0.1).repeat(2, 1) * 0.1
    reference = video.permute(0, 2, 3, 1).mul(255).round().clamp(0, 255).to(torch.uint8)
    for name, frames, count in (("reference", reference, 1),
                                ("chunked", gpu_infer.video_uint8_chunks(video), 2)):
        encode_video(frames, fps=24, output_path=str(tmp_path / f"{name}.mp4"),
                     audio=audio, audio_sample_rate=32000, video_chunks_number=count)
    for kind in ("video", "audio"):
        with av.open(str(tmp_path / "reference.mp4")) as left, av.open(str(tmp_path / "chunked.mp4")) as right:
            expected = list(left.decode(**{kind: 0}))
            actual = list(right.decode(**{kind: 0}))
            assert len(actual) == len(expected) > 0
            for a, b in zip(actual, expected):
                assert (a.pts, a.time_base) == (b.pts, b.time_base)
                np.testing.assert_array_equal(a.to_ndarray(), b.to_ndarray())


class _Timer:
    def __init__(self):
        self.calls = 0
        self.devices = []

    @contextlib.contextmanager
    def span(self, device):
        self.calls += 1
        self.devices.append(device)
        yield


class _Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(self, value, *, marker=None):
        return value + self.weight, marker


class _BufferOnlyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.ones(()))

    def forward(self, value):
        return value * self.scale


class _EmptyModule(torch.nn.Module):
    def forward(self, value):
        return value


def test_instrument_is_idempotent_and_restorable():
    module = _Module()
    original = module.forward
    first_timer = _Timer()
    second_timer = _Timer()

    returned = gpu_infer.instrument(module, first_timer)
    wrapped = module.forward
    assert returned.__func__ is original.__func__
    assert inspect.signature(module.forward) == inspect.signature(original)

    # A second install updates the existing wrapper's timer instead of stacking another wrapper.
    returned_again = gpu_infer.instrument(module, second_timer)
    assert module.forward is wrapped
    assert returned_again.__func__ is original.__func__
    assert module.forward(torch.tensor(2), marker="x")[0].item() == 3
    assert first_timer.calls == 0
    assert second_timer.calls == 1

    assert gpu_infer.restore_instrumentation(module, original=returned_again)
    assert module.forward.__func__ is original.__func__
    assert not gpu_infer.restore_instrumentation(module)


def test_restore_does_not_clobber_external_replacement():
    module = _Module()
    original = gpu_infer.instrument(module, _Timer())
    replacement = lambda value, **kwargs: (value, kwargs)
    module.forward = replacement

    assert not gpu_infer.restore_instrumentation(module, original=original)
    assert module.forward is replacement


def test_instrument_uses_buffer_device_when_module_has_no_parameters():
    module = _BufferOnlyModule()
    timer = _Timer()
    gpu_infer.instrument(module, timer)

    assert module(torch.tensor(2)).item() == 2
    assert timer.devices == [module.scale.device]
    assert gpu_infer.restore_instrumentation(module)


def test_instrument_falls_back_to_current_cuda_for_empty_module(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    module = _EmptyModule()
    timer = _Timer()
    gpu_infer.instrument(module, timer)

    assert module(torch.tensor(2)).item() == 2
    assert timer.devices == [torch.device("cuda", 3)]
    assert gpu_infer.restore_instrumentation(module)


def test_request_median_uses_average_for_even_sample():
    assert gpu_infer._request_s_median([1.0, 3.0]) == 2.0


def test_checkpoint_provenance_hashes_the_effective_file(monkeypatch, tmp_path):
    checkpoint = tmp_path / "dit.safetensors"
    checkpoint.write_bytes(b"effective checkpoint")
    expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setenv("H3_VERIFY_DIT_SHA256", "1")
    monkeypatch.setenv("H3_EXPECTED_DIT_SHA256", expected)

    provenance = gpu_infer._checkpoint_provenance(str(checkpoint))

    assert provenance == {
        "path": str(checkpoint.resolve()),
        "size_bytes": checkpoint.stat().st_size,
        "sha256": expected,
    }


def test_checkpoint_provenance_rejects_a_digest_mismatch(monkeypatch, tmp_path):
    checkpoint = tmp_path / "dit.safetensors"
    checkpoint.write_bytes(b"wrong checkpoint")
    monkeypatch.setenv("H3_VERIFY_DIT_SHA256", "1")
    monkeypatch.setenv("H3_EXPECTED_DIT_SHA256", "0" * 64)

    try:
        gpu_infer._checkpoint_provenance(str(checkpoint))
    except RuntimeError as exc:
        assert "DiT checkpoint SHA-256 mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("mismatched DiT checkpoint was accepted")


def test_unaccounted_time_uses_per_request_component_total():
    # Timers are accumulated over two requests: 6 seconds total is 3 seconds/request.
    assert gpu_infer._unaccounted_s(5.0, 6.0, 2) == 2.0


def test_sol_stats_accumulate_counts_and_keep_latest_layout():
    total = {}
    gpu_infer._accumulate_sol_attn_stats(
        total,
        {
            "backend": "triton",
            "call": 4,
            "dense_calls": 2,
            "prefix_dense_calls": 1,
            "sparse_calls": 1,
            "step": 6,
            "layout_key": "first",
        },
    )
    gpu_infer._accumulate_sol_attn_stats(
        total,
        {
            "backend": "triton",
            "call": 5,
            "dense_calls": 3,
            "prefix_dense_calls": 2,
            "sparse_calls": 2,
            "step": 6,
            "layout_key": "second",
        },
    )

    assert total["request_count"] == 2
    assert total["call"] == 9
    assert total["dense_calls"] == 5
    assert total["prefix_dense_calls"] == 3
    assert total["sparse_calls"] == 3
    assert total["layout_key"] == "second"


def test_request_count_validation_rejects_empty_measurement():
    gpu_infer._validate_request_counts(0, 1)
    gpu_infer._validate_request_counts(3, 2)
    for warmup, measure, expected in (
        (-1, 1, "warmup"),
        (0, 0, "measure"),
    ):
        try:
            gpu_infer._validate_request_counts(warmup, measure)
        except ValueError as exc:
            assert expected in str(exc)
        else:  # pragma: no cover
            raise AssertionError("invalid request count was accepted")


def test_sol_attn_threshold_type_is_validated_and_normalized(monkeypatch):
    for name in (
        "H3_SOL_ATTN_THRESH_TYPE",
        "H3_SOL_THRESH_TYPE",
        "SOL_ATTN_THRESH_TYPE",
    ):
        monkeypatch.delenv(name, raising=False)
    assert gpu_infer._sol_attn_thresh_type() == "diag"

    monkeypatch.setenv("H3_SOL_ATTN_THRESH_TYPE", " exact ")
    assert gpu_infer._sol_attn_thresh_type() == "exact"

    monkeypatch.delenv("H3_SOL_ATTN_THRESH_TYPE")
    monkeypatch.setenv("H3_SOL_THRESH_TYPE", "diag")
    assert gpu_infer._sol_attn_thresh_type() == "diag"

    monkeypatch.delenv("H3_SOL_THRESH_TYPE")
    monkeypatch.setenv("SOL_ATTN_THRESH_TYPE", " exact ")
    assert gpu_infer._sol_attn_thresh_type() == "exact"

    monkeypatch.setenv("H3_SOL_ATTN_THRESH_TYPE", "diag")
    try:
        gpu_infer._sol_attn_thresh_type()
    except ValueError as exc:
        assert "conflicting" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("conflicting Sol-Attn aliases were accepted")

    monkeypatch.delenv("H3_SOL_ATTN_THRESH_TYPE")
    monkeypatch.delenv("SOL_ATTN_THRESH_TYPE")
    monkeypatch.setenv("H3_SOL_ATTN_THRESH_TYPE", "invalid")
    try:
        gpu_infer._sol_attn_thresh_type()
    except ValueError as exc:
        assert "diag" in str(exc) and "exact" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid Sol-Attn threshold type was accepted")


def test_restore_process_patches_removes_cache_before_sol_layout(monkeypatch):
    events = []
    model = object()
    vae = object()
    processor = object()
    text_encoder = object()
    fake_cache = SimpleNamespace(remove_cache=lambda value: events.append(("cache", value)))
    fake_fusion = SimpleNamespace(
        restore_fused_patches=lambda: events.append(("fused",)),
        uninstall_sol_attn=lambda value: events.append(("sol", value)),
    )
    fake_vae = SimpleNamespace(unpatch_batched_tiles=lambda value: events.append(("vae", value)))
    fake_compat = SimpleNamespace(
        restore_transformers_patches=lambda **kwargs: events.append(
            ("transformers", kwargs)
        )
    )
    monkeypatch.setitem(sys.modules, "cache_line", fake_cache)
    monkeypatch.setitem(sys.modules, "fusion_install", fake_fusion)
    monkeypatch.setitem(sys.modules, "vae_shard", fake_vae)
    monkeypatch.setitem(sys.modules, "transformers_compat", fake_compat)
    monkeypatch.setattr(gpu_infer._restore_process_patches, "_model", model, raising=False)
    monkeypatch.setattr(gpu_infer._restore_process_patches, "_vae", vae, raising=False)
    monkeypatch.setattr(
        gpu_infer._restore_process_patches, "_processor", processor, raising=False
    )
    monkeypatch.setattr(
        gpu_infer._restore_process_patches, "_text_encoder", text_encoder, raising=False
    )

    gpu_infer._restore_process_patches()

    assert [event[0] for event in events] == ["cache", "fused", "sol", "vae", "transformers"]
    assert events[-1][1] == {
        "processor": processor,
        "text_encoder": text_encoder,
    }
