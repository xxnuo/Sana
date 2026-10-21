from __future__ import annotations

import functools
import sys
from pathlib import Path
from types import ModuleType

import pytest

GB10_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GB10_ROOT))

import transformers_compat  # noqa: E402


class _Tokenizer:
    def convert_tokens_to_ids(self, token):
        return {"<|image_pad|>": 10, "<|video_pad|>": 20}[token]


class _LegacyProcessor:
    def __init__(self):
        self.tokenizer = _Tokenizer()


class _LegacyTextModel:
    def forward(self, value, **kwargs):
        return value, kwargs


class _TextEncoder:
    def __init__(self):
        self.model = _LegacyTextModel()


def test_processor_instance_patch_is_idempotent_and_restorable():
    processor = _LegacyProcessor()

    assert transformers_compat.patch_processor(processor) is processor
    patched = processor.create_mm_token_type_ids
    assert patched([[0, 10, 20]]) == [[0, 1, 2]]
    assert transformers_compat.patch_processor(processor) is processor
    assert processor.create_mm_token_type_ids is patched

    assert transformers_compat.restore_instance_patches(processor=processor)
    assert "create_mm_token_type_ids" not in vars(processor)
    assert not hasattr(processor, "_h3_processor_patch_state")


def test_text_encoder_instance_patch_restores_inherited_forward():
    text_encoder = _TextEncoder()
    inner = text_encoder.model
    assert "forward" not in vars(inner)

    assert transformers_compat.patch_text_encoder(text_encoder) is text_encoder
    patched = inner.forward
    assert patched("value", mm_token_type_ids=[0], keep=True) == (
        "value",
        {"keep": True},
    )
    assert transformers_compat.patch_text_encoder(text_encoder) is text_encoder
    assert inner.forward is patched

    assert transformers_compat.restore_instance_patches(text_encoder=text_encoder)
    assert "forward" not in vars(inner)
    assert not hasattr(inner, "_h3_mm_token_type_ids_patched")
    assert not hasattr(inner, "_h3_original_forward")
    assert not hasattr(inner, "_h3_text_encoder_patch_state")
    assert inner.forward("value", keep=True) == ("value", {"keep": True})


def test_instance_restore_preserves_external_method_replacements():
    processor = _LegacyProcessor()
    transformers_compat.patch_processor(processor)
    external_processor_method = lambda values: ["external"]  # noqa: E731
    processor.create_mm_token_type_ids = external_processor_method

    text_encoder = _TextEncoder()
    inner = text_encoder.model
    transformers_compat.patch_text_encoder(text_encoder)
    external_forward = lambda *args, **kwargs: "external"  # noqa: E731
    inner.forward = external_forward

    transformers_compat.restore_instance_patches(
        processor=processor, text_encoder=text_encoder
    )
    assert processor.create_mm_token_type_ids is external_processor_method
    assert inner.forward is external_forward
    assert not hasattr(processor, "_h3_processor_patch_state")
    assert not hasattr(inner, "_h3_text_encoder_patch_state")


def test_repatch_preserves_external_forward():
    text_encoder = _TextEncoder()
    inner = text_encoder.model
    transformers_compat.patch_text_encoder(text_encoder)
    external_forward = lambda *args, **kwargs: (args, kwargs)  # noqa: E731
    inner.forward = external_forward

    transformers_compat.patch_text_encoder(text_encoder)
    assert inner.forward is external_forward
    assert inner.forward("value", mm_token_type_ids=[0], keep=True) == (
        ("value",),
        {"mm_token_type_ids": [0], "keep": True},
    )
    assert not transformers_compat.restore_instance_patches(text_encoder=text_encoder)
    assert inner.forward is external_forward


def test_weak_registry_restores_all_live_instance_patches():
    processor = _LegacyProcessor()
    text_encoder = _TextEncoder()
    transformers_compat.patch_processor(processor)
    transformers_compat.patch_text_encoder(text_encoder)

    assert transformers_compat.restore_instance_patches()
    assert not hasattr(processor, "create_mm_token_type_ids")
    assert "forward" not in vars(text_encoder.model)


def test_text_encoder_with_native_argument_is_not_patched():
    class NativeTextModel:
        def forward(self, value, mm_token_type_ids=None, **kwargs):
            return value, mm_token_type_ids, kwargs

    text_encoder = _TextEncoder()
    text_encoder.model = NativeTextModel()

    assert transformers_compat.patch_text_encoder(text_encoder) is text_encoder
    assert "forward" not in vars(text_encoder.model)
    assert not hasattr(text_encoder.model, "_h3_text_encoder_patch_state")


def test_text_encoder_detects_argument_on_decorated_forward():
    text_encoder = _TextEncoder()

    @functools.wraps(_LegacyTextModel.forward)
    def decorated_forward(value, mm_token_type_ids=None, **kwargs):
        return value, mm_token_type_ids, kwargs

    text_encoder.model.forward = decorated_forward

    assert transformers_compat.patch_text_encoder(text_encoder) is text_encoder
    assert text_encoder.model.forward is decorated_forward
    assert not hasattr(text_encoder.model, "_h3_text_encoder_patch_state")


def test_patch_allocator_warmup_is_opt_in_and_idempotent(monkeypatch):
    original = lambda *args, **kwargs: "warm"  # noqa: E731
    modeling_utils = ModuleType("transformers.modeling_utils")
    modeling_utils.caching_allocator_warmup = original
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", modeling_utils)

    monkeypatch.delenv("H3_DISABLE_ALLOCATOR_WARMUP", raising=False)
    assert transformers_compat.patch_allocator_warmup() == "enabled"
    assert modeling_utils.caching_allocator_warmup is original

    monkeypatch.setenv("H3_DISABLE_ALLOCATOR_WARMUP", "1")
    assert transformers_compat.patch_allocator_warmup() == "disabled"
    assert modeling_utils.caching_allocator_warmup() is None
    patched = modeling_utils.caching_allocator_warmup
    assert transformers_compat.patch_allocator_warmup() == "disabled"
    assert modeling_utils.caching_allocator_warmup is patched

    assert transformers_compat.restore_allocator_warmup()
    assert modeling_utils.caching_allocator_warmup is original
    assert not hasattr(modeling_utils, "_h3_allocator_warmup_patch_state")


def test_allocator_restore_preserves_external_replacement(monkeypatch):
    original = lambda *args, **kwargs: "warm"  # noqa: E731
    external = lambda *args, **kwargs: "external"  # noqa: E731
    modeling_utils = ModuleType("transformers.modeling_utils")
    modeling_utils.caching_allocator_warmup = original
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", modeling_utils)
    monkeypatch.setenv("H3_DISABLE_ALLOCATOR_WARMUP", "1")

    transformers_compat.patch_allocator_warmup()
    modeling_utils.caching_allocator_warmup = external
    assert not transformers_compat.restore_allocator_warmup()
    assert modeling_utils.caching_allocator_warmup is external


def test_patch_finegrained_fp8_installs_transformers_55_globals(monkeypatch):
    functions = {
        name: object()
        for name in (
            "fp8_act_quant",
            "w8a8_fp8_matmul",
            "w8a8_fp8_matmul_batched",
            "w8a8_fp8_matmul_grouped",
        )
    }
    legacy = ModuleType("local_finegrained_fp8.legacy")
    act_quant = ModuleType("local_finegrained_fp8.legacy.act_quant")
    act_quant.fp8_act_quant = functions["fp8_act_quant"]
    matmul = ModuleType("local_finegrained_fp8.legacy.matmul")
    matmul.w8a8_fp8_matmul = functions["w8a8_fp8_matmul"]
    batched = ModuleType("local_finegrained_fp8.legacy.batched")
    batched.w8a8_fp8_matmul_batched = functions["w8a8_fp8_matmul_batched"]
    grouped = ModuleType("local_finegrained_fp8.legacy.grouped")
    grouped.w8a8_fp8_matmul_grouped = functions["w8a8_fp8_matmul_grouped"]

    finegrained_fp8 = ModuleType("transformers.integrations.finegrained_fp8")
    finegrained_fp8._triton_available = None
    finegrained_fp8._deepgemm_available = None
    for name in (
        "triton_fp8_matmul",
        "triton_fp8_act_quant",
        "triton_batched_fp8_matmul",
        "triton_grouped_fp8_matmul",
    ):
        setattr(finegrained_fp8, name, None)
    integrations = ModuleType("transformers.integrations")
    integrations.finegrained_fp8 = finegrained_fp8
    transformers = ModuleType("transformers")
    transformers.integrations = integrations

    modules = {
        "local_finegrained_fp8.legacy": legacy,
        "local_finegrained_fp8.legacy.act_quant": act_quant,
        "local_finegrained_fp8.legacy.matmul": matmul,
        "local_finegrained_fp8.legacy.batched": batched,
        "local_finegrained_fp8.legacy.grouped": grouped,
        "transformers": transformers,
        "transformers.integrations": integrations,
        "transformers.integrations.finegrained_fp8": finegrained_fp8,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", "1")

    assert transformers_compat.patch_finegrained_fp8() == "local_triton_55"

    assert finegrained_fp8._triton_available is True
    assert finegrained_fp8._deepgemm_available is False
    assert finegrained_fp8.triton_fp8_act_quant is functions["fp8_act_quant"]
    assert finegrained_fp8.triton_fp8_matmul is functions["w8a8_fp8_matmul"]
    assert (
        finegrained_fp8.triton_batched_fp8_matmul
        is functions["w8a8_fp8_matmul_batched"]
    )
    assert (
        finegrained_fp8.triton_grouped_fp8_matmul
        is functions["w8a8_fp8_matmul_grouped"]
    )

    assert transformers_compat.patch_finegrained_fp8() == "local_triton_55"

    external_matmul = object()
    finegrained_fp8.triton_fp8_matmul = external_matmul
    assert transformers_compat.patch_finegrained_fp8() == "local_triton_55"
    assert finegrained_fp8.triton_fp8_matmul is functions["w8a8_fp8_matmul"]

    assert transformers_compat.restore_finegrained_fp8()
    assert finegrained_fp8.triton_fp8_act_quant is None
    assert finegrained_fp8.triton_fp8_matmul is external_matmul
    assert finegrained_fp8._triton_available is None
    assert finegrained_fp8._deepgemm_available is None


def test_patch_finegrained_fp8_installs_transformers_58_loader(monkeypatch):
    functions = {
        name: object()
        for name in (
            "fp8_act_quant",
            "w8a8_fp8_matmul",
            "w8a8_fp8_matmul_batched",
            "w8a8_fp8_matmul_grouped",
        )
    }
    legacy = ModuleType("local_finegrained_fp8.legacy")
    act_quant = ModuleType("local_finegrained_fp8.legacy.act_quant")
    act_quant.fp8_act_quant = functions["fp8_act_quant"]
    matmul = ModuleType("local_finegrained_fp8.legacy.matmul")
    matmul.w8a8_fp8_matmul = functions["w8a8_fp8_matmul"]
    batched = ModuleType("local_finegrained_fp8.legacy.batched")
    batched.w8a8_fp8_matmul_batched = functions["w8a8_fp8_matmul_batched"]
    grouped = ModuleType("local_finegrained_fp8.legacy.grouped")
    grouped.w8a8_fp8_matmul_grouped = functions["w8a8_fp8_matmul_grouped"]

    class FineGrainedFP8:
        def __init__(
            self,
            *,
            fp8_matmul,
            fp8_act_quant,
            batched_fp8_matmul,
            grouped_fp8_matmul,
        ):
            self.fp8_matmul = fp8_matmul
            self.fp8_act_quant = fp8_act_quant
            self.batched_fp8_matmul = batched_fp8_matmul
            self.grouped_fp8_matmul = grouped_fp8_matmul

    finegrained_fp8 = ModuleType("transformers.integrations.finegrained_fp8")
    finegrained_fp8.FineGrainedFP8 = FineGrainedFP8
    finegrained_fp8._load_finegrained_fp8_kernel = lambda: "hub"
    finegrained_fp8.w8a8_fp8_matmul = object()
    integrations = ModuleType("transformers.integrations")
    integrations.finegrained_fp8 = finegrained_fp8
    transformers = ModuleType("transformers")
    transformers.integrations = integrations

    modules = {
        "local_finegrained_fp8.legacy": legacy,
        "local_finegrained_fp8.legacy.act_quant": act_quant,
        "local_finegrained_fp8.legacy.matmul": matmul,
        "local_finegrained_fp8.legacy.batched": batched,
        "local_finegrained_fp8.legacy.grouped": grouped,
        "transformers": transformers,
        "transformers.integrations": integrations,
        "transformers.integrations.finegrained_fp8": finegrained_fp8,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", "1")

    assert transformers_compat.patch_finegrained_fp8() == "local_triton_58"
    bundle = finegrained_fp8._load_finegrained_fp8_kernel()
    assert bundle.fp8_act_quant is functions["fp8_act_quant"]
    assert bundle.fp8_matmul is functions["w8a8_fp8_matmul"]
    assert bundle.batched_fp8_matmul is functions["w8a8_fp8_matmul_batched"]
    assert bundle.grouped_fp8_matmul is functions["w8a8_fp8_matmul_grouped"]
    assert finegrained_fp8.w8a8_fp8_matmul is functions["w8a8_fp8_matmul"]


@pytest.mark.parametrize("global_cache", [True, False])
def test_patch_finegrained_fp8_installs_modern_bundle(monkeypatch, global_cache):
    functions = {
        name: object()
        for name in ("matmul_2d", "matmul_batched", "matmul_grouped")
    }
    modern = ModuleType("local_finegrained_fp8")
    modern.matmul_2d = functions["matmul_2d"]
    modern.matmul_batched = functions["matmul_batched"]
    modern.matmul_grouped = functions["matmul_grouped"]

    class FineGrainedFP8:
        def __init__(self, *, matmul, batched_matmul, grouped_matmul):
            self.matmul = matmul
            self.batched_matmul = batched_matmul
            self.grouped_matmul = grouped_matmul

    finegrained_fp8 = ModuleType("transformers.integrations.finegrained_fp8")
    finegrained_fp8.FineGrainedFP8 = FineGrainedFP8
    if global_cache:
        finegrained_fp8._FINEGRAINED_FP8 = None
    original_loader = lambda: "hub"
    finegrained_fp8._load_finegrained_fp8_kernel = original_loader
    integrations = ModuleType("transformers.integrations")
    integrations.finegrained_fp8 = finegrained_fp8
    transformers = ModuleType("transformers")
    transformers.integrations = integrations

    modules = {
        "local_finegrained_fp8": modern,
        "transformers": transformers,
        "transformers.integrations": integrations,
        "transformers.integrations.finegrained_fp8": finegrained_fp8,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    assert transformers_compat.patch_finegrained_fp8() == "local_triton_v4"
    bundle = (finegrained_fp8._FINEGRAINED_FP8 if global_cache
              else finegrained_fp8._load_finegrained_fp8_kernel())
    assert bundle.matmul is functions["matmul_2d"]
    assert bundle.batched_matmul is functions["matmul_batched"]
    assert bundle.grouped_matmul is functions["matmul_grouped"]
    assert transformers_compat.patch_finegrained_fp8() == "local_triton_v4"
    assert transformers_compat.restore_finegrained_fp8()
    assert finegrained_fp8._load_finegrained_fp8_kernel is original_loader
    if global_cache:
        assert finegrained_fp8._FINEGRAINED_FP8 is None
    else:
        assert not hasattr(finegrained_fp8, "_FINEGRAINED_FP8")


def test_patch_finegrained_fp8_replaces_preloaded_transformers_515_bundle(monkeypatch):
    local = {
        name: object() for name in ("matmul_2d", "matmul_batched", "matmul_grouped")
    }
    modern = ModuleType("local_finegrained_fp8")
    modern.matmul_2d = local["matmul_2d"]
    modern.matmul_batched = local["matmul_batched"]
    modern.matmul_grouped = local["matmul_grouped"]

    class FineGrainedFP8:
        def __init__(self, *, matmul, batched_matmul, grouped_matmul):
            self.matmul = matmul
            self.batched_matmul = batched_matmul
            self.grouped_matmul = grouped_matmul

    hub = FineGrainedFP8(matmul=object(), batched_matmul=object(), grouped_matmul=object())
    finegrained_fp8 = ModuleType("transformers.integrations.finegrained_fp8")
    finegrained_fp8.FineGrainedFP8 = FineGrainedFP8
    finegrained_fp8._FINEGRAINED_FP8 = hub
    finegrained_fp8._load_finegrained_fp8_kernel = lambda: None
    integrations = ModuleType("transformers.integrations")
    integrations.finegrained_fp8 = finegrained_fp8
    transformers = ModuleType("transformers")
    transformers.integrations = integrations

    modules = {
        "local_finegrained_fp8": modern,
        "transformers": transformers,
        "transformers.integrations": integrations,
        "transformers.integrations.finegrained_fp8": finegrained_fp8,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    assert transformers_compat.patch_finegrained_fp8() == "local_triton_v4"
    assert finegrained_fp8._FINEGRAINED_FP8 is not hub
    assert finegrained_fp8._FINEGRAINED_FP8.matmul is local["matmul_2d"]


def test_real_finegrained_fp8_linear_offline(monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    integration = pytest.importorskip("transformers.integrations.finegrained_fp8")
    monkeypatch.setenv("TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    transformers_compat.patch_finegrained_fp8()
    try:
        generator = torch.Generator(device="cuda").manual_seed(42)
        layer = integration.FP8Linear(256, 256, block_size=(128, 128)).cuda()
        with torch.no_grad():
            weight = torch.randn(256, 256, device="cuda", generator=generator)
            layer.weight.copy_(weight.to(layer.weight.dtype))
            layer.weight_scale_inv.fill_(1)
            value = torch.randn(1, 17, 256, device="cuda", dtype=torch.bfloat16, generator=generator)
            actual = layer(value)
            reference = torch.nn.functional.linear(value.float(), layer.weight.float())
        assert actual.shape == reference.shape
        assert torch.isfinite(actual).all()
        assert (actual.float() - reference).norm() / reference.norm() < 0.05
    finally:
        transformers_compat.restore_finegrained_fp8()
