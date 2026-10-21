#!/usr/bin/env python3
"""Registered MiniMax-H3 *gb10* runtime entrypoint — one GB10, FP8, no context parallelism.

Same contract as the baseline and optimized entrypoints: the config manifest sets OUT_DIR
and the H3_* switches, this builds the pipeline and writes `benchmark.json` and `out.mp4` into
it. Every acceleration is env-gated, so `run_minimax_h3_gpu.sh` is identical for every
config in `config/minimax_h3_gb10_*.toml`.

What this variant is
--------------------
MiniMax-H3 t2va on one GB10, from the pruned FP8 checkpoint.

Sol-Engine's own baseline (`models/minimax_h3/GB200/gpu_infer.py`) puts the 33B DiT, the
Qwen3-VL-32B conditioner and both VAEs resident on one 186 GB GB200 and calls that the
denominator. None of that fits here: GB10 has 128 GB of *unified* memory shared with the
host, and the released BF16 weights alone are 134 GB. Two substitutions close the gap and
neither changes the pipeline's structure:

* the DiT is the ComfyUI `pruned_fp8_scaled` checkpoint, loaded through `build.py` — 23.3 GB,
  with 250 of its 300 quantised linears running on the FP8 tensor cores;
* the conditioner is `Qwen/Qwen3-VL-32B-Instruct-FP8` — 33 GB against 62 GB, and structurally
  the same model (H3's `text_encoder/config.json` is stock Qwen3-VL-32B-Instruct: 64 layers,
  hidden 5120, vocab 151936, `rope_theta` 5e6).

Context parallelism, the first stage of Sol-Engine's acceleration line, has nothing to do on
one card. Sol-Attn runs through the released entry point with nothing bypassed and nothing
forked: `_backend_for_arch` has no CuTe kernel for GB10's SM121 and returns "triton", while
the exact multimodal prefix is passed through `sink_tokens` from each request's packed layout.
Both were true only after the `sol-engine` head moved; `fusion_install.load_sol_attn` records
what the port carried before and the measurement that let it be deleted.

The timing block is kept as Sol-Engine writes it — CUDA events around the components, one
warmup request excluded — so the numbers stay comparable in shape, though not in value.
"""

from __future__ import annotations

import paths

paths.setup()

import argparse
import atexit
import functools
import glob
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from contextlib import contextmanager

os.environ.setdefault("HF_HOME", str(paths.HF_CACHE))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


import torch

from model_index_overlay import localise_component_index as _localise_component_index

try:
    import triton
except ImportError:
    triton = None

QWEN_FP8 = "Qwen/Qwen3-VL-32B-Instruct-FP8"


def _restore_process_patches() -> None:
    """Best-effort cleanup for module/class hooks owned by this one-shot runner."""
    model = getattr(_restore_process_patches, "_model", None)
    vae = getattr(_restore_process_patches, "_vae", None)
    processor = getattr(_restore_process_patches, "_processor", None)
    text_encoder = getattr(_restore_process_patches, "_text_encoder", None)
    try:
        import cache_line

        if model is not None:
            # The cache wrapper is installed outside the Sol-Attn layout wrapper. Remove it
            # first so its saved forward points at the live base method, not a stale layout hook.
            cache_line.remove_cache(model)
    except (ImportError, AttributeError, RuntimeError):
        pass
    try:
        import fusion_install

        fusion_install.restore_fused_patches()
        if model is not None:
            fusion_install.uninstall_sol_attn(model)
    except (ImportError, AttributeError, RuntimeError):
        pass
    try:
        import vae_shard

        if vae is not None:
            vae_shard.unpatch_batched_tiles(vae)
    except (ImportError, AttributeError, RuntimeError):
        pass
    try:
        import transformers_compat

        transformers_compat.restore_transformers_patches(
            processor=processor, text_encoder=text_encoder
        )
    except (ImportError, AttributeError, RuntimeError):
        pass


atexit.register(_restore_process_patches)


H3_SNAPSHOT = paths.h3_snapshot()
QWEN_FP8_PATH = paths.qwen_fp8()
_QWEN_QUANT = json.load(open(f"{QWEN_FP8_PATH}/config.json"))["quantization_config"]
DIT_CHECKPOINT = paths.dit_checkpoint()


class GpuTimer:
    """Accumulates device time for one component with CUDA events.

    Events are recorded on the component's own stream and synchronized once, at `total()`, so
    instrumenting every DiT evaluation does not serialize the pipeline.
    """

    def __init__(self) -> None:
        self.spans: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    @contextmanager
    def span(self, device: torch.device):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(device):
            start.record()
        try:
            yield
        finally:
            with torch.cuda.device(device):
                end.record()
            self.spans.append((start, end))

    def total(self) -> float | None:
        # None, not 0.0: a timer that never fired means the wrapper missed its call site, and
        # that must not read as "this component costs nothing".
        if not self.spans:
            return None
        torch.cuda.synchronize()
        return sum(s.elapsed_time(e) for s, e in self.spans) / 1000.0

    def __len__(self) -> int:
        return len(self.spans)


def _module_device(module: torch.nn.Module) -> torch.device:
    """Return the device on which a module's work is expected to run.

    Most instrumented modules expose a parameter, but rotary/utility modules can be
    parameter-free and carry only a buffer (or no registered tensors at all).  Keep the
    fallback on the current CUDA device because this entrypoint is a GPU benchmark.
    """
    for tensor in module.parameters():
        return tensor.device
    for tensor in module.buffers():
        return tensor.device
    return torch.device("cuda", torch.cuda.current_device())


def _uniform_floating_parameter_dtype(module: torch.nn.Module, name: str) -> torch.dtype:
    """Return one observed floating-point parameter dtype for a loaded component."""
    dtypes = {parameter.dtype for parameter in module.parameters() if parameter.is_floating_point()}
    if not dtypes:
        raise RuntimeError(f"{name} has no floating-point parameters")
    if len(dtypes) != 1:
        raise RuntimeError(f"{name} has mixed floating-point parameter dtypes: {sorted(map(str, dtypes))}")
    return next(iter(dtypes))


def instrument(module: torch.nn.Module, timer: GpuTimer, method: str = "forward"):
    """Install a timing wrapper once and return the method it replaced.

    Modules are reused when profiles are run in one interpreter.  Keep the wrapper and its
    original in an instance-local registry so a second instrumentation call updates the timer
    instead of wrapping the old wrapper again.  The returned original is accepted by
    :func:`restore_instrumentation` for callers that want an explicit ownership check.
    """
    registry = getattr(module, "_h3_instrumentation", None)
    if registry is None:
        registry = {}
        module._h3_instrumentation = registry

    current = getattr(module, method)
    existing = registry.get(method)
    if existing is not None and current is existing["wrapper"]:
        existing["timer"] = timer
        return existing["original"]

    # The method may have been replaced by another lifecycle hook since our last install.  Do
    # not retain a stale registry entry or accidentally restore over the newer owner.
    if existing is not None:
        registry.pop(method, None)
    original = current
    state = {"original": original, "timer": timer}

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        device = _module_device(module)
        with state["timer"].span(device):
            return original(*args, **kwargs)

    state["wrapper"] = wrapped
    registry[method] = state
    setattr(module, method, wrapped)
    return original


def restore_instrumentation(
    module: torch.nn.Module, method: str = "forward", original=None
) -> bool:
    """Restore one timing wrapper, without clobbering a newer replacement.

    ``original`` is compared by bound-method identity when supplied; this lets a caller that
    owns an older wrapper safely clean up after an exception while another profile has already
    installed a new method.
    """
    registry = getattr(module, "_h3_instrumentation", None)
    if not registry or method not in registry:
        return False
    state = registry[method]
    current = getattr(module, method, None)
    if current is not state["wrapper"]:
        registry.pop(method, None)
        if not registry:
            try:
                delattr(module, "_h3_instrumentation")
            except AttributeError:
                pass
        return False
    if original is not None and not _same_bound_method(original, state["original"]):
        return False
    setattr(module, method, state["original"])
    registry.pop(method, None)
    if not registry:
        try:
            delattr(module, "_h3_instrumentation")
        except AttributeError:
            pass
    return True


def _same_bound_method(first, second) -> bool:
    """Compare bound methods without relying on ephemeral object identity."""
    if first is second:
        return True
    sentinel = object()
    first_self = getattr(first, "__self__", sentinel)
    second_self = getattr(second, "__self__", sentinel)
    first_func = getattr(first, "__func__", sentinel)
    second_func = getattr(second, "__func__", sentinel)
    if any(value is sentinel for value in (first_self, second_self, first_func, second_func)):
        return False
    return first_self is second_self and first_func is second_func


def _round(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(value, digits)


def _request_s_median(requests: list[float]) -> float:
    """Return the statistical median for a non-empty measured-request sample."""
    if not requests:
        raise ValueError("at least one measured request is required")
    return float(statistics.median(requests))


def _unaccounted_s(request_s_median: float, accounted_s: float, request_count: int) -> float:
    """Estimate per-request time outside the component CUDA timers."""
    if int(request_count) < 1:
        raise ValueError("at least one measured request is required")
    return round(float(request_s_median) - float(accounted_s) / int(request_count), 3)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_provenance(path: str) -> dict:
    resolved = os.path.realpath(path)
    stat = os.stat(resolved)
    verify_digest = _flag("H3_VERIFY_DIT_SHA256") or bool(
        os.environ.get("H3_EXPECTED_DIT_SHA256")
    )
    digest = _sha256(resolved) if verify_digest else None
    expected = os.environ.get("H3_EXPECTED_DIT_SHA256")
    if expected and digest != expected:
        raise RuntimeError(
            f"DiT checkpoint SHA-256 mismatch: {resolved} {digest} != {expected}"
        )
    return {
        "path": resolved,
        "size_bytes": stat.st_size,
        "sha256": digest,
    }


def _system_memory() -> dict:
    """Read host memory counters, which are the relevant capacity on UMA parts."""
    values = {}
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                key, raw = line.split(":", 1)
                if key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                    values[key] = int(raw.strip().split()[0]) // 1024
    except (OSError, ValueError):
        return {}
    return {f"{key.lower()}_mib": value for key, value in values.items()}


def build_pipeline(args, **transformer_options):
    from diffusers import ComponentsManager, MiniMaxH3Blocks
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    import fusion_install
    fusion_install.restore_fused_patches()
    from build import build_pruned_fp8_transformer
    from transformers_compat import (
        patch_allocator_warmup,
        patch_finegrained_fp8,
        patch_processor,
        patch_text_encoder,
        restore_transformers_patches,
    )

    device = f"cuda:{args.device}"
    t0 = time.perf_counter()

    load_order = os.environ.get("H3_LOAD_ORDER", "dit_first").strip().lower()
    if load_order not in {"dit_first", "qwen_first"}:
        raise ValueError("H3_LOAD_ORDER must be 'dit_first' or 'qwen_first'")

    dit_config = json.load(open(f"{H3_SNAPSHOT}/transformer/config.json"))

    def _load_dit():
        model, model_info = build_pruned_fp8_transformer(
            DIT_CHECKPOINT, dit_config, device=device, **transformer_options
        )
        print(
            f"[h3] DiT: {torch.cuda.memory_allocated(args.device) / 2**30:.2f} GiB, "
            f"{model_info['quantized_layers']} quantised linears",
            flush=True,
        )
        return model, model_info

    # On Thor, loading the DiT first leaves only ~100 GiB of the shared UMA pool for the Qwen
    # shard loader.  The loader's transient CPU tensors then push the host below earlyoom's
    # limit even though the final resident models fit.  Keep the historical order for GB10,
    # while allowing the Thor wrapper to load Qwen into its final device before the DiT.
    transformer = None
    info = None
    # A caller may reuse this entrypoint in one interpreter with a different profile. Remove
    # only the previous shim-owned globals before selecting the new load policy.
    restore_transformers_patches()
    if load_order == "dit_first":
        transformer, info = _load_dit()

    # The FP8 conditioner's exclusion list has to be spelled the way transformers reads it, and
    # it has to be edited *into the config*: Qwen ships the vLLM/compressed-tensors key
    # `ignored_layers`, transformers' `FineGrainedFP8Config` reads `modules_to_not_convert`, and
    # passing a corrected `quantization_config=` to `from_pretrained` does not help — a checkpoint
    # that carries its own config wins, with only a `UserWarning` to say so.
    #
    # Left uncorrected, transformers converts the whole model, including the 27-block vision
    # tower and `lm_head` that the checkpoint deliberately left in BF16. Those layers then find
    # no `weight_scale_inv` to load, silently stay on meta — `missing_keys` is empty, because
    # nothing was missing from the file — and `dispatch_model` dies moving them to the device.
    # Only the 64 language-model layers are quantised: 448 scales = 64 x 7 projections.
    qwen_config = AutoConfig.from_pretrained(QWEN_FP8_PATH)
    qwen_config.quantization_config["modules_to_not_convert"] = _QWEN_QUANT["ignored_layers"]

    # Install the offline FP8 bundle before constructing the quantised model.  Current
    # Transformers releases load the bundle lazily on first forward, but doing this first keeps
    # the launcher correct if a future release validates or materialises a quantised layer during
    # `from_pretrained` itself.
    allocator_warmup = patch_allocator_warmup()
    fp8_backend = patch_finegrained_fp8()

    # `device_map` loads the conditioner straight onto the card instead of materialising it on
    # the host first, which on a unified-memory part would double-count against the same pool.
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        QWEN_FP8_PATH, config=qwen_config, dtype=torch.bfloat16, device_map={"": device}
    )
    print(f"[h3] + conditioner: {torch.cuda.memory_allocated(args.device) / 2**30:.2f} GiB",
          flush=True)

    if load_order == "qwen_first":
        # Release shard-loader temporaries and unused allocator blocks before materialising the
        # 23 GiB DiT.  This matters on UMA because CPU RSS and CUDA allocations draw from one pool.
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        transformer, info = _load_dit()

    pipeline_root = _localise_component_index(H3_SNAPSHOT)
    if pipeline_root != H3_SNAPSHOT:
        print(f"[h3] component index localised in overlay {pipeline_root}", flush=True)
    manager = ComponentsManager()
    # The shared H3 index may contain both transformer partitions. Resolve the text-to-video/audio
    # workflow to an explicit component list instead of relying on the newer ``workflow=`` loader
    # keyword, which is absent from the pinned latent-capable Diffusers snapshot.
    blocks = MiniMaxH3Blocks()
    workflow_components = {spec.name for spec in blocks.get_workflow("t2va").expected_components}
    if "transformer" not in workflow_components or "transformer_ref" in workflow_components:
        raise RuntimeError(
            f"unexpected MiniMax-H3 t2va component contract: {sorted(workflow_components)}"
        )
    pipe = blocks.init_pipeline(pipeline_root, components_manager=manager)
    component_specs = getattr(pipe, "_component_specs", {})
    if "transformer" not in component_specs:
        raise RuntimeError(
            "t2va loader did not expose transformer; refusing to allocate the Ref2VA DiT"
        )
    pipe.update_components(transformer=transformer, text_encoder=text_encoder)
    components_to_load = sorted(
        name
        for name in workflow_components
        if name in component_specs
        and component_specs[name].default_creation_method == "from_pretrained"
        and getattr(pipe, name, None) is None
    )
    # The pinned H3 decoder runs the video VAE under FP16 autocast while keeping both VAE
    # parameter sets in FP32. Preserve the lighter historical BF16 load on non-Thor profiles.
    vae_load_dtype = torch.float32 if _flag("H3_REQUIRE_SM110") else torch.bfloat16
    component_dtypes = {
        "vae": vae_load_dtype,
        "audio_vae": vae_load_dtype,
        "default": torch.bfloat16,
    }
    pipe.load_components(names=components_to_load, dtype=component_dtypes)
    if getattr(pipe, "transformer_ref", None) is not None:
        raise RuntimeError("t2va loader materialised transformer_ref")

    # The branch targets transformers 5.x; this environment is 4.57. See `transformers_compat`.
    patch_processor(pipe.processor)
    patch_text_encoder(pipe.text_encoder)

    # Only the two VAEs and the tokenizers are left to place; the two big models are already
    # on the card and `pipe.to(...)` would walk them again.
    pipe.vae.to(device)
    pipe.audio_vae.to(device)
    torch.cuda.synchronize()

    vae_weight_dtype = _uniform_floating_parameter_dtype(pipe.vae, "video VAE")
    audio_vae_weight_dtype = _uniform_floating_parameter_dtype(pipe.audio_vae, "audio VAE")
    if _flag("H3_REQUIRE_SM110") and (
        vae_weight_dtype != torch.float32 or audio_vae_weight_dtype != torch.float32
    ):
        raise RuntimeError(
            "Thor requires FP32 VAE weights, got "
            f"video={vae_weight_dtype}, audio={audio_vae_weight_dtype}"
        )

    load_s = time.perf_counter() - t0
    print(f"[h3] all components: {torch.cuda.memory_allocated(args.device) / 2**30:.2f} GiB",
          flush=True)
    return pipe, {
        "load_s": load_s,
        "device": device,
        "qwen_fp8_backend": fp8_backend,
        "allocator_warmup": allocator_warmup,
        "load_order": load_order,
        "vae_weight_dtype": str(vae_weight_dtype),
        "audio_vae_weight_dtype": str(audio_vae_weight_dtype),
        "kernel_fusion_installation": {
            "fused_qkv": bool(info["fused_qkv"]),
            "fused_qkv_modules": int(info["fused_qkv_modules"]),
            "quantizer": str(info["quantizer"]),
            "fused_adaln_blocks": int(info["fused_adaln_blocks"]),
            "fused_rope": bool(info["fused_rope"]),
            "fused_swiglu": bool(info["fused_swiglu"]),
        },
    }


# Every acceleration is opt-in through the environment, so one entrypoint serves every
# config in `config/minimax_h3_gb10_*.toml` and the run script is identical for all of
# them. This mirrors `optimized/gpu_infer.py`, which gates its Ulysses and cache line the same
# way; what differs is only which techniques exist on a single card.
def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) not in ("", "0", "false", "False")


def _optional_float(name: str) -> float | None:
    raw = os.environ.get(name, "")
    return float(raw) if raw not in ("", "none", "None") else None


_SOL_ATTN_THRESH_ENV_NAMES = (
    "H3_SOL_ATTN_THRESH_TYPE",
    "H3_SOL_THRESH_TYPE",
    "SOL_ATTN_THRESH_TYPE",
)


def _sol_attn_thresh_type() -> str:
    """Resolve the Thor and upstream Sol-Attn threshold variable names.

    Thor's profile uses ``H3_SOL_ATTN_THRESH_TYPE``.  The published GB200
    profile and the public Sol-Attn launcher use the shorter
    ``H3_SOL_THRESH_TYPE`` and ``SOL_ATTN_THRESH_TYPE`` names respectively.
    Accept all three so a profile can be reused, but reject invalid or
    conflicting aliases instead of silently selecting one.
    """
    configured = []
    for name in _SOL_ATTN_THRESH_ENV_NAMES:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            continue
        value = raw.strip().lower()
        if value not in ("diag", "exact"):
            names = ", ".join(_SOL_ATTN_THRESH_ENV_NAMES)
            raise ValueError(
                f"{name} must be 'diag' or 'exact' (supported aliases: {names}), "
                f"got {raw!r}"
            )
        configured.append((name, value))
    if not configured:
        return "diag"
    values = {value for _name, value in configured}
    if len(values) > 1:
        detail = ", ".join(f"{name}={value!r}" for name, value in configured)
        raise ValueError(f"conflicting Sol-Attn threshold aliases: {detail}")
    # The tuple is ordered canonical-first; equal aliases therefore retain the
    # Thor spelling in diagnostics and metadata without changing the value.
    return configured[0][1]


def _step_counts(requested_steps: int) -> tuple[int, int]:
    """Return ``(scheduler_points, denoise_evals)`` for a user step request.

    MiniMax-H3's scheduler treats ``num_inference_steps`` as sigma *points* and
    evaluates the transformer one fewer time.  It also requires at least two
    points.  Keep the existing meaning for requests above one, while making a
    requested one-step bring-up run represent exactly one transformer call.
    """
    requested_steps = int(requested_steps)
    if requested_steps < 1:
        raise ValueError(f"steps must be >= 1, got {requested_steps}")
    scheduler_steps = max(2, requested_steps)
    return scheduler_steps, scheduler_steps - 1


def _validate_request_counts(warmup_requests: int, measure_requests: int) -> None:
    """Reject request counts that would make benchmark aggregates undefined."""
    if int(warmup_requests) < 0:
        raise ValueError("--warmup-requests must be >= 0")
    if int(measure_requests) < 1:
        raise ValueError("--measure-requests must be >= 1")


def _sol_sparse_opportunity(
    denoise_steps: int, dense_first_steps: int, dense_last_steps: int = 0
) -> bool:
    """Whether the step policy leaves any denoising step eligible for Sol-Attn.

    A run can legitimately have zero sparse calls when every step is inside the
    configured dense prefix/suffix (notably a one-eval bring-up with two dense
    first steps).  Strict dispatch validation must distinguish that case from a
    requested sparse backend that silently never ran.
    """
    denoise_steps = int(denoise_steps)
    dense_first_steps = max(0, int(dense_first_steps))
    dense_last_steps = max(0, int(dense_last_steps))
    for step in range(max(0, denoise_steps)):
        if step < dense_first_steps:
            continue
        if dense_last_steps and step >= denoise_steps - dense_last_steps:
            continue
        return True
    return False


def lossless_kernels() -> dict:
    """The bit-identical fusions. Verified against `dit_reference/`, not asserted."""
    if not _flag("H3_KERNEL_FUSIONS", "1"):
        return {}
    return dict(fuse_qkv=True, quantizer="triton", fuse_adaln=True,
                fuse_rope=True, fuse_swiglu=True)


def apply_approximations(pipe, steps: int) -> dict:
    """Sol-Attn and the cache: the two techniques that change the output.

    Rebuilt per request rather than installed once. The Sol-Attn dispatch carries a call
    counter that the step-indexed policy reads, and the cache carries the previous sample's
    residual; either leaking across requests would make the second sample depend on the first.
    """
    import cache_line
    import fusion_install

    applied = {}
    # Remove every process-global hook before selecting the next profile. This makes repeated
    # baseline/kernel/Sol runs in one interpreter deterministic and prevents a stale wrapper from
    # observing a dispatch that no longer owns the model.
    cache_line.remove_cache(pipe.transformer)
    fusion_install.uninstall_sol_attn(pipe.transformer)
    tau = _optional_float("H3_SOL_ATTN_TAU")
    if tau is not None:
        thresh_type = _sol_attn_thresh_type()
        dense_blocks = int(os.environ.get("H3_SOL_ATTN_DENSE_BLOCKS", 2))
        dense_first_steps = int(os.environ.get("H3_SOL_ATTN_DENSE_FIRST_STEPS", 10))
        dense_last_steps = int(os.environ.get("H3_SOL_ATTN_DENSE_LAST_STEPS", 0))
        dispatch = fusion_install.patch_sol_attn(
            pipe.transformer,
            tau=tau,
            thresh_type=thresh_type,
            dense_blocks=dense_blocks,
            dense_first_steps=dense_first_steps,
            dense_last_steps=dense_last_steps,
            num_steps=steps,
        )
        dispatch.reset()
        applied["sol_attn_tau"] = tau
        applied["sol_attn_thresh_type"] = dispatch.thresh_type
        applied["sol_attn_backend"] = dispatch.backend
        applied["sol_attn_dense_blocks"] = dense_blocks
        applied["sol_attn_dense_first_steps"] = dense_first_steps
        applied["sol_attn_dense_last_steps"] = dense_last_steps

    threshold = _optional_float("H3_CACHE_THRESHOLD")
    if threshold is not None:
        cache_line.apply_cache(pipe.transformer, threshold=threshold)
        cache_line.reset_cache(pipe.transformer)
        applied["cache_threshold"] = threshold

    batch = int(os.environ.get("H3_VAE_TILE_BATCH", 0))
    import vae_shard

    vae_shard.unpatch_batched_tiles(pipe.vae)
    if batch:
        vae_shard.patch_batched_tiles(pipe.vae, batch=batch)
        applied["vae_tile_batch"] = batch
    return applied


def reset_request_state(pipe) -> None:
    """Reset request-scoped sparse/cache state before every pipeline call."""
    from diffusers.models.transformers import transformer_minimax_h3 as h3

    dispatch = h3.dispatch_attention_fn
    if hasattr(dispatch, "reset"):
        dispatch.reset()
    if getattr(pipe.transformer, "is_cache_enabled", False):
        import cache_line

        cache_line.reset_cache(pipe.transformer)


def _sol_attn_stats() -> dict:
    from diffusers.models.transformers import transformer_minimax_h3 as h3

    dispatch = h3.dispatch_attention_fn
    return dispatch.stats() if hasattr(dispatch, "stats") else {}


_SOL_ATTN_SUM_KEYS = ("call", "dense_calls", "prefix_dense_calls", "sparse_calls")


def _accumulate_sol_attn_stats(total: dict, sample: dict) -> dict:
    """Accumulate per-request Sol-Attn counters while retaining latest layout metadata."""
    if not sample:
        return total
    if not total:
        total.update({
            key: value
            for key, value in sample.items()
            if key not in _SOL_ATTN_SUM_KEYS and key != "request_count"
        })
        for key in _SOL_ATTN_SUM_KEYS:
            if key in sample:
                total[key] = 0
        total["request_count"] = 0
    for key in _SOL_ATTN_SUM_KEYS:
        if key in sample:
            total[key] = int(total.get(key, 0)) + int(sample[key])
    for key, value in sample.items():
        if key not in _SOL_ATTN_SUM_KEYS and key != "request_count":
            total[key] = value
    total["request_count"] = int(total.get("request_count", 0)) + 1
    return total


def run_request(pipe, args, seed: int):
    generator = torch.Generator().manual_seed(seed)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    state = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.scheduler_steps,
        output_type=args.output_type,
        generator=generator,
    )
    torch.cuda.synchronize()
    return state, time.perf_counter() - t0


def video_uint8_chunks(video):
    """Convert FCHW pipeline output in 16-frame chunks for the native encoder."""
    for start in range(0, video.shape[0], 16):
        chunk = video[start : start + 16].permute(0, 2, 3, 1)
        yield chunk.mul(255).round().clamp(0, 255).to(torch.uint8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Defaults are a bring-up cell, not the published one: GB10 has ~1/30th of a GB200's
    # memory bandwidth, so the official 1344x768/124f/50-step cell is a long run to discover
    # a wiring bug in. Pass --official for it.
    #
    # Only the canvas and the step count shrink. The frame count cannot: the pipeline accepts
    # `17 * n + 5` frames between 5 and 15 seconds at 24 fps, so 124 (n=7) is both the floor
    # and what the published cell uses. `height` / `width` must be multiples of 32.
    parser.add_argument("--height", type=int, default=int(os.environ.get("H3_HEIGHT", 256)))
    parser.add_argument("--width", type=int, default=int(os.environ.get("H3_WIDTH", 448)))
    parser.add_argument("--num-frames", type=int, default=int(os.environ.get("H3_NUM_FRAMES", 124)))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("H3_STEPS", 8)))
    parser.add_argument("--official", action="store_true",
                        help="the published cell: 1344x768, 124 frames, 50 steps")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("H3_SEED", 0)))
    parser.add_argument("--warmup-requests", type=int, default=int(os.environ.get("H3_WARMUP", 0)))
    parser.add_argument("--measure-requests", type=int, default=int(os.environ.get("H3_MEASURE", 1)))
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--prompt", default=os.environ.get("H3_PROMPT") or None)
    parser.add_argument("--prompt-file", default=os.environ.get("H3_PROMPT_FILE", str(paths.PROMPT_FILE)))
    parser.add_argument("--output-dir", default=os.environ.get("H3_OUTPUT_DIR", "outputs"))
    parser.add_argument("--tag", default=os.environ.get("H3_TAG", "gb10"))
    parser.add_argument("--attention-backend", default=os.environ.get("H3_ATTENTION_BACKEND") or None)
    parser.add_argument(
        "--output-type",
        choices=("pil", "np", "pt", "latent"),
        default=os.environ.get("H3_OUTPUT_TYPE", "pil"),
    )
    parser.add_argument("--no-export", action="store_true")
    args = parser.parse_args()

    if args.official:
        args.height, args.width, args.num_frames, args.steps = 768, 1344, 124, 50

    capability = tuple(torch.cuda.get_device_capability(args.device))
    if _flag("H3_REQUIRE_SM110"):
        cuda_version = torch.version.cuda or ""
        if capability != (11, 0):
            raise RuntimeError(
                f"Thor profile requires SM110, got SM{capability[0]}{capability[1]}"
            )
        if not cuda_version.startswith("13."):
            raise RuntimeError(f"Thor profile requires CUDA 13.x, got {cuda_version!r}")
        autocast_contract = {
            "H3_VIDEO_VAE_AUTOCAST_DTYPE": "torch.float16",
            "H3_AUDIO_VAE_AUTOCAST_DTYPE": "disabled",
        }
        mismatched = {
            name: os.environ.get(name)
            for name, expected in autocast_contract.items()
            if os.environ.get(name) != expected
        }
        if mismatched:
            raise RuntimeError(f"Thor VAE autocast contract is missing or invalid: {mismatched}")

    try:
        _validate_request_counts(args.warmup_requests, args.measure_requests)
    except ValueError as exc:
        parser.error(str(exc))

    # Validate configuration before hashing or allocating the large checkpoints.  The public
    # Sol-Attn interface repeats this check at kernel entry, but a run whose policy keeps every
    # step dense would otherwise accept a typo silently.
    if _optional_float("H3_SOL_ATTN_TAU") is not None:
        try:
            _sol_attn_thresh_type()
        except ValueError as exc:
            parser.error(str(exc))

    args.requested_steps = args.steps
    args.scheduler_steps, args.denoise_steps = _step_counts(args.requested_steps)

    # Resolve and, for strict Thor launches, hash the exact file before allocating
    # either large model.  The wrapper also pins this path during its asset gate;
    # recording the digest here proves which artifact the shared assembler used.
    dit_checkpoint = _checkpoint_provenance(DIT_CHECKPOINT)

    if args.prompt is None:
        with open(args.prompt_file) as handle:
            args.prompt = json.load(handle)["prompt"]

    os.makedirs(args.output_dir, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()

    kernel_options = lossless_kernels()
    pipe, placement = build_pipeline(args, **kernel_options)
    _restore_process_patches._model = pipe.transformer
    _restore_process_patches._vae = pipe.vae
    _restore_process_patches._processor = pipe.processor
    _restore_process_patches._text_encoder = pipe.text_encoder
    approximations = apply_approximations(pipe, args.denoise_steps)
    if approximations:
        print(f"[h3] approximations: {approximations}", flush=True)
    if args.attention_backend:
        pipe.transformer.set_attention_backend(args.attention_backend)
    print(f"[h3] loaded in {placement['load_s']:.2f}s on {placement['device']}  "
          f"{args.width}x{args.height} {args.num_frames}f "
          f"{args.requested_steps} requested steps ({args.scheduler_steps} scheduler points, "
          f"{args.denoise_steps} denoise evals)", flush=True)

    for i in range(args.warmup_requests):
        reset_request_state(pipe)
        _, warm_s = run_request(pipe, args, seed=args.seed)
        print(f"[h3] warmup {i + 1}/{args.warmup_requests}: {warm_s:.2f}s", flush=True)

    import vae_shard
    import fusions

    # Warmup decode work is excluded from all other timing counters; keep tile
    # execution counters on the same boundary.
    vae_shard.reset_tile_stats(pipe.vae)
    fusions.reset_fusion_stats()

    cache_counter = None
    if getattr(pipe.transformer, "is_cache_enabled", False):
        import cache_line

        cache_counter = cache_line.SkipCounter(pipe.transformer)

    timers = {k: GpuTimer() for k in ("denoise", "decode", "audio_decode", "encode")}
    instrumented = []
    requests, state = [], None
    sol_attn_stats = {}
    try:
        def _install(module, timer, method="forward"):
            original = instrument(module, timer, method=method)
            instrumented.append((module, method, original))

        _install(pipe.transformer, timers["denoise"])
        _install(pipe.vae, timers["decode"], method="decode")
        _install(pipe.audio_vae, timers["audio_decode"], method="decode")
        # The encoder block calls `text_encoder.model(...)` directly to skip the LM head, so
        # wrapping `text_encoder.forward` would never fire.
        _install(pipe.text_encoder.model, timers["encode"])
        torch.cuda.reset_peak_memory_stats()
        for i in range(args.measure_requests):
            reset_request_state(pipe)
            state, request_s = run_request(pipe, args, seed=args.seed)
            requests.append(request_s)
            _accumulate_sol_attn_stats(sol_attn_stats, _sol_attn_stats())
            print(f"[h3] request {i + 1}/{args.measure_requests}: {request_s:.3f}s", flush=True)
    finally:
        for module, method, original in reversed(instrumented):
            restore_instrumentation(module, method, original)
        if cache_counter is not None:
            cache_counter.restore()

    sol_tau = _optional_float("H3_SOL_ATTN_TAU")
    dense_first_steps = None
    dense_last_steps = None
    sparse_opportunity = False
    if sol_tau is not None:
        dense_first_steps = int(os.environ.get("H3_SOL_ATTN_DENSE_FIRST_STEPS", 10))
        dense_last_steps = int(os.environ.get("H3_SOL_ATTN_DENSE_LAST_STEPS", 0))
        sparse_opportunity = _sol_sparse_opportunity(
            args.denoise_steps, dense_first_steps, dense_last_steps
        )
    if sol_tau is not None and not sol_attn_stats.get("sparse_calls") and sparse_opportunity:
        message = f"Sol-Attn was requested but no sparse call ran: {sol_attn_stats}"
        if _flag("SOL_ATTN_STRICT"):
            raise RuntimeError(message)
        print(f"[h3] WARNING: {message}", flush=True)
    elif sol_tau is not None and not sparse_opportunity:
        print(
            f"[h3] Sol-Attn had no sparse opportunity ({args.denoise_steps} denoise evals, "
            f"dense_first={dense_first_steps}, dense_last={dense_last_steps})",
            flush=True,
        )

    kernel_fusion_hits = fusions.read_fusion_stats()
    if kernel_options:
        missing_fusion_hits = {
            name: count for name, count in kernel_fusion_hits.items() if count <= 0
        }
        if missing_fusion_hits:
            raise RuntimeError(
                "lossless kernel fusion was requested but did not execute: "
                f"{missing_fusion_hits}"
            )
    elif any(kernel_fusion_hits.values()):
        raise RuntimeError(
            "baseline executed a lossless fusion unexpectedly: "
            f"{kernel_fusion_hits}"
        )

    diffusers_module = sys.modules.get("diffusers")
    transformers_module = sys.modules.get("transformers")
    diffusers_path = getattr(diffusers_module, "__file__", None)
    serialized_requests = [round(value, 6) for value in requests]
    result = {
        "tag": args.tag,
        "variant": os.environ.get(
            "H3_VARIANT", "diffusers_modular_pruned_fp8_scaled_single_gb10"
        ),
        "dit_checkpoint": dit_checkpoint["path"],
        "dit_checkpoint_size_bytes": dit_checkpoint["size_bytes"],
        "dit_checkpoint_sha256": dit_checkpoint["sha256"],
        "text_encoder": QWEN_FP8,
        "task": "t2va",
        "output_type": args.output_type,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "fps": 24,
        "steps": args.steps,
        "requested_steps": args.requested_steps,
        "scheduler_steps": args.scheduler_steps,
        "denoise_steps": args.denoise_steps,
        "seed": args.seed,
        "prompt_sha256": hashlib.sha256(args.prompt.encode("utf-8")).hexdigest(),
        # Request timing starts immediately before the pipeline call and ends
        # after CUDA synchronization. Model loading, warmups, and artifact
        # encoding are measured separately and are not part of request_s.
        "timing_scope": (
            "pipeline_call_wall_clock_after_warmup_"
            "excluding_load_and_artifact_write"
        ),
        "load_s": round(placement["load_s"], 3),
        "request_s": serialized_requests,
        "request_s_median": round(_request_s_median(serialized_requests), 6),
        "denoise_gpu_s": _round(timers["denoise"].total()),
        "dit_evals": len(timers["denoise"]),
        "encode_gpu_s": _round(timers["encode"].total()),
        "video_decode_gpu_s": _round(timers["decode"].total()),
        "audio_decode_gpu_s": _round(timers["audio_decode"].total()),
        "vae_decode_calls": len(timers["decode"]),
        "vae_tile_stats": vae_shard.read_tile_stats(pipe.vae),
        "peak_memory_mib": round(torch.cuda.max_memory_allocated(args.device) / 1024**2),
        "torch": torch.__version__,
        "transformers": getattr(transformers_module, "__version__", None),
        "diffusers": getattr(diffusers_module, "__version__", None),
        "diffusers_path": os.path.realpath(diffusers_path) if diffusers_path else None,
        "diffusers_source": os.environ.get("H3_DIFFUSERS_SRC"),
        "source_snapshot_sha256": os.environ.get("H3_SOURCE_SNAPSHOT_SHA256"),
        "runtime_source_sha256": os.environ.get("H3_RUNTIME_SOURCE_SHA256"),
        "diffusers_commit": os.environ.get("H3_DIFFUSERS_COMMIT"),
        "diffusers_tree_sha256": os.environ.get("H3_DIFFUSERS_TREE_SHA256"),
        "asset_manifest": os.environ.get("H3_ASSET_MANIFEST_PATH"),
        "asset_manifest_sha256": os.environ.get("H3_ASSET_MANIFEST_SHA256"),
        "triton": getattr(triton, "__version__", None),
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(capability),
        "host": platform.node(),
        "config_id": os.environ.get("AUTOVIDEO_CONFIG_ID"),
        "sana_commit": os.environ.get("H3_SANA_COMMIT"),
        "sana_dirty": (
            _flag("H3_SANA_DIRTY") if os.environ.get("H3_SANA_DIRTY") is not None else None
        ),
        "attention_backend": args.attention_backend or "default",
        "kernel_fusions": bool(kernel_options),
        "kernel_fusion_installation": placement["kernel_fusion_installation"],
        "kernel_fusion_hits": kernel_fusion_hits,
        "qwen_fp8_backend": placement["qwen_fp8_backend"],
        "allocator_warmup": placement["allocator_warmup"],
        "load_order": placement["load_order"],
        "vae_weight_dtype": placement["vae_weight_dtype"],
        "audio_vae_weight_dtype": placement["audio_vae_weight_dtype"],
        "video_vae_autocast_dtype": os.environ.get("H3_VIDEO_VAE_AUTOCAST_DTYPE"),
        "audio_vae_autocast_dtype": os.environ.get("H3_AUDIO_VAE_AUTOCAST_DTYPE"),
        **approximations,
        "sol_attn_stats": sol_attn_stats,
        "triton_tools": {
            key: os.environ.get(key)
            for key in (
                "TRITON_PTXAS_PATH",
                "TRITON_PTXAS_BLACKWELL_PATH",
                "TRITON_CUOBJDUMP_PATH",
                "TRITON_NVDISASM_PATH",
            )
        },
        **_system_memory(),
        "preflight_memavailable_mib": (
            int(os.environ["H3_PREFLIGHT_MEMAVAILABLE_MIB"])
            if os.environ.get("H3_PREFLIGHT_MEMAVAILABLE_MIB")
            else None
        ),
        "min_available_mib": (
            int(os.environ["H3_MIN_AVAILABLE_MIB"])
            if os.environ.get("H3_MIN_AVAILABLE_MIB")
            else None
        ),
        "runtime_min_available_mib": (
            int(os.environ["H3_BENCH_RUNTIME_MIN_AVAILABLE_MIB"])
            if os.environ.get("H3_BENCH_RUNTIME_MIN_AVAILABLE_MIB")
            else None
        ),
        "warmup_requests": args.warmup_requests,
    }
    if cache_counter is not None:
        result["cache_stats"] = {
            "decisions": cache_counter.calls,
            "skips": cache_counter.skips,
            "skip_rate": round(cache_counter.skips / cache_counter.calls, 6)
            if cache_counter.calls
            else 0.0,
        }
    result["per_step_gpu_ms"] = (
        round(result["denoise_gpu_s"] * 1000 / result["dit_evals"], 2)
        if result["denoise_gpu_s"] and result["dit_evals"] else None
    )
    accounted = sum(v for v in (result["denoise_gpu_s"], result["encode_gpu_s"],
                                result["video_decode_gpu_s"], result["audio_decode_gpu_s"]) if v)
    result["unaccounted_s"] = _unaccounted_s(
        result["request_s_median"], accounted, len(requests)
    )

    if state is not None and args.output_type == "latent":
        from safetensors.torch import save_file

        latent_path = os.path.join(args.output_dir, "latents.safetensors")
        video_latent = state.get("videos").detach().cpu().contiguous()
        audio_latent = state.get("audio").detach().cpu().contiguous()
        sampling_rate = int(state.get("sampling_rate"))
        artifact_started = time.perf_counter()
        save_file(
            {"video": video_latent, "audio": audio_latent},
            latent_path,
            metadata={
                "schema": "minimax_h3_latents_v1",
                "audio_sample_rate": str(sampling_rate),
                "fps": "24",
            },
        )
        os.chmod(latent_path, 0o644)
        result["artifact_write_s"] = round(time.perf_counter() - artifact_started, 3)
        result["latents"] = latent_path
        result["latents_sha256"] = _sha256(latent_path)
        result["audio_sample_rate"] = sampling_rate
        result["latent_tensors"] = {
            "video": {"shape": list(video_latent.shape), "dtype": str(video_latent.dtype)},
            "audio": {"shape": list(audio_latent.shape), "dtype": str(audio_latent.dtype)},
        }
        print(f"[h3] wrote {latent_path}", flush=True)

    if not args.no_export and state is not None and args.output_type != "latent":
        from diffusers.utils.export_utils import encode_video

        video_path = os.path.join(args.output_dir, "out.mp4")
        video = state.get("videos")[0]
        artifact_started = time.perf_counter()
        video_chunks_number = 1
        # ``encode_video`` consumes HWC uint8 frames. Diffusers' ``pt`` output
        # is CHW float in [0, 1], while PIL/NumPy paths are already normalized
        # for the exporter. Keep the explicit conversion here so ``--output-type
        # pt`` cannot silently produce a malformed video.
        if args.output_type == "pt":
            video_chunks_number = (video.shape[0] + 15) // 16
            video = video_uint8_chunks(video)
        encode_video(
            video,
            fps=24,
            output_path=video_path,
            audio=state.get("audio")[0],
            audio_sample_rate=state.get("sampling_rate"),
            video_chunks_number=video_chunks_number,
        )
        result["artifact_write_s"] = round(time.perf_counter() - artifact_started, 3)
        result["video"] = video_path
        result["video_sha256"] = _sha256(video_path)
        print(f"[h3] wrote {video_path}", flush=True)

    if result.get("artifact_write_s") is not None:
        result["request_plus_artifact_s"] = round(
            result["request_s_median"] + result["artifact_write_s"], 3
        )

    result_path = os.path.join(args.output_dir, "benchmark.json")
    with open(result_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)

    print(f"[h3] wrote {result_path}", flush=True)
    _restore_process_patches()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
