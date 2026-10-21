"""Bridge the diffusers `minimax-h3` branch back to transformers 4.57.

The branch was developed against transformers 5.x (Sol-Engine's `model.toml` pins 5.8.1 for
its GB200 environment). Between the two, Qwen3-VL changed how it is told which tokens are
text and which are vision:

* **5.x** has the processor produce `mm_token_type_ids` (`0` text, `1` image, `2` video) and
  passes them into `Qwen3VLModel.forward`, which lays out the 3D rotary positions per
  modality run from them.
* **4.57** has no such argument. `Qwen3VLModel.forward` calls `get_rope_index(input_ids,
  image_grid_thw, video_grid_thw, attention_mask)`, which recovers the same runs by scanning
  `input_ids` for the vision pad ids the encoder block already inserted.

The two derive the same thing from the same information, so the shim is a translation and
not a reimplementation: it restores the processor method the block calls, and drops the
argument 4.57's forward does not accept rather than letting it fall through `**kwargs` into
the decoder layers.
"""

from __future__ import annotations

import functools
import inspect
import os
import sys
import weakref

import torch


_MISSING = object()
_PROCESSOR_PATCH_STATE = "_h3_processor_patch_state"
_TEXT_ENCODER_PATCH_STATE = "_h3_text_encoder_patch_state"
_INSTANCE_PATCH_REFS: list[weakref.ReferenceType] = []


def _track_instance(module) -> None:
    """Keep weak ownership records so process-wide cleanup can restore instance patches."""
    try:
        for reference in _INSTANCE_PATCH_REFS:
            if reference() is module:
                return
        _INSTANCE_PATCH_REFS.append(weakref.ref(module))
    except TypeError:
        # The normal processor/nn.Module implementations are weak-referenceable.  A small
        # third-party proxy may not be; explicit arguments to ``restore_transformers_patches``
        # still provide a cleanup path for that case.
        return


def _tracked_instances() -> list[object]:
    """Return live tracked objects and discard dead weak references."""
    live = []
    kept = []
    for reference in _INSTANCE_PATCH_REFS:
        module = reference()
        if module is not None:
            live.append(module)
            kept.append(reference)
    _INSTANCE_PATCH_REFS[:] = kept
    return live


def _remember_attr(module, state: dict, name: str) -> None:
    """Record one module attribute before an owned runtime patch changes it."""
    if name not in state["attrs"]:
        # Prefer the instance/module namespace over ``getattr``.  For an inherited method,
        # ``getattr`` returns a bound method and restoring that object would leave an instance
        # attribute behind instead of restoring the descriptor on the class.
        try:
            namespace = vars(module)
        except TypeError:
            namespace = None
        if namespace is not None and name not in namespace:
            state["attrs"][name] = (_MISSING, False)
        elif namespace is not None:
            state["attrs"][name] = (namespace[name], True)
        else:
            state["attrs"][name] = (
                getattr(module, name, _MISSING),
                hasattr(module, name),
            )


def _restore_module_state(module, marker: str) -> bool:
    """Restore only attributes that still contain our values.

    A host application may install another compatibility layer after this one.  In that case
    silently writing the saved value over the newer layer is worse than leaving a small marker
    behind, so the identity check deliberately preserves the external replacement.
    """
    state = getattr(module, marker, None)
    if not isinstance(state, dict):
        return False
    changed = False
    for name, (original, existed) in reversed(tuple(state.get("attrs", {}).items())):
        current = getattr(module, name, _MISSING)
        patched = state.get("patched", {}).get(name, _MISSING)
        if current is not patched:
            continue
        if existed:
            setattr(module, name, original)
        elif current is not _MISSING:
            delattr(module, name)
        changed = True
    if getattr(module, marker, None) is state:
        delattr(module, marker)
    return changed


def _create_mm_token_type_ids(processor, batched_ids):
    """`0` text, `1` image, `2` video — recovered from the vision pad ids, as 5.x does."""
    config = processor.tokenizer
    image_pad = config.convert_tokens_to_ids("<|image_pad|>")
    video_pad = config.convert_tokens_to_ids("<|video_pad|>")
    out = []
    for ids in batched_ids:
        out.append([2 if i == video_pad else 1 if i == image_pad else 0 for i in ids])
    return out


def patch_processor(processor):
    """Give a 4.57 `Qwen3VLProcessor` the `create_mm_token_type_ids` the block expects."""
    state = getattr(processor, _PROCESSOR_PATCH_STATE, None)
    current = getattr(processor, "create_mm_token_type_ids", _MISSING)
    if isinstance(state, dict):
        patched = state.get("patched", {}).get("create_mm_token_type_ids", _MISSING)
        if current is patched:
            _track_instance(processor)
            return processor
        # A caller replaced or removed our method. Restore only our metadata (and never
        # overwrite that replacement), then inspect the current processor below.
        _restore_module_state(processor, _PROCESSOR_PATCH_STATE)

    if hasattr(processor, "create_mm_token_type_ids"):
        return processor
    state = {"attrs": {}, "patched": {}}

    def create_mm_token_type_ids(batched_ids):
        return _create_mm_token_type_ids(processor, batched_ids)

    _remember_attr(processor, state, "create_mm_token_type_ids")
    processor.create_mm_token_type_ids = create_mm_token_type_ids
    state["patched"]["create_mm_token_type_ids"] = create_mm_token_type_ids
    setattr(processor, _PROCESSOR_PATCH_STATE, state)
    _track_instance(processor)
    return processor


def patch_text_encoder(text_encoder):
    """Drop `mm_token_type_ids` before it reaches a forward that cannot consume it.

    4.57 recomputes the equivalent inside `get_rope_index`, so dropping it changes nothing;
    forwarding it would push an unknown keyword down into the decoder layers.
    """
    inner = text_encoder.model

    state = getattr(inner, _TEXT_ENCODER_PATCH_STATE, None)
    current = getattr(inner, "forward", _MISSING)
    if isinstance(state, dict):
        patched = state.get("patched", {}).get("forward", _MISSING)
        if current is patched:
            _track_instance(inner)
            return text_encoder
        # A caller replaced our wrapper and owns the active method now. Drop stale ownership
        # metadata, but do not wrap that hook: it may intentionally consume
        # ``mm_token_type_ids``, and swallowing the keyword would change its contract.
        _restore_module_state(inner, _TEXT_ENCODER_PATCH_STATE)
        return text_encoder

    original = current
    if original is _MISSING:
        raise AttributeError("text encoder model has no forward method")

    # ``forward`` is normally a bound method (``nn.Module`` exposes it through the class),
    # hence inspect the bound object directly.  Looking at ``__code__.co_varnames`` is not
    # equivalent: it also contains local variable names and is unavailable for some callable
    # implementations.  Deliberately do not treat a bare ``**kwargs`` as support for this
    # argument; older Qwen forwards pass unknown kwargs further down into decoder layers.
    try:
        parameter = inspect.signature(original, follow_wrapped=False).parameters.get(
            "mm_token_type_ids"
        )
    except (TypeError, ValueError):
        parameter = None
    if parameter is not None:
        return text_encoder

    # The wrapper accepts the 5.x-only keyword at the outer boundary, then drops it before
    # invoking the 4.57 implementation.  ``wraps`` preserves the original method metadata and
    # lets Diffusers/Accelerate continue to inspect the real bound signature.
    @functools.wraps(original)
    def forward(*args, mm_token_type_ids=None, **kwargs):
        return original(*args, **kwargs)

    forward._h3_mm_token_type_ids_wrapper = True
    forward._h3_original_forward = original
    state = {"attrs": {}, "patched": {}}
    _remember_attr(inner, state, "forward")
    inner.forward = forward
    state["patched"]["forward"] = forward
    # Keep a truthy marker for callers that need a cheap check, but include it in the
    # ownership record so cleanup can remove it without touching a later external hook.
    _remember_attr(inner, state, "_h3_mm_token_type_ids_patched")
    inner._h3_mm_token_type_ids_patched = True
    state["patched"]["_h3_mm_token_type_ids_patched"] = True
    _remember_attr(inner, state, "_h3_original_forward")
    inner._h3_original_forward = original
    state["patched"]["_h3_original_forward"] = original
    setattr(inner, _TEXT_ENCODER_PATCH_STATE, state)
    _track_instance(inner)
    return text_encoder


def patch_allocator_warmup() -> str:
    """Disable Transformers' model-sized CUDA allocator preallocation when requested.

    ``caching_allocator_warmup`` is useful on discrete GPUs, but on Thor the GPU and host
    share one UMA pool.  Reserving the complete quantized conditioner before loading its
    shards creates a needless high-water mark and can leave the driver with fragmented
    memory even though the final model fits.  Keep the workaround opt-in so the shared GB10
    profile retains the upstream loader behavior; the Thor wrapper enables it by default.
    """
    enabled = os.environ.get("H3_DISABLE_ALLOCATOR_WARMUP", "0") not in (
        "",
        "0",
        "false",
        "False",
    )
    if not enabled:
        return "enabled"

    try:
        import transformers.modeling_utils as modeling_utils
    except ImportError:
        # Older Transformers releases did not expose the helper.  There is nothing to patch,
        # and model loading can proceed using their existing path.
        return "unavailable"

    state = getattr(modeling_utils, "_h3_allocator_warmup_patch_state", None)
    if isinstance(state, dict):
        patched = state.get("patched", {}).get("caching_allocator_warmup")
        if getattr(modeling_utils, "caching_allocator_warmup", None) is patched:
            return "disabled"
        # A caller replaced our wrapper. Treat the current callable as the new base rather than
        # restoring over code we do not own.
        if getattr(modeling_utils, "_h3_allocator_warmup_patch_state", None) is state:
            delattr(modeling_utils, "_h3_allocator_warmup_patch_state")

    original = getattr(modeling_utils, "caching_allocator_warmup", None)
    if original is None:
        return "unavailable"
    if getattr(original, "_h3_allocator_warmup_disabled", False):
        return "disabled"

    @functools.wraps(original)
    def disabled(*args, **kwargs):
        return None

    disabled._h3_allocator_warmup_disabled = True
    disabled._h3_original_allocator_warmup = original
    state = {
        "attrs": {},
        "patched": {"caching_allocator_warmup": disabled},
    }
    _remember_attr(modeling_utils, state, "caching_allocator_warmup")
    modeling_utils.caching_allocator_warmup = disabled
    modeling_utils._h3_allocator_warmup_patch_state = state
    return "disabled"


def restore_allocator_warmup() -> bool:
    """Undo :func:`patch_allocator_warmup` when the module still belongs to this shim."""
    modeling_utils = sys.modules.get("transformers.modeling_utils")
    if modeling_utils is None:
        try:
            import transformers.modeling_utils as modeling_utils
        except ImportError:
            return False
    return _restore_module_state(modeling_utils, "_h3_allocator_warmup_patch_state")


def patch_finegrained_fp8() -> str:
    """Install the bundled Triton FP8 implementation into Transformers 5.x.

    The stock loader resolves ``kernels-community/finegrained-fp8`` lazily from
    the Hub. Thor runs are intentionally offline, so provide the same public
    kernel bundle before the first quantized Qwen linear is evaluated. Cached
    loaders, module-global bundles, and the older 5.5 globals are supported
    because the host bring-up environment and target container can differ.
    """
    from transformers.integrations import finegrained_fp8

    existing_state = getattr(finegrained_fp8, "_h3_finegrained_fp8_patch_state", None)
    if isinstance(existing_state, dict):
        patched = existing_state.get("patched", {})
        if all(getattr(finegrained_fp8, name, _MISSING) is value for name, value in patched.items()):
            return str(existing_state["backend"])
        # Restore every symbol that is still ours while preserving a newer external
        # replacement. The fresh patch below then records that replacement as its base, so a
        # later cleanup cannot resurrect the remaining symbols from the stale patch.
        _restore_module_state(
            finegrained_fp8, "_h3_finegrained_fp8_patch_state"
        )

    state = {"attrs": {}, "patched": {}, "backend": None}

    def set_owned(name: str, value) -> None:
        _remember_attr(finegrained_fp8, state, name)
        setattr(finegrained_fp8, name, value)
        state["patched"][name] = value

    disable_deepgemm = os.environ.get(
        "TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR", "0"
    ) not in ("", "0", "false", "False")

    # Transformers 5.8 returns a cached dataclass of the four pre-quantized
    # kernel entry points. Replace the loader itself; assigning an unrelated
    # module global would leave the first forward on the Hub-backed code path.
    # Transformers 5.15 still exposes this loader for compatibility, but its
    # ``FineGrainedFP8`` constructor has the modern three-field signature. Use
    # the constructor signature as the API discriminator rather than the loader
    # name, which is present in both releases.
    bundle_type = getattr(finegrained_fp8, "FineGrainedFP8", None)
    bundle_parameters = set()
    if bundle_type is not None:
        try:
            bundle_parameters = set(inspect.signature(bundle_type).parameters)
        except (TypeError, ValueError):
            bundle_parameters = set()
    if bundle_type is not None and {
        "fp8_matmul",
        "fp8_act_quant",
        "batched_fp8_matmul",
        "grouped_fp8_matmul",
    }.issubset(bundle_parameters):
        from local_finegrained_fp8.legacy.act_quant import fp8_act_quant
        from local_finegrained_fp8.legacy.batched import w8a8_fp8_matmul_batched
        from local_finegrained_fp8.legacy.grouped import w8a8_fp8_matmul_grouped
        from local_finegrained_fp8.legacy.matmul import w8a8_fp8_matmul

        bundle = bundle_type(
            fp8_matmul=w8a8_fp8_matmul,
            fp8_act_quant=fp8_act_quant,
            batched_fp8_matmul=w8a8_fp8_matmul_batched,
            grouped_fp8_matmul=w8a8_fp8_matmul_grouped,
        )

        def load_local_finegrained_fp8():
            return bundle

        set_owned("_load_finegrained_fp8_kernel", load_local_finegrained_fp8)
        if disable_deepgemm:
            # The 5.8 dense-linear dispatcher resolves DeepGEMM before Triton.
            # Replacing it with the same local W8A8 function prevents an offline
            # kernels-community/deep-gemm lookup on every 128x128 linear.
            set_owned("w8a8_fp8_matmul", w8a8_fp8_matmul)
        state["backend"] = "local_triton_58"
        set_owned("_h3_local_triton_backend", state["backend"])
        finegrained_fp8._h3_finegrained_fp8_patch_state = state
        return state["backend"]

    # Newer integrations use the same raw-activation bundle, either through a
    # cached loader (5.12) or a module global (5.15).
    if bundle_type is not None and {
        "matmul",
        "batched_matmul",
        "grouped_matmul",
    }.issubset(bundle_parameters):
        # Replace an already-populated bundle too. A prior lazy forward may have
        # loaded the Hub kernel; reporting a local backend while retaining that
        # object would violate the offline Thor contract.
        from local_finegrained_fp8 import matmul_2d, matmul_batched, matmul_grouped

        bundle = bundle_type(
            matmul=matmul_2d,
            batched_matmul=matmul_batched,
            grouped_matmul=matmul_grouped,
        )
        if hasattr(finegrained_fp8, "_FINEGRAINED_FP8"):
            set_owned("_FINEGRAINED_FP8", bundle)
        elif callable(getattr(finegrained_fp8, "_load_finegrained_fp8_kernel", None)):
            def load_local_finegrained_fp8():
                return bundle

            set_owned("_load_finegrained_fp8_kernel", load_local_finegrained_fp8)
        else:
            raise RuntimeError("unsupported Transformers fine-grained FP8 bundle loader")
        state["backend"] = "local_triton_v4"
        set_owned("_h3_local_triton_backend", state["backend"])
        finegrained_fp8._h3_finegrained_fp8_patch_state = state
        return state["backend"]

    # Transformers 5.5 (used by the host-side fallback environment) keeps four
    # legacy globals and asks the loader for pre-quantized activations.  The v4
    # source is kept above for the target image; these compatibility functions are
    # the matching older implementation and avoid the same Hub lookup.
    required = ("triton_fp8_matmul", "triton_fp8_act_quant", "triton_batched_fp8_matmul",
                "triton_grouped_fp8_matmul", "_triton_available")
    if not all(hasattr(finegrained_fp8, name) for name in required):
        raise RuntimeError("unsupported Transformers fine-grained FP8 integration API")
    from local_finegrained_fp8.legacy.act_quant import fp8_act_quant
    from local_finegrained_fp8.legacy.batched import w8a8_fp8_matmul_batched
    from local_finegrained_fp8.legacy.grouped import w8a8_fp8_matmul_grouped
    from local_finegrained_fp8.legacy.matmul import w8a8_fp8_matmul

    set_owned("triton_fp8_matmul", w8a8_fp8_matmul)
    set_owned("triton_fp8_act_quant", fp8_act_quant)
    set_owned("triton_batched_fp8_matmul", w8a8_fp8_matmul_batched)
    set_owned("triton_grouped_fp8_matmul", w8a8_fp8_matmul_grouped)
    set_owned("_triton_available", True)

    # Transformers 5.5 tries the separately hosted DeepGEMM kernel before the
    # Triton fallback for 128x128 linears.  Thor has no validated DeepGEMM path,
    # and the run is explicitly offline; mark it unavailable so the built-in
    # ImportError fallback never attempts a Hub lookup.
    if disable_deepgemm:
        set_owned("_deepgemm_available", False)
    state["backend"] = "local_triton_55"
    set_owned("_h3_local_triton_backend", state["backend"])
    finegrained_fp8._h3_finegrained_fp8_patch_state = state
    return state["backend"]


def restore_finegrained_fp8() -> bool:
    """Restore the locally injected FP8 symbols without clobbering later patches."""
    finegrained_fp8 = sys.modules.get("transformers.integrations.finegrained_fp8")
    if finegrained_fp8 is None:
        try:
            from transformers.integrations import finegrained_fp8
        except ImportError:
            return False
    return _restore_module_state(
        finegrained_fp8, "_h3_finegrained_fp8_patch_state"
    )


def restore_instance_patches(processor=None, text_encoder=None) -> bool:
    """Restore instance-level compatibility hooks while preserving external replacements.

    ``patch_processor`` and ``patch_text_encoder`` cannot be undone through a module-global
    symbol, so keep weak ownership records and accept explicit objects for callers that already
    hold a pipeline.  Cleanup is identity-safe: if another hook replaced one of our methods, the
    replacement remains untouched and only our bookkeeping is removed.
    """
    targets = []
    if processor is not None:
        targets.append((processor, _PROCESSOR_PATCH_STATE))
    if text_encoder is not None:
        inner = getattr(text_encoder, "model", text_encoder)
        targets.append((inner, _TEXT_ENCODER_PATCH_STATE))

    # The weak registry is intentionally consulted even when explicit objects are supplied: a
    # pipeline may have both a processor and a text encoder while a previous pipeline is still
    # live in the same interpreter.
    for module in _tracked_instances():
        targets.append((module, _PROCESSOR_PATCH_STATE))
        targets.append((module, _TEXT_ENCODER_PATCH_STATE))

    changed = False
    seen = set()
    for module, marker in targets:
        identity = (id(module), marker)
        if identity in seen:
            continue
        seen.add(identity)
        changed = _restore_module_state(module, marker) or changed
    return changed


def restore_transformers_patches(processor=None, text_encoder=None) -> bool:
    """Restore all compatibility hooks installed by this module."""
    changed = restore_instance_patches(processor, text_encoder)
    changed = restore_finegrained_fp8() or changed
    return restore_allocator_warmup() or changed


def assert_text_only(processor, prompt_ids) -> None:
    """The equivalence above is exact for text-only prompts; flag anything else loudly."""
    types = torch.tensor(processor.create_mm_token_type_ids([prompt_ids]))
    if bool((types != 0).any()):
        raise ValueError(
            "this compatibility shim was validated for text-only (t2va) prompts; the prompt "
            "carries vision tokens, so verify the 4.57 rope layout before trusting the output"
        )
