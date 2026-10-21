"""FirstBlockCache for MiniMax-H3 on one GPU.

diffusers already implements the cache; what it lacks is an entry for
`MiniMaxH3TransformerBlock` in `TransformerBlockRegistry`, which the hooks consult to learn
how to read a block's output. Registering it at runtime is the whole integration — the same
approach Sol-Engine's `cache_line.py` takes, minus its collective-decision fix, which exists
only to keep context-parallel ranks from disagreeing about whether to skip and is not a
concern with one GPU.

What the cache does: after the first block runs, it compares that block's residual against
the one cached from the previous denoising step. If the relative change is below `threshold`,
the remaining 49 blocks are skipped and the previous step's residual is reused. Sol-Engine
measures 2.58x at threshold 0.08, deleting ~69% of block-stack calls.

The interaction with Sol-Attn is subtractive, not multiplicative: every deleted call is one
Sol-Attn would have accelerated, so Sol-Attn keeps about 28% of its standalone benefit once
the cache is on.
"""

from __future__ import annotations

import functools

import paths

paths.setup()

import torch


def _same_callable(first, second) -> bool:
    """Compare bound methods without relying on transient wrapper identities."""
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


def _clear_context_wrapper_state(transformer) -> None:
    transformer._h3_cache_context_wrapped = False
    transformer._h3_cache_original_forward = None
    transformer._h3_cache_context_wrapper = None


def register_h3_block() -> bool:
    """Teach the cache hooks how to read `MiniMaxH3TransformerBlock`. Idempotent."""
    from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock

    try:
        TransformerBlockRegistry.get(MiniMaxH3TransformerBlock)
        return False
    except Exception:
        pass

    TransformerBlockRegistry.register(
        model_class=MiniMaxH3TransformerBlock,
        metadata=TransformerBlockMetadata(
            # The block returns a bare tensor, which the hooks branch on; the index is only
            # read on the tuple path. There is no separate encoder stream either — H3 packs
            # text, audio and video into the one sequence.
            return_hidden_states_index=0,
            return_encoder_hidden_states_index=None,
        ),
    )
    return True


def apply_cache(transformer, threshold: float = 0.08) -> None:
    """Turn FirstBlockCache on, and put every forward inside a cache context.

    Installing the hooks is not sufficient. The state they read lives on the hook registry
    keyed by a *context name*, and pipelines are expected to wrap each denoiser call in
    `transformer.cache_context(...)` — which the H3 modular blocks do not do, so the hooks
    raise "No context is set" on the first call. Wrapping `forward` here supplies it without
    touching the vendored pipeline; state persists across calls because it is keyed by the
    name, not by the context's lifetime.
    """
    from diffusers.hooks import FirstBlockCacheConfig

    register_h3_block()
    transformer.enable_cache(FirstBlockCacheConfig(threshold=threshold))

    if getattr(transformer, "_h3_cache_context_wrapped", False):
        installed = getattr(transformer, "_h3_cache_context_wrapper", None)
        if _same_callable(getattr(transformer, "forward", None), installed):
            return
        # Another hook replaced our wrapper. Start from that current method and keep ownership
        # metadata tied to the new wrapper instead of restoring over the external hook later.
        _clear_context_wrapper_state(transformer)
    original = transformer.forward

    @functools.wraps(original)
    def forward(*args, **kwargs):
        with transformer.cache_context("denoise"):
            return original(*args, **kwargs)

    transformer.forward = forward
    transformer._h3_cache_context_wrapped = True
    transformer._h3_cache_original_forward = original
    transformer._h3_cache_context_wrapper = forward


def remove_cache(transformer) -> None:
    """Take it off again, so one loaded model can serve cached and uncached runs."""
    if getattr(transformer, "_h3_cache_context_wrapped", False):
        installed = getattr(transformer, "_h3_cache_context_wrapper", None)
        if _same_callable(getattr(transformer, "forward", None), installed):
            transformer.forward = transformer._h3_cache_original_forward
        _clear_context_wrapper_state(transformer)
    if getattr(transformer, "is_cache_enabled", False):
        transformer.disable_cache()


def reset_cache(transformer) -> None:
    """Clear the cached residuals between requests, so one sample cannot seed the next."""
    if getattr(transformer, "is_cache_enabled", False):
        transformer._reset_stateful_cache(recurse=True)


class SkipCounter:
    """Count how many steps the cache actually skipped.

    Worth measuring rather than assuming: the skip rate is what the speedup is made of, and
    it depends on the trajectory, so a threshold that deletes 69% of calls on one prompt need
    not do so on another.
    """

    def __init__(self, transformer):
        from diffusers.hooks.first_block_cache import FBCHeadBlockHook

        self.calls = 0
        self.skips = 0
        state = getattr(FBCHeadBlockHook, "_h3_skip_counter_patch_state", None)
        current = FBCHeadBlockHook._should_compute_remaining_blocks
        if not isinstance(state, dict) or current is not state.get("patched"):
            state = {"original": current, "counters": []}

            def counted(hook_self, residual):
                compute = state["original"](hook_self, residual)
                for counter in tuple(state["counters"]):
                    counter.calls += 1
                    counter.skips += not compute
                return compute

            state["patched"] = counted
            FBCHeadBlockHook._should_compute_remaining_blocks = counted
            FBCHeadBlockHook._h3_skip_counter_patch_state = state
        state["counters"].append(self)
        self._state = state
        self._hook_class = FBCHeadBlockHook
        self._original = state["original"]
        self._restored = False

    def restore(self):
        if self._restored:
            return False
        self._restored = True
        counters = self._state["counters"]
        if self in counters:
            counters.remove(self)
        current = self._hook_class._should_compute_remaining_blocks
        if not counters:
            if current is self._state.get("patched"):
                self._hook_class._should_compute_remaining_blocks = self._state["original"]
            if getattr(self._hook_class, "_h3_skip_counter_patch_state", None) is self._state:
                delattr(self._hook_class, "_h3_skip_counter_patch_state")
        return True

    def __str__(self) -> str:
        rate = self.skips / self.calls if self.calls else 0.0
        return f"{self.skips}/{self.calls} steps skipped ({rate:.1%})"
