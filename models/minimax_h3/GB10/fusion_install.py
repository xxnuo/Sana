"""Installing the accelerations onto a built model, and taking them off again.

Sibling of `optimized/fusion_install.py`. Everything here is a patch rather than a structural
change, which is what lets one loaded model serve every configuration in
`config/minimax_h3_gb10_*.toml` — and, more importantly, what lets the benchmark interleave
configurations inside a single process. That matters: this machine's clocks drift up to 15%
between processes, while three iterations inside one agree to 0.7%.

The Sol-Attn dispatch is built here to carry MiniMax-H3's step/block policy. The public kernel
interface selects an exact-match CuTe backend where one exists and otherwise uses Triton.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import os
import sys

import torch
from fusions import (
    fused_apply_rotary_emb,
    fused_gate_add,
    fused_modulate,
    fused_swiglu,
)
from torch import nn

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0

def _ensure_sol_attn_on_path() -> None:
    """Put `techniques/sparse_backends` on `sys.path`, as the entrypoint script intends.

    `H3_SOL_ATTN_ROOT` is set by `run_minimax_h3_gpu.sh`, the same variable the
    8xGB200 entrypoint uses; the fallback walks up to the repo root so the module is still
    importable when a benchmark is run by hand.
    """
    root = os.environ.get("H3_SOL_ATTN_ROOT")
    if not root:
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.normpath(os.path.join(here, "..", "..", "..",
                                             "techniques", "sparse_backends"))
    if root not in sys.path:
        sys.path.insert(0, root)


def _target_video_prefix(video_indices: torch.Tensor, sequence_length: int) -> int:
    """Return the start of the final, generated-video run in the packed layout.

    MiniMax-H3 does not pad text to a fixed length.  In t2va the video rows are one
    contiguous tail; in fl2va a conditioning-video run appears earlier, followed by
    audio and then the generated-video tail.  Taking the final contiguous video run
    therefore gives the exact prefix sink for both layouts without assuming a prompt
    token count.
    """
    if video_indices is None or video_indices.numel() == 0:
        raise ValueError("MiniMax-H3 layout has no video rows")
    positions = sorted(int(value) for value in video_indices.detach().cpu().tolist())
    end = sequence_length - 1
    start = end
    for position in reversed(positions):
        if position != start:
            break
        start -= 1
    target_start = start + 1
    if target_start <= 0 or target_start >= sequence_length:
        raise ValueError(
            "MiniMax-H3 layout does not end in a generated-video tail: "
            f"video_indices={positions[:4]}...{positions[-4:]}, sequence_length={sequence_length}"
        )
    return target_start


def _layout_key(video_indices: torch.Tensor, sequence_length: int) -> tuple:
    """Return a content-derived key for a packed H3 layout.

    ``data_ptr`` is not a layout identity: allocator reuse can give a new request the same
    address, while a graph/caller may also hand us a different view of the same indices.  Copying
    the small index vector to the host is deliberate here; it avoids a false cache hit without
    retaining a device tensor or synchronizing several individual reductions.
    """
    indices = video_indices.detach().contiguous()
    try:
        payload = indices.cpu().numpy().tobytes()
    except (AttributeError, TypeError):
        payload = repr(indices.cpu().tolist()).encode("ascii")
    digest = hashlib.blake2b(payload, digest_size=16).hexdigest()
    return (
        tuple(indices.shape),
        str(indices.dtype),
        str(indices.device),
        int(sequence_length),
        digest,
    )


def load_sol_attn():
    """The released Sol-Attn entry point. Nothing is bypassed and nothing is forked.

    Both were true once. The snapshot this port was written against had no architecture table
    and its dispatcher refused anything that was not SM90 or SM100, so the Triton reference was
    copied out and called directly; that copy also carried a hardcoded 951-row KV sink, because
    the entry point had no way to ask for one.

    Neither is true of the current head. `_backend_for_arch` selects a CuTe backend when the
    capability matches exactly and returns "triton" otherwise, so GB10's (12, 1) reaches the
    Triton path through the public function. And `sink_tokens` / `sink_start` are parameters
    now. Measured on real captured q/k/v across three blocks, the released path with
    `sink_tokens=951` reproduces the fork's output to six decimal places — cos 0.994420,
    0.991519, 0.985544 against dense, identical on both — at the same speed.

    The released MMDiT contract has one additional requirement: a KV sink does not make its own
    query rows exact, so the multimodal prefix must be recomputed with dense attention after the
    sparse call. The dispatcher below performs that overwrite explicitly.
    """
    _ensure_sol_attn_on_path()
    import sol_attn

    return sol_attn.sol_attn


def make_sol_attn_dispatch(
    tau: float = 1.0,
    dense_blocks: int = 2,
    dense_first_steps: int = 0,
    dense_last_steps: int = 0,
    num_steps: int | None = None,
    policy: bool = True,
    thresh_type: str = "diag",
):
    """Build a `dispatch_attention_fn` replacement carrying H3's Sol-Attn policy.

    The tunable surface, all of which the released H3 profile pins:

    * `tau` — routing threshold. Higher keeps fewer key blocks exact.
    * `thresh_type` — diagonal or exact threshold statistics passed to the public kernel.
    * `dense_blocks` — leave the first N transformer blocks dense (released: 2).
    * `dense_first_steps` — leave the first N denoising steps dense (released: 10).
    * `dense_last_steps` — the mirror of the above; not in the released profile, but the
      cheap end of the trajectory to protect if late steps turn out to matter more.
    * `policy` — an exact KV sink covering every row before the generated-video tail. The
      boundary is supplied by the transformer's per-request packed layout, so prompt length
      and keyframe conditioning do not change the policy silently.

    The enclosing transformer wrapper starts a new step explicitly. This matters when
    FirstBlockCache skips the remainder of the block stack: a global attention-call count
    no longer has a fixed number of calls per denoising step.
    """
    from diffusers.models.transformers import transformer_minimax_h3 as h3

    sol_attn = load_sol_attn()
    from sol_attn import get_sol_attn_backend

    backend = get_sol_attn_backend()
    current = h3.dispatch_attention_fn
    dense = getattr(current, "_h3_dense_dispatch", current)
    state = {
        "call": 0,
        "step": -1,
        "position": 0,
        "dense_calls": 0,
        "prefix_dense_calls": 0,
        "sparse_calls": 0,
        "sink_start": 0,
        "sink_tokens": 0,
        "sequence_length": None,
        "layout_key": None,
        "layout_tensor": None,
        "layout_version": None,
    }

    def set_layout(video_indices, sequence_length: int) -> None:
        sequence_length = int(sequence_length)
        version = int(getattr(video_indices, "_version", 0))
        if (
            state["layout_tensor"] is video_indices
            and state["layout_version"] == version
            and state["sequence_length"] == sequence_length
        ):
            return
        layout_key = _layout_key(video_indices, sequence_length)
        if layout_key == state["layout_key"]:
            state["layout_tensor"] = video_indices
            state["layout_version"] = version
            return
        if not policy:
            state["sink_start"] = 0
            state["sink_tokens"] = 0
            state["sequence_length"] = sequence_length
            state["layout_key"] = layout_key
            state["layout_tensor"] = video_indices
            state["layout_version"] = version
            return
        prefix = _target_video_prefix(video_indices, sequence_length)
        state["sink_start"] = 0
        state["sink_tokens"] = prefix
        state["sequence_length"] = sequence_length
        state["layout_key"] = layout_key
        state["layout_tensor"] = video_indices
        state["layout_version"] = version

    def start_forward() -> None:
        state["step"] += 1
        state["position"] = 0

    def dispatch(query, key, value, **kwargs):
        position = state["position"]
        state["position"] += 1
        state["call"] += 1
        step = state["step"]
        block = position - 2                    # the two refiner blocks come first

        late = (
            num_steps is not None
            and dense_last_steps
            and step >= num_steps - dense_last_steps
        )
        if (
            block < dense_blocks
            or step < dense_first_steps
            or late
            or kwargs.get("attn_mask") is not None
        ):
            state["dense_calls"] += 1
            return dense(query, key, value, **kwargs)
        # Sol-Attn requires contiguous BTHD. The processor's q/k/v are `chunk` + `unflatten`
        # views of the fused QKV output, so they are strided; flash takes them as they are,
        # this path cannot.
        sink = (
            {
                "sink_start": state["sink_start"],
                "sink_tokens": state["sink_tokens"],
            }
            if policy
            else {}
        )
        if policy and not state["sink_tokens"]:
            raise RuntimeError("Sol-Attn layout sink was not initialized before attention")
        output = sol_attn(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            tau=tau,
            thresh_type=thresh_type,
            **sink,
        )
        # A KV sink makes the prefix keys exact, but does not constrain its query rows.  Sol-Attn's
        # MMDiT contract requires those rows to be recomputed densely; otherwise text/audio
        # queries still route through the sparse grid and can lose cross-modal conditioning.
        if policy and state["sink_tokens"]:
            lo = state["sink_start"]
            hi = lo + state["sink_tokens"]
            prefix = dense(query[:, lo:hi], key, value, **kwargs)
            output = output.clone()
            output[:, lo:hi] = prefix
            state["prefix_dense_calls"] += 1
        state["sparse_calls"] += 1
        return output

    dispatch.backend = backend
    dispatch.thresh_type = thresh_type
    dispatch.set_layout = set_layout
    dispatch.start_forward = start_forward
    dispatch.reset = lambda: state.update(
        call=0,
        step=-1,
        position=0,
        dense_calls=0,
        prefix_dense_calls=0,
        sparse_calls=0,
        sink_start=0,
        sink_tokens=0,
        sequence_length=None,
        layout_key=None,
        layout_tensor=None,
        layout_version=None,
    )
    def stats() -> dict:
        # Do not expose the live tensor used for identity/version checks: benchmark.json is
        # serialized directly and must remain JSON-compatible.
        return {
            key: value
            for key, value in state.items()
            if key not in {"layout_tensor", "layout_version"}
        } | {"backend": backend, "thresh_type": thresh_type}

    dispatch.stats = stats
    dispatch.prefix_dense_calls = lambda: state.get("prefix_dense_calls", 0)
    dispatch._h3_dense_dispatch = dense
    return dispatch


def _same_callable(first, second) -> bool:
    """Compare functions or bound methods without relying on ephemeral identities."""
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


def _clear_layout_tracking_state(transformer: nn.Module) -> None:
    """Drop bookkeeping after a caller has replaced our forward wrapper."""
    transformer._h3_sol_layout_wrapped = False
    transformer._h3_sol_dispatch = None
    transformer._h3_sol_original_forward = None
    transformer._h3_sol_layout_wrapper = None


def install_layout_tracking(transformer: nn.Module, dispatch) -> None:
    """Feed each request's packed layout to the Sol-Attn dispatch.

    The Diffusers attention callback receives only Q/K/V, while the layout tensors are
    arguments of the enclosing transformer forward.  A small wrapper keeps the callback
    stateless across prompt lengths and requests.  It is installed once per process and is
    deliberately marked so repeated setup cannot stack wrappers.
    """
    if getattr(transformer, "_h3_sol_layout_wrapped", False):
        installed_wrapper = getattr(transformer, "_h3_sol_layout_wrapper", None)
        if _same_callable(getattr(transformer, "forward", None), installed_wrapper):
            transformer._h3_sol_dispatch = dispatch
            return
        # Another hook replaced the wrapper. Do not let stale state make a later uninstall
        # overwrite that hook; the current forward is the new base for this installation.
        _clear_layout_tracking_state(transformer)
    original = transformer.forward
    try:
        forward_signature = inspect.signature(original)
    except (TypeError, ValueError):
        forward_signature = None

    @functools.wraps(original)
    def forward(*args, **kwargs):
        # Diffusers normally forwards these as keywords, but direct callers may use the
        # transformer's positional API. Bind against the original signature once per install so
        # both forms observe the same layout without guessing argument offsets.
        bound = None
        if forward_signature is not None:
            try:
                bound = forward_signature.bind_partial(*args, **kwargs)
            except TypeError:
                # Let the original method report malformed calls; still inspect explicit kwargs
                # so a useful layout error is not introduced by this bookkeeping wrapper.
                pass
        arguments = bound.arguments if bound is not None else kwargs
        video_indices = arguments.get("video_indices")
        position_ids = arguments.get("position_ids")
        if video_indices is None or position_ids is None:
            raise RuntimeError("MiniMax-H3 Sol-Attn requires video_indices and position_ids")
        active_dispatch = transformer._h3_sol_dispatch
        active_dispatch.set_layout(video_indices, int(position_ids.shape[0]))
        active_dispatch.start_forward()
        return original(*args, **kwargs)

    transformer.forward = forward
    transformer._h3_sol_layout_wrapped = True
    transformer._h3_sol_original_forward = original
    transformer._h3_sol_dispatch = dispatch
    transformer._h3_sol_layout_wrapper = forward


def uninstall_layout_tracking(transformer: nn.Module, dispatch=None) -> bool:
    """Restore a forward wrapper installed by :func:`install_layout_tracking`.

    The identity check matters when a caller has already installed a newer dispatch: removing an
    old configuration must not detach the newer one from the transformer.
    """
    if not getattr(transformer, "_h3_sol_layout_wrapped", False):
        return False
    installed_wrapper = getattr(transformer, "_h3_sol_layout_wrapper", None)
    if not _same_callable(getattr(transformer, "forward", None), installed_wrapper):
        # The caller owns the replacement now. Clear stale metadata, but leave its method intact.
        _clear_layout_tracking_state(transformer)
        return False
    if dispatch is not None and getattr(transformer, "_h3_sol_dispatch", None) is not dispatch:
        return False
    transformer.forward = transformer._h3_sol_original_forward
    _clear_layout_tracking_state(transformer)
    return True


def patch_sol_attn(_model: nn.Module, tau: float = 1.0, dense_blocks: int = 2, **kwargs):
    """Route the block stack's self-attention through Sol-Attn, keeping a dense prefix.

    `dense_blocks` mirrors the released policy's first-two-blocks-dense rule. The two token
    refiner blocks run first and over a different prompt-length shape, so they are counted
    out and always left dense.
    """
    from diffusers.models.transformers import transformer_minimax_h3 as h3

    uninstall_sol_attn(_model)
    previous = h3.dispatch_attention_fn
    dispatch = make_sol_attn_dispatch(tau, dense_blocks, **kwargs)
    dispatch._h3_previous_dispatch = previous
    h3.dispatch_attention_fn = dispatch
    install_layout_tracking(_model, dispatch)
    return dispatch


def uninstall_sol_attn(model: nn.Module, dispatch=None) -> bool:
    """Undo the module-level Sol-Attn callback and the model layout wrapper."""
    from diffusers.models.transformers import transformer_minimax_h3 as h3

    active = h3.dispatch_attention_fn
    attached = getattr(model, "_h3_sol_dispatch", None)
    # An explicit dispatch identifies the model wrapper the caller wants removed.  The
    # module-level callback may already belong to another model (or an external hook), so it is
    # only eligible for restoration when it is the same object as the dispatch owned by `model`.
    owned = dispatch if dispatch is not None else attached
    if owned is None:
        return False
    changed = uninstall_layout_tracking(model, owned)
    if active is not owned:
        return changed
    previous = getattr(owned, "_h3_previous_dispatch", None)
    if previous is None:
        return changed
    h3.dispatch_attention_fn = previous
    return True


def patch_fused_swiglu(_model: nn.Module) -> None:
    """Replace `SwiGLU.forward`'s chunk/silu/mul with the fused kernel."""
    from diffusers.models.activations import SwiGLU

    if getattr(SwiGLU, "_h3_fused_swiglu", False):
        installed = getattr(SwiGLU, "_h3_fused_swiglu_forward", None)
        if SwiGLU.forward is installed:
            return
        # A later hook owns the class method. Clear only our stale marker and install on the
        # current method below; unpatch will then remain identity-safe.
        SwiGLU._h3_fused_swiglu = False

    def forward(self, hidden_states):
        return fused_swiglu(self.proj(hidden_states))

    SwiGLU._h3_original_forward = SwiGLU.forward
    SwiGLU._h3_fused_swiglu = True
    SwiGLU._h3_fused_swiglu_forward = forward
    SwiGLU.forward = forward


def unpatch_fused_swiglu() -> bool:
    from diffusers.models.activations import SwiGLU

    if not getattr(SwiGLU, "_h3_fused_swiglu", False):
        return False
    installed = getattr(SwiGLU, "_h3_fused_swiglu_forward", None)
    if SwiGLU.forward is not installed:
        SwiGLU._h3_fused_swiglu = False
        SwiGLU._h3_fused_swiglu_forward = None
        return False
    SwiGLU.forward = SwiGLU._h3_original_forward
    SwiGLU._h3_fused_swiglu = False
    SwiGLU._h3_fused_swiglu_forward = None
    return True


def patch_fused_rope(_model: nn.Module) -> None:
    """Route the attention processor's rotary call through the fused kernel."""
    from diffusers.models.transformers import transformer_minimax_h3 as h3

    if getattr(h3, "_h3_fused_rope", False):
        if h3._apply_rotary_emb is getattr(h3, "_h3_fused_rope_function", None):
            return
        h3._h3_fused_rope = False
    h3._h3_original_apply_rotary_emb = h3._apply_rotary_emb
    h3._h3_fused_rope = True
    h3._h3_fused_rope_function = fused_apply_rotary_emb
    h3._apply_rotary_emb = fused_apply_rotary_emb


def unpatch_fused_rope() -> bool:
    from diffusers.models.transformers import transformer_minimax_h3 as h3

    if not getattr(h3, "_h3_fused_rope", False):
        return False
    if h3._apply_rotary_emb is not getattr(h3, "_h3_fused_rope_function", None):
        h3._h3_fused_rope = False
        h3._h3_fused_rope_function = None
        return False
    h3._apply_rotary_emb = h3._h3_original_apply_rotary_emb
    h3._h3_fused_rope = False
    h3._h3_fused_rope_function = None
    return True


def patch_fused_adaln(model: nn.Module) -> int:
    """Route every block's modulation and gating through the fused kernels."""
    from diffusers.models.transformers import transformer_minimax_h3 as h3

    cls = h3.MiniMaxH3TransformerBlock
    if getattr(cls, "_h3_fused_adaln", False) and cls.forward is getattr(
        cls, "_h3_fused_adaln_forward", None
    ):
        return sum(1 for m in model.modules() if isinstance(m, cls))
    if getattr(cls, "_h3_fused_adaln", False):
        cls._h3_fused_adaln = False
    if not getattr(cls, "_h3_fused_adaln", False):
        h3.MiniMaxH3TransformerBlock._h3_original_forward = h3.MiniMaxH3TransformerBlock.forward

    def forward(self, hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)

        residual = hidden_states
        norm_hidden_states = fused_modulate(
            self.norm1(hidden_states), scale_msa, shift_msa, adaln_indices
        )
        attn_output = self.attn(norm_hidden_states, rotary_emb, attention_mask)
        hidden_states = fused_gate_add(residual, gate_msa, attn_output, adaln_indices)

        residual = hidden_states
        norm_hidden_states = fused_modulate(
            self.norm2(hidden_states), scale_mlp, shift_mlp, adaln_indices
        )
        ff_output = self.ff(norm_hidden_states)
        return fused_gate_add(residual, gate_mlp, ff_output, adaln_indices)

    cls.forward = forward
    cls._h3_fused_adaln = True
    cls._h3_fused_adaln_forward = forward
    return sum(1 for m in model.modules() if isinstance(m, cls))


def unpatch_fused_adaln() -> bool:
    from diffusers.models.transformers import transformer_minimax_h3 as h3

    cls = h3.MiniMaxH3TransformerBlock
    if not getattr(cls, "_h3_fused_adaln", False):
        return False
    if cls.forward is not getattr(cls, "_h3_fused_adaln_forward", None):
        cls._h3_fused_adaln = False
        cls._h3_fused_adaln_forward = None
        return False
    cls.forward = cls._h3_original_forward
    cls._h3_fused_adaln = False
    cls._h3_fused_adaln_forward = None
    return True


def restore_fused_patches() -> bool:
    """Restore all optional class/module fusions before another profile is built."""
    changed = unpatch_fused_swiglu()
    changed = unpatch_fused_rope() or changed
    changed = unpatch_fused_adaln() or changed
    return changed
