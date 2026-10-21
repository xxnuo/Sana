from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

GB10_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GB10_ROOT))

import fusion_install


def test_target_video_prefix_uses_final_contiguous_video_run():
    assert fusion_install._target_video_prefix(torch.tensor([5, 6, 7]), 8) == 5
    assert fusion_install._target_video_prefix(torch.tensor([2, 3, 6, 7, 8]), 9) == 6


@pytest.mark.parametrize("video_indices", [torch.tensor([]), torch.tensor([1, 2])])
def test_target_video_prefix_rejects_missing_video_tail(video_indices):
    with pytest.raises(ValueError):
        fusion_install._target_video_prefix(video_indices, 4)


def test_layout_key_is_content_based_not_pointer_based():
    first = torch.tensor([2, 3, 7, 8], dtype=torch.int64)
    second = first.clone()
    assert fusion_install._layout_key(first, 9) == fusion_install._layout_key(second, 9)
    changed = first.clone()
    changed[-1] = 6
    assert fusion_install._layout_key(first, 9) != fusion_install._layout_key(changed, 9)


def test_dispatch_step_tracking_survives_cache_skipped_blocks(monkeypatch):
    dense_calls = []
    sparse_calls = []

    def dense(query, key, value, **kwargs):
        dense_calls.append(kwargs)
        return query

    def sparse(query, key, value, **kwargs):
        sparse_calls.append(kwargs)
        return query

    h3 = SimpleNamespace(dispatch_attention_fn=dense)
    transformers = ModuleType("diffusers.models.transformers")
    transformers.transformer_minimax_h3 = h3
    models = ModuleType("diffusers.models")
    models.transformers = transformers
    diffusers = ModuleType("diffusers")
    diffusers.models = models
    sol_attn = ModuleType("sol_attn")
    sol_attn.get_sol_attn_backend = lambda: "triton"

    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.models", models)
    monkeypatch.setitem(sys.modules, "diffusers.models.transformers", transformers)
    monkeypatch.setitem(sys.modules, "sol_attn", sol_attn)
    monkeypatch.setattr(fusion_install, "load_sol_attn", lambda: sparse)

    dispatch = fusion_install.make_sol_attn_dispatch(
        dense_blocks=2,
        dense_first_steps=2,
        num_steps=4,
    )
    dispatch.set_layout(torch.tensor([5, 6, 7]), 8)
    query = torch.zeros((1, 8, 1, 128), dtype=torch.bfloat16)

    # A cached step may execute only the two refiner calls and block zero.
    for calls_this_step in (3, 3):
        dispatch.start_forward()
        for _ in range(calls_this_step):
            dispatch(query, query, query)

    # Step two is no longer in the dense prefix. Refiner calls and blocks 0/1
    # remain dense; block 2 is the first sparse call.
    dispatch.start_forward()
    for _ in range(5):
        dispatch(query, query, query)

    assert len(sparse_calls) == 1
    assert sparse_calls[0]["thresh_type"] == "diag"
    assert sparse_calls[0]["sink_start"] == 0
    assert sparse_calls[0]["sink_tokens"] == 5
    assert dispatch.stats()["step"] == 2
    # Ten policy-dense calls plus one dense prefix-query overwrite for the sparse call.
    assert len(dense_calls) == 11
    assert dispatch.stats()["prefix_dense_calls"] == 1


def test_dispatch_forwards_explicit_threshold_type(monkeypatch):
    sparse_calls = []

    def dense(query, key, value, **kwargs):
        return query

    def sparse(query, key, value, **kwargs):
        sparse_calls.append(kwargs)
        return query

    h3 = SimpleNamespace(dispatch_attention_fn=dense)
    transformers = ModuleType("diffusers.models.transformers")
    transformers.transformer_minimax_h3 = h3
    models = ModuleType("diffusers.models")
    models.transformers = transformers
    diffusers = ModuleType("diffusers")
    diffusers.models = models
    sol_attn = ModuleType("sol_attn")
    sol_attn.get_sol_attn_backend = lambda: "triton"
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.models", models)
    monkeypatch.setitem(sys.modules, "diffusers.models.transformers", transformers)
    monkeypatch.setitem(sys.modules, "sol_attn", sol_attn)
    monkeypatch.setattr(fusion_install, "load_sol_attn", lambda: sparse)

    dispatch = fusion_install.make_sol_attn_dispatch(
        thresh_type="exact", dense_blocks=0, dense_first_steps=0
    )
    dispatch.set_layout(torch.tensor([4, 5, 6, 7]), 8)
    dispatch.start_forward()
    query = torch.zeros((1, 8, 1, 128), dtype=torch.bfloat16)
    dispatch(query, query, query)
    dispatch(query, query, query)
    dispatch(query, query, query)

    assert dispatch.thresh_type == "exact"
    assert dispatch.stats()["thresh_type"] == "exact"
    assert sparse_calls[0]["thresh_type"] == "exact"


def test_dispatch_keeps_configured_suffix_steps_dense(monkeypatch):
    dense_calls = []
    sparse_calls = []

    def dense(query, key, value, **kwargs):
        dense_calls.append(kwargs)
        return query

    def sparse(query, key, value, **kwargs):
        sparse_calls.append(kwargs)
        return query

    h3 = SimpleNamespace(dispatch_attention_fn=dense)
    transformers = ModuleType("diffusers.models.transformers")
    transformers.transformer_minimax_h3 = h3
    models = ModuleType("diffusers.models")
    models.transformers = transformers
    diffusers = ModuleType("diffusers")
    diffusers.models = models
    sol_attn = ModuleType("sol_attn")
    sol_attn.get_sol_attn_backend = lambda: "triton"

    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.models", models)
    monkeypatch.setitem(sys.modules, "diffusers.models.transformers", transformers)
    monkeypatch.setitem(sys.modules, "sol_attn", sol_attn)
    monkeypatch.setattr(fusion_install, "load_sol_attn", lambda: sparse)

    dispatch = fusion_install.make_sol_attn_dispatch(
        dense_blocks=0,
        dense_last_steps=1,
        num_steps=3,
    )
    dispatch.set_layout(torch.tensor([5, 6, 7]), 8)
    query = torch.zeros((1, 8, 1, 128), dtype=torch.bfloat16)

    for _ in range(3):
        dispatch.start_forward()
        for _ in range(3):
            dispatch(query, query, query)

    assert len(sparse_calls) == 2
    assert dispatch.stats()["dense_calls"] == 7
    assert dispatch.stats()["prefix_dense_calls"] == 2
    assert len(dense_calls) == 9


def test_sparse_dispatch_overwrites_prefix_queries(monkeypatch):
    dense_calls = []

    def dense(query, key, value, **kwargs):
        dense_calls.append(tuple(query.shape))
        return query + 7

    def sparse(query, key, value, **kwargs):
        return torch.zeros_like(query)

    h3 = SimpleNamespace(dispatch_attention_fn=dense)
    transformers = ModuleType("diffusers.models.transformers")
    transformers.transformer_minimax_h3 = h3
    models = ModuleType("diffusers.models")
    models.transformers = transformers
    diffusers = ModuleType("diffusers")
    diffusers.models = models
    sol_attn = ModuleType("sol_attn")
    sol_attn.get_sol_attn_backend = lambda: "triton"
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.models", models)
    monkeypatch.setitem(sys.modules, "diffusers.models.transformers", transformers)
    monkeypatch.setitem(sys.modules, "sol_attn", sol_attn)
    monkeypatch.setattr(fusion_install, "load_sol_attn", lambda: sparse)

    dispatch = fusion_install.make_sol_attn_dispatch(dense_blocks=0)
    dispatch.set_layout(torch.tensor([4, 5, 6, 7]), 8)
    dispatch.start_forward()
    query = torch.ones((1, 8, 1, 128), dtype=torch.bfloat16)
    # The two token-refiner calls are always dense; block zero is the first sparse call.
    dispatch(query, query, query)
    dispatch(query, query, query)
    result = dispatch(query, query, query)

    assert torch.equal(result[:, :4], torch.full_like(result[:, :4], 8))
    assert torch.equal(result[:, 4:], torch.zeros_like(result[:, 4:]))
    assert dense_calls[-1] == (1, 4, 1, 128)
    assert dispatch.stats()["prefix_dense_calls"] == 1


def test_layout_wrapper_can_be_uninstalled(monkeypatch):
    original = lambda *args, **kwargs: "original"
    transformer = SimpleNamespace(forward=original)
    dispatch = SimpleNamespace(set_layout=lambda *_: None, start_forward=lambda: None)
    fusion_install.install_layout_tracking(transformer, dispatch)
    assert transformer.forward is not original
    assert fusion_install.uninstall_layout_tracking(transformer, dispatch)
    assert transformer.forward is original
    assert not fusion_install.uninstall_layout_tracking(transformer, dispatch)


def test_layout_uninstall_does_not_clobber_external_replacement():
    original = lambda *args, **kwargs: "original"
    transformer = SimpleNamespace(forward=original)
    dispatch = SimpleNamespace(set_layout=lambda *_: None, start_forward=lambda: None)
    fusion_install.install_layout_tracking(transformer, dispatch)

    replacement = lambda *args, **kwargs: "replacement"
    transformer.forward = replacement

    assert not fusion_install.uninstall_layout_tracking(transformer, dispatch)
    assert transformer.forward is replacement
    assert not getattr(transformer, "_h3_sol_layout_wrapped", False)


def test_layout_wrapper_preserves_forward_signature_for_diffusers_filtering():
    class Transformer:
        def forward(
            self,
            hidden_states,
            *,
            video_indices=None,
            position_ids=None,
            timestep=None,
        ):
            return hidden_states, video_indices, position_ids, timestep

    calls = []
    transformer = Transformer()
    original = transformer.forward
    dispatch = SimpleNamespace(
        set_layout=lambda video_indices, sequence_length: calls.append(
            (video_indices, sequence_length)
        ),
        start_forward=lambda: calls.append("start"),
    )

    fusion_install.install_layout_tracking(transformer, dispatch)

    signature = inspect.signature(transformer.forward)
    assert "video_indices" in signature.parameters
    assert "position_ids" in signature.parameters
    assert "timestep" in signature.parameters
    assert signature.parameters["video_indices"].kind is inspect.Parameter.KEYWORD_ONLY

    video_indices = torch.tensor([3, 4])
    position_ids = torch.zeros((6, 3), dtype=torch.long)
    filtered = {
        name: value
        for name, value in {
            "video_indices": video_indices,
            "position_ids": position_ids,
            "timestep": torch.tensor([0.5]),
        }.items()
        if name in signature.parameters
    }
    result = transformer.forward(torch.ones(1), **filtered)
    assert result[0].equal(torch.ones(1))
    assert result[1] is video_indices
    assert result[2] is position_ids
    assert calls[0] == (video_indices, 6)
    assert calls[1] == "start"

    assert fusion_install.uninstall_layout_tracking(transformer, dispatch)
    assert transformer.forward.__func__ is original.__func__


def test_layout_wrapper_accepts_positional_layout_arguments():
    class Transformer:
        def forward(self, hidden_states, video_indices, position_ids, timestep=None):
            return hidden_states, video_indices, position_ids, timestep

    calls = []
    transformer = Transformer()
    dispatch = SimpleNamespace(
        set_layout=lambda video_indices, sequence_length: calls.append(
            (video_indices, sequence_length)
        ),
        start_forward=lambda: calls.append("start"),
    )
    fusion_install.install_layout_tracking(transformer, dispatch)

    video_indices = torch.tensor([3, 4])
    position_ids = torch.zeros((6, 3), dtype=torch.long)
    result = transformer.forward(torch.ones(1), video_indices, position_ids)

    assert result[1] is video_indices
    assert result[2] is position_ids
    assert calls == [(video_indices, 6), "start"]
    assert fusion_install.uninstall_layout_tracking(transformer, dispatch)


def _install_fake_h3_module(monkeypatch, dispatch):
    h3 = SimpleNamespace(dispatch_attention_fn=dispatch)
    transformers = ModuleType("diffusers.models.transformers")
    transformers.transformer_minimax_h3 = h3
    models = ModuleType("diffusers.models")
    models.transformers = transformers
    diffusers = ModuleType("diffusers")
    diffusers.models = models
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.models", models)
    monkeypatch.setitem(sys.modules, "diffusers.models.transformers", transformers)
    return h3


def _attached_transformer(dispatch):
    transformer = SimpleNamespace(forward=lambda *args, **kwargs: "original")
    fusion_install.install_layout_tracking(transformer, dispatch)
    return transformer


def test_uninstall_unrelated_model_does_not_change_active_dispatch(monkeypatch):
    dense = lambda *args, **kwargs: "dense"
    active = SimpleNamespace(_h3_previous_dispatch=dense)
    h3 = _install_fake_h3_module(monkeypatch, active)
    unrelated = SimpleNamespace(forward=lambda *args, **kwargs: "unrelated")

    assert not fusion_install.uninstall_sol_attn(unrelated)
    assert h3.dispatch_attention_fn is active


def test_uninstall_preserves_external_dispatch_hook(monkeypatch):
    dense = lambda *args, **kwargs: "dense"
    owned = SimpleNamespace(_h3_previous_dispatch=dense)
    h3 = _install_fake_h3_module(monkeypatch, owned)
    transformer = _attached_transformer(owned)
    external = lambda *args, **kwargs: "external"
    h3.dispatch_attention_fn = external

    assert fusion_install.uninstall_sol_attn(transformer)
    assert h3.dispatch_attention_fn is external


def test_uninstall_old_dispatch_does_not_clobber_new_dispatch(monkeypatch):
    dense = lambda *args, **kwargs: "dense"
    old = SimpleNamespace(_h3_previous_dispatch=dense)
    new = SimpleNamespace(_h3_previous_dispatch=old)
    h3 = _install_fake_h3_module(monkeypatch, old)
    transformer = _attached_transformer(old)
    h3.dispatch_attention_fn = new

    assert fusion_install.uninstall_sol_attn(transformer, old)
    assert h3.dispatch_attention_fn is new
