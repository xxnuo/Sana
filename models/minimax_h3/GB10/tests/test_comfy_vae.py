from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn
from torch.nn.utils import remove_weight_norm, weight_norm


GB10_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GB10_ROOT))

import comfy_vae  # noqa: E402


class _Projection(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features)


class _ConvHolder(nn.Module):
    def __init__(self, conv: nn.Module):
        super().__init__()
        self.conv = conv


class _ModuleHolder(nn.Module):
    def __init__(self, **modules: nn.Module):
        super().__init__()
        for name, module in modules.items():
            setattr(self, name, module)


class _TinyVideo(nn.Module):
    def __init__(self, **_ignored):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.conv_in = nn.Conv3d(1, 2, 1)
        self.encoder.down_blocks = nn.ModuleList(
            [
                _ModuleHolder(
                    resnets=nn.ModuleList([_TinyResnet()]),
                    downsamplers=nn.ModuleList([_ConvHolder(nn.Conv3d(2, 2, 1))]),
                )
            ]
        )
        self.encoder.norm_out = nn.GroupNorm(1, 2)
        self.encoder.conv_out = nn.Conv3d(2, 2, 1)
        self.quant_conv = nn.Conv3d(2, 2, 1)
        self.post_quant_conv = nn.Conv3d(1, 1, 1)

        self.decoder = nn.Module()
        self.decoder.proj_in = nn.Linear(1, 6)
        self.decoder.transformer_blocks = nn.ModuleList([_TinyVideoBlock()])
        self.decoder.norm_out = nn.LayerNorm(6)
        self.decoder.proj_out = nn.Linear(6, 1)


class _TinyResnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, 2)
        self.conv1 = nn.Conv3d(2, 2, 1)
        self.norm2 = nn.GroupNorm(1, 2)
        self.conv2 = nn.Conv3d(2, 2, 1)
        self.conv_shortcut = nn.Conv3d(2, 2, 1)


class _TinyVideoBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(6)
        self.attn = nn.Module()
        self.attn.to_q = nn.Linear(6, 6)
        self.attn.to_k = nn.Linear(6, 6)
        self.attn.to_v = nn.Linear(6, 6)
        self.attn.to_out = nn.ModuleList([nn.Linear(6, 6), nn.Dropout(0.0)])
        self.scale1 = nn.Parameter(torch.zeros(6))
        self.norm2 = nn.LayerNorm(6)
        self.ff = nn.Module()
        self.ff.net = nn.ModuleList([_Projection(6, 12), nn.Identity(), nn.Linear(12, 6)])
        self.scale2 = nn.Parameter(torch.zeros(6))


class _TinyAudio(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = weight_norm(nn.Conv1d(1, 2, 1))
        self.pre_block = nn.Linear(2, 2)
        self.mean_proj = nn.Conv1d(2, 2, 1)
        self.logs_proj = nn.Conv1d(2, 2, 1)
        self.dec_in_proj = nn.Conv1d(2, 2, 1)
        self.decoder = weight_norm(nn.Conv1d(2, 1, 1))


def _video_config():
    return {"decoder_num_attention_heads": 2, "decoder_attention_head_dim": 3}


def _audio_config():
    return {}


def _video_source_from_model(model: _TinyVideo) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    source: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("encoder.down_blocks.0.resnets.0."):
            source["encoder.down.0.block.0." + key.split(".resnets.0.", 1)[1].replace("conv_shortcut", "nin_shortcut")] = value
        elif key.startswith("encoder.down_blocks.0.downsamplers.0.conv."):
            source["encoder.down.0.downsample.conv." + key.split(".conv.", 1)[1]] = value
        elif key.startswith("decoder.proj_in."):
            source[key.replace("decoder.proj_in.", "decoder.x_embedder.")] = value
        elif key.startswith("decoder.transformer_blocks.0.attn.to_q.") or key.startswith(
            "decoder.transformer_blocks.0.attn.to_k."
        ) or key.startswith("decoder.transformer_blocks.0.attn.to_v."):
            continue
        elif key.startswith("decoder.transformer_blocks.0.ff.net.0.proj.") or key.startswith(
            "decoder.transformer_blocks.0.ff.net.2."
        ):
            continue
        elif key.startswith("decoder.transformer_blocks.0.attn.to_out.0."):
            source[key.replace("decoder.transformer_blocks.0.attn.to_out.0.", "decoder.transformer_blocks.0.attn.to_out.")] = value
        else:
            source[key] = value

    heads, head_dim = 2, 3
    qkv = torch.cat(
        [
            state[f"decoder.transformer_blocks.0.attn.to_{part}.weight"].reshape(heads, head_dim, 6)
            for part in ("q", "k", "v")
        ],
        dim=1,
    ).reshape(3 * heads * head_dim, 6)
    qkv_bias = torch.cat(
        [
            state[f"decoder.transformer_blocks.0.attn.to_{part}.bias"].reshape(heads, head_dim)
            for part in ("q", "k", "v")
        ],
        dim=1,
    ).reshape(3 * heads * head_dim)
    source["decoder.transformer_blocks.0.attn.to_qkv.weight"] = qkv
    source["decoder.transformer_blocks.0.attn.to_qkv.bias"] = qkv_bias
    for index in (0, 2):
        for suffix in ("weight", "bias"):
            target = f"decoder.transformer_blocks.0.ff.net.{index}.proj.{suffix}" if index == 0 else f"decoder.transformer_blocks.0.ff.net.{index}.{suffix}"
            value = state[target]
            if index == 0:
                up, gate = value.chunk(2, dim=0)
                value = torch.cat((gate, up), dim=0)
            source[f"decoder.transformer_blocks.0.ff.w{1 if index == 0 else 2}.{suffix}"] = value
    source["decoder.mask_token"] = torch.zeros(1, 1, 6)
    source["latents_mean"] = torch.zeros(1)
    source["latents_std"] = torch.ones(1)
    return source


def test_video_loader_maps_fused_qkv_and_renamed_blocks(tmp_path, monkeypatch):
    reference = _TinyVideo()
    source = _video_source_from_model(reference)
    checkpoint = tmp_path / "video.safetensors"
    save_file(source, str(checkpoint))
    monkeypatch.setattr(comfy_vae, "_load_diffusers_class", lambda kind: _TinyVideo)

    loaded = comfy_vae.build_comfy_video_vae(checkpoint, _video_config(), device="cpu", dtype=torch.float32)

    assert all(parameter.device.type != "meta" for parameter in loaded.parameters())
    for key, value in reference.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value), key


def test_audio_loader_removes_weight_norm_before_loading(tmp_path, monkeypatch):
    reference = _TinyAudio()
    remove_weight_norm(reference.encoder)
    remove_weight_norm(reference.decoder)
    source = dict(reference.state_dict())
    source["latents_mean"] = torch.zeros(2)
    source["latents_std"] = torch.ones(2)
    checkpoint = tmp_path / "audio.safetensors"
    save_file(source, str(checkpoint))
    monkeypatch.setattr(comfy_vae, "_load_diffusers_class", lambda kind: _TinyAudio)

    loaded = comfy_vae.build_comfy_audio_vae(checkpoint, _audio_config(), device="cpu", dtype=torch.float32)

    assert all(parameter.device.type != "meta" for parameter in loaded.parameters())
    assert not hasattr(loaded.encoder, "weight_g")
    assert not hasattr(loaded.decoder, "weight_g")
    for key, value in reference.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value), key


def test_video_loader_rejects_nonzero_ignored_mask(tmp_path, monkeypatch):
    reference = _TinyVideo()
    source = _video_source_from_model(reference)
    source["decoder.mask_token"].fill_(1)
    checkpoint = tmp_path / "video.safetensors"
    save_file(source, str(checkpoint))
    monkeypatch.setattr(comfy_vae, "_load_diffusers_class", lambda kind: _TinyVideo)

    # The mask is not part of the Diffusers implementation; a nonzero value
    # must not be silently discarded.
    with pytest.raises(ValueError, match="mask_token"):
        comfy_vae.build_comfy_video_vae(checkpoint, _video_config(), device="cpu", dtype=torch.float32)
