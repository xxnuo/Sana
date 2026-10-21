"""Load the single-file MiniMax-H3 VAE exports published for ComfyUI.

The ComfyUI video file contains the same weights as the Diffusers video VAE, but
uses a fused QKV projection and a different module naming scheme.  The audio
file uses the original (unparametrized) convolution weights while Diffusers
constructs those convolutions with ``weight_norm``.  This module performs that
small format conversion while the model is still on ``meta`` and reads one
tensor at a time directly onto the requested device.

It intentionally does not write a converted checkpoint.  Keeping the source
file as the only persistent copy avoids another 5 GB artifact and lets the
caller choose a dtype appropriate for the target GPU.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn


_VIDEO_IGNORED = frozenset({"decoder.mask_token", "latents_mean", "latents_std"})
_AUDIO_IGNORED = frozenset({"latents_mean", "latents_std"})


def _config_dict(config: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Return constructor arguments from a Diffusers config or config path."""

    if isinstance(config, (str, Path)):
        with Path(config).open(encoding="utf-8") as handle:
            config = json.load(handle)
    if not isinstance(config, Mapping):
        raise TypeError(f"VAE config must be a mapping or JSON path, got {type(config)!r}")
    # `_class_name`, `_diffusers_version`, and similar metadata are not
    # constructor arguments.
    return {key: value for key, value in config.items() if not str(key).startswith("_")}


def _target_module(root: nn.Module, key: str) -> tuple[nn.Module, str]:
    parent_name, separator, leaf = key.rpartition(".")
    if not separator:
        return root, key
    return root.get_submodule(parent_name), leaf


def _assign(root: nn.Module, key: str, value: torch.Tensor) -> None:
    """Replace one meta parameter/buffer without retaining a CPU staging copy."""

    parent, leaf = _target_module(root, key)
    if leaf in parent._parameters:
        parent._parameters[leaf] = nn.Parameter(value, requires_grad=False)
    elif leaf in parent._buffers:
        parent._buffers[leaf] = value
    else:
        raise KeyError(f"checkpoint key {key!r} is not a parameter or buffer in {type(root).__name__}")


def _materialize(value: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Move/cast one tensor, allowing the source safetensor to be released immediately."""

    return value.to(device=device, dtype=dtype, non_blocking=True)


def _build_on_meta(cls: type[nn.Module], config: Mapping[str, Any]) -> nn.Module:
    """Instantiate a Diffusers VAE without allocating its parameters."""

    try:
        from accelerate import init_empty_weights
    except ImportError:
        init_empty_weights = None

    if init_empty_weights is not None:
        with init_empty_weights():
            return cls(**dict(config))
    with torch.device("meta"):
        return cls(**dict(config))


def _load_diffusers_class(kind: str) -> type[nn.Module]:
    if kind == "video":
        from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import AutoencoderKLMiniMaxH3

        return AutoencoderKLMiniMaxH3
    if kind == "audio":
        from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_audio import (
            AutoencoderKLMiniMaxH3Audio,
        )

        return AutoencoderKLMiniMaxH3Audio
    raise ValueError(f"unknown MiniMax-H3 VAE kind: {kind!r}")


def _video_target_keys(source_key: str) -> tuple[str, ...]:
    """Map one Comfy video key to one or more Diffusers keys."""

    if source_key in _VIDEO_IGNORED:
        return ()

    match = re.fullmatch(r"encoder\.down\.(\d+)\.block\.(\d+)\.(.+)", source_key)
    if match:
        tail = match.group(3).replace("nin_shortcut", "conv_shortcut")
        return (f"encoder.down_blocks.{match.group(1)}.resnets.{match.group(2)}.{tail}",)

    match = re.fullmatch(r"encoder\.down\.(\d+)\.downsample\.conv\.(.+)", source_key)
    if match:
        return (f"encoder.down_blocks.{match.group(1)}.downsamplers.0.conv.{match.group(2)}",)

    if source_key.startswith("decoder.x_embedder."):
        return (source_key.replace("decoder.x_embedder.", "decoder.proj_in.", 1),)
    if ".attn.to_qkv." in source_key:
        prefix, _, suffix = source_key.partition(".attn.to_qkv.")
        return tuple(f"{prefix}.attn.to_{part}.{suffix}" for part in ("q", "k", "v"))
    if ".attn.to_out." in source_key:
        return (source_key.replace(".attn.to_out.", ".attn.to_out.0.", 1),)
    if ".ff.w1." in source_key:
        return (source_key.replace(".ff.w1.", ".ff.net.0.proj.", 1),)
    if ".ff.w2." in source_key:
        return (source_key.replace(".ff.w2.", ".ff.net.2.", 1),)
    return (source_key,)


def _reorder_interleaved_qkv(
    tensor: torch.Tensor, num_heads: int, head_dim: int
) -> torch.Tensor:
    """Turn Comfy's ``[q0,k0,v0,q1,k1,v1,...]`` rows into ``[q*,k*,v*]``."""

    expected_rows = num_heads * 3 * head_dim
    if tensor.ndim == 0 or tensor.shape[0] != expected_rows:
        raise ValueError(
            f"fused QKV tensor has shape {tuple(tensor.shape)}, expected first dimension {expected_rows}"
        )
    grouped = tensor.reshape(num_heads, 3 * head_dim, *tensor.shape[1:])
    query, key, value = grouped.split(head_dim, dim=1)
    return torch.cat(
        [part.reshape(num_heads * head_dim, *part.shape[2:]) for part in (query, key, value)], dim=0
    ).contiguous()


def _video_tensor_for_targets(
    source_key: str,
    tensor: torch.Tensor,
    config: Mapping[str, Any],
) -> torch.Tensor:
    """Apply source-layout transforms shared by video QKV and the gated FFN."""

    if ".attn.to_qkv." in source_key:
        return _reorder_interleaved_qkv(
            tensor,
            int(config["decoder_num_attention_heads"]),
            int(config["decoder_attention_head_dim"]),
        )
    if ".ff.w1." in source_key:
        gate, up = tensor.chunk(2, dim=0)
        return torch.cat((up, gate), dim=0).contiguous()
    return tensor


def _audio_remove_weight_norm(model: nn.Module) -> int:
    """Convert Diffusers' weight-normalized Conv1d modules to plain weights."""

    from torch.nn.utils import remove_weight_norm

    removed = 0
    for module in list(model.modules()):
        try:
            remove_weight_norm(module)
        except (AttributeError, ValueError):
            continue
        removed += 1
    return removed


def _expected_shapes(model: nn.Module) -> dict[str, tuple[int, ...]]:
    return {key: tuple(value.shape) for key, value in model.state_dict().items()}


def _finish(model: nn.Module, loaded: set[str], expected: Mapping[str, tuple[int, ...]], device: torch.device) -> nn.Module:
    missing = sorted(set(expected) - loaded)
    if missing:
        raise RuntimeError(f"Comfy VAE did not provide {len(missing)} Diffusers tensors: {missing[:8]}")

    still_meta = [name for name, value in model.named_parameters() if value.device.type == "meta"]
    still_meta += [name for name, value in model.named_buffers() if value.device.type == "meta"]
    if still_meta:
        raise RuntimeError(f"Comfy VAE has unmaterialized tensors: {still_meta[:8]}")

    # The generated rotary/filter buffers are tiny; this moves those buffers to
    # the same device without making a second copy of the loaded checkpoint.
    model = model.to(device=device)
    return model.eval()


def build_comfy_video_vae(
    checkpoint: str | Path,
    config: Mapping[str, Any] | str | Path,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Build a Diffusers video VAE from ``minimax_h3_video_vae_fp16.safetensors``."""

    target_device = torch.device(device)
    if target_device.type == "meta":
        raise ValueError("Comfy VAE loading requires a material device, not 'meta'")
    cls = _load_diffusers_class("video")
    config = _config_dict(config)
    model = _build_on_meta(cls, config)
    expected = _expected_shapes(model)
    loaded: set[str] = set()

    from safetensors import safe_open

    with safe_open(str(checkpoint), framework="pt", device=str(target_device)) as source:
        for source_key in source.keys():
            target_keys = _video_target_keys(source_key)
            if not target_keys:
                if source_key == "decoder.mask_token":
                    mask = source.get_tensor(source_key)
                    if bool(mask.detach().abs().max().item()):
                        raise ValueError("video checkpoint contains a nonzero decoder.mask_token")
                continue
            tensor = _materialize(source.get_tensor(source_key), target_device, dtype)
            tensor = _video_tensor_for_targets(source_key, tensor, config)
            if len(target_keys) == 1:
                target = target_keys[0]
                if target not in expected or tuple(tensor.shape) != expected[target]:
                    raise ValueError(
                        f"video tensor {source_key!r} -> {target!r} has shape {tuple(tensor.shape)}, "
                        f"expected {expected.get(target)}"
                    )
                _assign(model, target, tensor)
                loaded.add(target)
                continue

            if tensor.shape[0] % len(target_keys) != 0:
                raise ValueError(f"video fused QKV tensor {source_key!r} has invalid shape {tuple(tensor.shape)}")
            pieces = tensor.chunk(len(target_keys), dim=0)
            for target, piece in zip(target_keys, pieces):
                if target not in expected or tuple(piece.shape) != expected[target]:
                    raise ValueError(
                        f"video tensor {source_key!r} -> {target!r} has shape {tuple(piece.shape)}, "
                        f"expected {expected.get(target)}"
                    )
                _assign(model, target, piece)
                loaded.add(target)

    return _finish(model, loaded, expected, target_device)


def build_comfy_audio_vae(
    checkpoint: str | Path,
    config: Mapping[str, Any] | str | Path,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """Build a Diffusers audio VAE from ``minimax_h3_audio_vae_fp32.safetensors``.

    The released audio model is intentionally kept in float32 by default.  Pass
    another dtype only when accepting the known quality trade-off is deliberate.
    """

    target_device = torch.device(device)
    if target_device.type == "meta":
        raise ValueError("Comfy VAE loading requires a material device, not 'meta'")
    cls = _load_diffusers_class("audio")
    model = _build_on_meta(cls, _config_dict(config))
    _audio_remove_weight_norm(model)
    expected = _expected_shapes(model)
    loaded: set[str] = set()

    from safetensors import safe_open

    with safe_open(str(checkpoint), framework="pt", device=str(target_device)) as source:
        for source_key in source.keys():
            if source_key in _AUDIO_IGNORED:
                continue
            if source_key not in expected:
                raise ValueError(f"audio checkpoint key {source_key!r} is not in the Diffusers VAE")
            tensor = _materialize(source.get_tensor(source_key), target_device, dtype)
            if tuple(tensor.shape) != expected[source_key]:
                raise ValueError(
                    f"audio tensor {source_key!r} has shape {tuple(tensor.shape)}, "
                    f"expected {expected[source_key]}"
                )
            _assign(model, source_key, tensor)
            loaded.add(source_key)

    return _finish(model, loaded, expected, target_device)


__all__ = ["build_comfy_audio_vae", "build_comfy_video_vae"]
