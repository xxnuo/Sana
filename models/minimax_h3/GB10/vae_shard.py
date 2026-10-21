"""Batch the video VAE's tile decodes instead of launching them one at a time.

Once the cache and Sol-Attn are on, the denoising loop stops being the whole request: the
fixed remainder is 31.8 s of which the video decode is 29.4 s — 92.6%, and a sixth of the
whole request. Inside it, `_decode_clip` walks a grid of tiles and calls the decoder once per
tile. At 832x480 that is a 3x4 grid, twelve sequential launches of about 2.45 s each.

Every tile is independent and `_split_tiles` gives them all the same size — the slack goes
into the overlaps, not into shorter edge tiles — so they can go through the decoder as one
batch. This is the same change Sol-Engine's `vae_shard.py` makes, where it measures 1.93x and
reports the result bit-identical.

Bit-identical is a claim about arithmetic, not about kernels: batching can move cuDNN and
cuBLAS onto different algorithms, and their reductions need not associate the same way.
`--check` verifies it here rather than inheriting the claim.
"""

from __future__ import annotations

import paths

paths.setup()

import torch


def _new_tile_stats(batch: int = 0) -> dict[str, int | bool]:
    """Create counters owned by one VAE instance."""
    return {
        "configured_batch_size": int(batch),
        "decode_clip_calls": 0,
        "multi_tile_clip_calls": 0,
        "logical_tiles": 0,
        "decoder_batch_calls": 0,
        "multi_tile_batch_calls": 0,
        "max_effective_batch_size": 0,
        "hit": False,
    }


def reset_tile_stats(vae) -> None:
    """Clear this VAE's tile counters without touching any other VAE."""
    batch = int(getattr(vae, "_h3_tile_batch", 0) or 0)
    vae._h3_tile_stats = _new_tile_stats(batch)


def read_tile_stats(vae) -> dict[str, int | bool]:
    """Return a detached snapshot of this VAE's tile counters."""
    stats = getattr(vae, "_h3_tile_stats", None)
    if stats is None:
        reset_tile_stats(vae)
        stats = vae._h3_tile_stats
    snapshot = dict(stats)
    snapshot["configured_batch_size"] = int(getattr(vae, "_h3_tile_batch", 0) or 0)
    snapshot["hit"] = bool(snapshot["multi_tile_batch_calls"] > 0)
    return snapshot


def _tile_stats(vae) -> dict[str, int | bool]:
    stats = getattr(vae, "_h3_tile_stats", None)
    if stats is None:
        reset_tile_stats(vae)
        stats = vae._h3_tile_stats
    return stats


def _batched_decode_clip(self, z: torch.Tensor) -> torch.Tensor:
    """`_decode_clip`, with the tile loop replaced by batched decoder calls."""
    stats = _tile_stats(self)
    stats["decode_clip_calls"] += 1
    if not self.use_tiling:
        return self.decoder(self.post_quant_conv(z))

    height = z.shape[-2] * self.spatial_compression_ratio
    width = z.shape[-1] * self.spatial_compression_ratio
    y_indices, y_lengths, y_overlaps = self._split_tiles(
        height, self.tile_sample_min_height, self.tile_sample_min_overlap_height
    )
    x_indices, x_lengths, x_overlaps = self._split_tiles(
        width, self.tile_sample_min_width, self.tile_sample_min_overlap_width
    )

    ratio = self.spatial_compression_ratio
    tiles = [
        z[..., i_pos // ratio : i_pos // ratio + i_len // ratio,
             j_pos // ratio : j_pos // ratio + j_len // ratio]
        for i_pos, i_len in zip(y_indices, y_lengths)
        for j_pos, j_len in zip(x_indices, x_lengths)
    ]
    tile_count = len(tiles)
    stats["logical_tiles"] += tile_count
    if tile_count > 1:
        stats["multi_tile_clip_calls"] += 1

    # A cap rather than one batch of everything: the ViT decoder's activations scale with the
    # batch, and this runs on a part whose memory is shared with the host.
    limit = getattr(self, "_h3_tile_batch", 0) or len(tiles)
    decoded = []
    for start in range(0, len(tiles), limit):
        group = tiles[start : start + limit]
        batch = torch.cat(group, dim=0)
        effective_batch = len(group) * int(group[0].shape[0])
        stats["decoder_batch_calls"] += 1
        if len(group) > 1:
            stats["multi_tile_batch_calls"] += 1
        stats["max_effective_batch_size"] = max(
            stats["max_effective_batch_size"], effective_batch
        )
        out = self.decoder(self.post_quant_conv(batch))
        decoded.extend(out.split(group[0].shape[0], dim=0))

    columns = len(x_indices)
    rows = [decoded[i * columns : (i + 1) * columns] for i in range(len(y_indices))]
    return self._stitch_tiles(rows, y_overlaps, x_overlaps)


def patch_batched_tiles(vae, batch: int = 0) -> None:
    """Route this VAE's tiled decode through the batched implementation.

    `batch` caps how many tiles go through the decoder at once; 0 means all of them.
    """
    if isinstance(batch, bool) or int(batch) != batch or int(batch) < 0:
        raise ValueError(f"tile batch size must be a non-negative integer, got {batch!r}")
    batch = int(batch)

    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import (
        AutoencoderKLMiniMaxH3,
    )

    vae._h3_tile_batch = batch
    reset_tile_stats(vae)
    state = getattr(AutoencoderKLMiniMaxH3, "_h3_batched_tiles_state", None)
    current = AutoencoderKLMiniMaxH3._decode_clip
    if isinstance(state, dict) and current is state.get("patched"):
        return
    if state is not None:
        # A different hook owns the class method now. Do not silently overwrite it with a stale
        # restoration record; installing this optimisation on top would make unpatch order unsafe.
        raise RuntimeError(
            "AutoencoderKLMiniMaxH3._decode_clip was replaced after the H3 tile patch; "
            "remove the external hook before enabling batched tiles"
        )
    patched = _batched_decode_clip
    AutoencoderKLMiniMaxH3._h3_decode_clip_original = current
    AutoencoderKLMiniMaxH3._decode_clip = patched
    AutoencoderKLMiniMaxH3._h3_batched_tiles = True
    AutoencoderKLMiniMaxH3._h3_batched_tiles_state = {
        "original": current,
        "patched": patched,
    }


def unpatch_batched_tiles(vae=None) -> None:
    from diffusers.models.autoencoders.autoencoder_kl_minimax_h3 import (
        AutoencoderKLMiniMaxH3,
    )

    state = getattr(AutoencoderKLMiniMaxH3, "_h3_batched_tiles_state", None)
    if isinstance(state, dict):
        if AutoencoderKLMiniMaxH3._decode_clip is state.get("patched"):
            AutoencoderKLMiniMaxH3._decode_clip = state["original"]
            changed = True
        else:
            changed = False
        if getattr(AutoencoderKLMiniMaxH3, "_h3_batched_tiles_state", None) is state:
            delattr(AutoencoderKLMiniMaxH3, "_h3_batched_tiles_state")
        AutoencoderKLMiniMaxH3._h3_batched_tiles = False
    else:
        changed = False
    if vae is not None:
        vae._h3_tile_batch = 0
        reset_tile_stats(vae)
    return changed
