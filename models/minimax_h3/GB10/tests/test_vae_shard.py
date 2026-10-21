from __future__ import annotations

import math
import sys
from pathlib import Path
from types import ModuleType

import torch

GB10_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GB10_ROOT))

import vae_shard


class _RecordingDecoder:
    def __init__(self, ratio: int = 16):
        self.ratio = ratio
        self.batch_sizes: list[int] = []

    def __call__(self, value: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(int(value.shape[0]))
        return torch.zeros(
            value.shape[0],
            3,
            value.shape[2],
            value.shape[-2] * self.ratio,
            value.shape[-1] * self.ratio,
            dtype=value.dtype,
        )


class _FakeVAE:
    use_tiling = True
    spatial_compression_ratio = 16
    tile_sample_min_height = 256
    tile_sample_min_width = 256
    tile_sample_min_overlap_height = 64
    tile_sample_min_overlap_width = 64

    def __init__(self, batch: int):
        self._h3_tile_batch = batch
        self.decoder = _RecordingDecoder(self.spatial_compression_ratio)

    @staticmethod
    def post_quant_conv(value: torch.Tensor) -> torch.Tensor:
        return value

    def _split_tiles(self, length: int, tile_size: int, min_overlap: int):
        if tile_size >= length:
            return [0], [length], []
        num_tiles = math.ceil(length / tile_size)
        while tile_size * num_tiles - min_overlap * (num_tiles - 1) - length < 0:
            num_tiles += 1
        overlaps = [min_overlap] * (num_tiles - 1)
        remaining = tile_size * num_tiles - sum(overlaps) - length
        for index in range(remaining // self.spatial_compression_ratio):
            overlaps[index % (num_tiles - 1)] += self.spatial_compression_ratio
        starts = [0]
        for index in range(num_tiles - 1):
            starts.append(starts[-1] + tile_size - overlaps[index])
        return starts, [tile_size] * num_tiles, overlaps

    @staticmethod
    def _stitch_tiles(rows, height_overlaps, width_overlaps):
        stitched_rows = []
        for row_index, row in enumerate(rows):
            pieces = []
            for column_index, tile in enumerate(row):
                if row_index < len(rows) - 1:
                    tile = tile[..., : -height_overlaps[row_index], :]
                if column_index < len(row) - 1:
                    tile = tile[..., :, : -width_overlaps[column_index]]
                pieces.append(tile)
            stitched_rows.append(torch.cat(pieces, dim=-1))
        return torch.cat(stitched_rows, dim=-2)


def _decode(vae: _FakeVAE, width_latents: int = 28):
    z = torch.zeros(1, 24, 7, 16, width_latents)
    return vae_shard._batched_decode_clip(vae, z)


def test_batched_tile_stats_capture_two_tiles_and_one_decoder_launch():
    vae = _FakeVAE(batch=4)

    output = _decode(vae)

    assert tuple(output.shape) == (1, 3, 7, 256, 448)
    assert vae.decoder.batch_sizes == [2]
    assert vae_shard.read_tile_stats(vae) == {
        "configured_batch_size": 4,
        "decode_clip_calls": 1,
        "multi_tile_clip_calls": 1,
        "logical_tiles": 2,
        "decoder_batch_calls": 1,
        "multi_tile_batch_calls": 1,
        "max_effective_batch_size": 2,
        "hit": True,
    }


def test_batch_one_and_single_tile_are_not_reported_as_batched_hits():
    batch_one = _FakeVAE(batch=1)
    _decode(batch_one)
    stats = vae_shard.read_tile_stats(batch_one)
    assert batch_one.decoder.batch_sizes == [1, 1]
    assert stats["logical_tiles"] == 2
    assert stats["decoder_batch_calls"] == 2
    assert stats["multi_tile_batch_calls"] == 0
    assert stats["hit"] is False

    single_tile = _FakeVAE(batch=4)
    _decode(single_tile, width_latents=16)
    stats = vae_shard.read_tile_stats(single_tile)
    assert single_tile.decoder.batch_sizes == [1]
    assert stats["logical_tiles"] == 1
    assert stats["multi_tile_clip_calls"] == 0
    assert stats["hit"] is False


def test_tile_stats_are_instance_scoped_and_resettable():
    first = _FakeVAE(batch=4)
    second = _FakeVAE(batch=4)
    _decode(first)

    snapshot = vae_shard.read_tile_stats(first)
    snapshot["logical_tiles"] = 999
    assert vae_shard.read_tile_stats(first)["logical_tiles"] == 2
    assert vae_shard.read_tile_stats(second)["logical_tiles"] == 0

    vae_shard.reset_tile_stats(first)
    assert vae_shard.read_tile_stats(first) == {
        "configured_batch_size": 4,
        "decode_clip_calls": 0,
        "multi_tile_clip_calls": 0,
        "logical_tiles": 0,
        "decoder_batch_calls": 0,
        "multi_tile_batch_calls": 0,
        "max_effective_batch_size": 0,
        "hit": False,
    }


def test_unpatch_preserves_external_decode_replacement(monkeypatch):
    class AutoencoderKLMiniMaxH3:
        @staticmethod
        def _decode_clip(value):
            return value

    module = ModuleType(
        "diffusers.models.autoencoders.autoencoder_kl_minimax_h3"
    )
    module.AutoencoderKLMiniMaxH3 = AutoencoderKLMiniMaxH3
    monkeypatch.setitem(
        sys.modules,
        "diffusers.models.autoencoders.autoencoder_kl_minimax_h3",
        module,
    )
    original = AutoencoderKLMiniMaxH3._decode_clip
    vae = _FakeVAE(batch=4)

    vae_shard.patch_batched_tiles(vae, batch=4)
    assert AutoencoderKLMiniMaxH3._decode_clip is vae_shard._batched_decode_clip

    external = lambda value: value  # noqa: E731
    AutoencoderKLMiniMaxH3._decode_clip = external
    assert not vae_shard.unpatch_batched_tiles(vae)
    assert AutoencoderKLMiniMaxH3._decode_clip is external
    assert vae._h3_tile_batch == 0
    assert not hasattr(AutoencoderKLMiniMaxH3, "_h3_batched_tiles_state")

    # The original remains available only as historical metadata; unpatch must not restore it
    # over the external hook.
    assert AutoencoderKLMiniMaxH3._h3_decode_clip_original is original
