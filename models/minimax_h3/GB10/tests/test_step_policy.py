from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path


GB10_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GB10_ROOT))

# ``gpu_infer`` resolves the checkpoint paths at import time. Keep this unit
# test independent from a downloaded 300+ GiB model set by providing only the
# config files needed for those top-level lookups.
_TEST_ROOT = Path(tempfile.mkdtemp(prefix="h3-step-policy-"))
(_TEST_ROOT / "qwen").mkdir()
(_TEST_ROOT / "model").mkdir()
(_TEST_ROOT / "comfy" / "diffusion_models").mkdir(parents=True)
(_TEST_ROOT / "qwen" / "config.json").write_text(
    json.dumps({"quantization_config": {"ignored_layers": []}})
)
os.environ.setdefault("H3_MODEL_ROOT", str(_TEST_ROOT / "model"))
os.environ.setdefault("H3_QWEN_FP8", str(_TEST_ROOT / "qwen"))
os.environ.setdefault("H3_COMFY_ROOT", str(_TEST_ROOT / "comfy"))

import gpu_infer  # noqa: E402


def test_step_counts_maps_single_denoise_to_two_scheduler_points():
    assert gpu_infer._step_counts(1) == (2, 1)
    assert gpu_infer._step_counts(2) == (2, 1)
    assert gpu_infer._step_counts(8) == (8, 7)


def test_step_counts_rejects_nonpositive_requests():
    try:
        gpu_infer._step_counts(0)
    except ValueError as exc:
        assert "steps must be >= 1" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("nonpositive step count was accepted")


def test_sol_sparse_opportunity_respects_dense_prefix_and_suffix():
    assert not gpu_infer._sol_sparse_opportunity(1, dense_first_steps=2)
    assert gpu_infer._sol_sparse_opportunity(3, dense_first_steps=2)
    assert not gpu_infer._sol_sparse_opportunity(
        3, dense_first_steps=2, dense_last_steps=1
    )
