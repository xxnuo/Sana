from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path


H3_ROOT = Path(__file__).resolve().parents[2]
SANA_ROOT = H3_ROOT.parents[1]
CONFIG_ROOT = SANA_ROOT / "config" / "minimax_h3"
THOR_WRAPPER = H3_ROOT / "Thor" / "run_minimax_h3_gpu.sh"
THOR_SHIM = H3_ROOT / "Thor" / "gpu_infer.py"
THOR_SNAPSHOT = H3_ROOT / "Thor" / "SOURCE_SNAPSHOT.json"


def _env(name: str) -> dict[str, str]:
    with (CONFIG_ROOT / name).open("rb") as handle:
        return tomllib.load(handle)["env"]


def _official(name: str) -> dict[str, object]:
    with (CONFIG_ROOT / name).open("rb") as handle:
        return tomllib.load(handle)["official_config"]


def test_dense_thor_profiles_clear_inherited_approximation_switches() -> None:
    for name in ("thor_baseline.toml", "thor_baseline_video.toml", "thor_kernel.toml"):
        env = _env(name)
        assert env["H3_SOL_ATTN_TAU"] == ""
        assert env["H3_CACHE_THRESHOLD"] == ""
        assert env["H3_VAE_TILE_BATCH"] == "0"
        assert env["H3_MIN_AVAILABLE_MIB"] == "81920"
        assert env["H3_BENCH_RUNTIME_MIN_AVAILABLE_MIB"] == "16384"
    assert _env("thor_baseline.toml")["H3_OUTPUT_TYPE"] == "latent"
    assert _env("thor_baseline_video.toml")["H3_OUTPUT_TYPE"] == "pt"
    assert _env("thor_kernel.toml")["H3_OUTPUT_TYPE"] == "latent"


def test_thor_sparse_profile_is_isolated_from_the_delivery_profile() -> None:
    kernel_sol = _env("thor_kernel_sol.toml")
    fullopt = _env("thor_fullopt.toml")
    assert kernel_sol["H3_SOL_ATTN_TAU"] == "1.0"
    assert kernel_sol["H3_SOL_ATTN_THRESH_TYPE"] == "diag"
    assert kernel_sol["H3_SOL_ATTN_DENSE_BLOCKS"] == "2"
    assert kernel_sol["H3_SOL_ATTN_DENSE_FIRST_STEPS"] == "4"
    assert kernel_sol["H3_SOL_ATTN_DENSE_LAST_STEPS"] == "0"
    assert kernel_sol["H3_OUTPUT_TYPE"] == "latent"
    assert kernel_sol["H3_VAE_TILE_BATCH"] == "0"

    assert fullopt["H3_OUTPUT_TYPE"] == "pt"
    assert fullopt["H3_SOL_ATTN_TAU"] == ""
    assert "H3_SOL_ATTN_DENSE_BLOCKS" not in fullopt
    assert "H3_SOL_ATTN_DENSE_FIRST_STEPS" not in fullopt
    assert "H3_SOL_ATTN_DENSE_LAST_STEPS" not in fullopt
    assert fullopt["H3_CACHE_THRESHOLD"] == ""
    assert fullopt["H3_VAE_TILE_BATCH"] == "4"
    assert fullopt["H3_MIN_AVAILABLE_MIB"] == "81920"
    assert fullopt["H3_BENCH_RUNTIME_MIN_AVAILABLE_MIB"] == "16384"


def test_thor_profiles_report_the_actual_vae_precision_contract() -> None:
    for name in (
        "thor_baseline.toml",
        "thor_baseline_video.toml",
        "thor_kernel.toml",
        "thor_fullopt.toml",
        "thor_kernel_sol.toml",
    ):
        official = _official(name)
        assert official["suite"] == "smoke"
        assert official["vae_dtype"] == "fp32"
        assert official["vae_autocast_dtype"] == "fp16"
        assert official["video_vae_autocast_dtype"] == "fp16"
        assert official["audio_vae_autocast_dtype"] == "disabled"


def test_thor_wrapper_only_auto_discovers_the_latent_capable_diffusers_pin() -> None:
    source = THOR_WRAPPER.read_text(encoding="utf-8")
    assert "diffusers-abc5e9bf/src" in source
    assert "diffusers-pr14355" not in source
    assert 'get_workflow("t2va")' in source
    # The pinned decoder stores the discriminator on ``block_state``.  Check that
    # the wrapper performs a semantic AST check instead of depending on one local
    # variable spelling.
    assert "has_latent_output_branch" in source
    assert "ast.parse" in source
    assert "isinstance(operator, ast.Eq)" in source


def test_thor_wrapper_checks_uma_headroom_before_python_preflight() -> None:
    source = THOR_WRAPPER.read_text(encoding="utf-8")
    memory_check = source.index('MEM_AVAILABLE_KIB=""')
    python_preflight = source.index('"${UV_BIN}" run --no-project')
    assert memory_check < python_preflight
    assert "H3_MIN_AVAILABLE_MIB" in source
    assert "H3_BENCH_RUNTIME_MIN_AVAILABLE_MIB" in source
    assert "MemAvailable" in source
    assert 'H3_PREFLIGHT_ONLY:-0' in source


def test_thor_wrapper_uses_uma_friendly_qwen_loading_defaults() -> None:
    source = THOR_WRAPPER.read_text(encoding="utf-8")
    assert "export HF_DEACTIVATE_ASYNC_LOAD=1" in source
    assert "export H3_DISABLE_ALLOCATOR_WARMUP=1" in source
    assert "export H3_LOAD_ORDER=qwen_first" in source
    assert "export TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1" in source


def test_thor_wrapper_owns_the_official_workload_switch() -> None:
    source = THOR_WRAPPER.read_text(encoding="utf-8")
    assert 'H3_OFFICIAL="${H3_OFFICIAL:-0}"' in source
    assert 'H3_OFFICIAL must be 0 or 1' in source
    assert 'set H3_OFFICIAL=1 instead of passing --official directly' in source
    assert 'INFER_ARGS+=(--official)' in source
    for name in (
        "thor_baseline.toml",
        "thor_baseline_video.toml",
        "thor_kernel.toml",
        "thor_fullopt.toml",
        "thor_kernel_sol.toml",
    ):
        assert _env(name)["H3_OFFICIAL"] == "0"


def test_thor_wrapper_pins_assets_diffusers_and_absolute_python() -> None:
    source = THOR_WRAPPER.read_text(encoding="utf-8")
    assert "export H3_PREFLIGHT_VERIFY_SHA256=1" in source
    assert "metadata_expected = {" in source
    assert "009da31ca91b3737d70e954cec1008d92732334b104c144fe10c279f0e4121c1" in source
    assert 'PYTHON_RESOLVED="$(command -v "${PYTHON_BIN}"' in source
    # Preserve a venv's `bin/python` symlink: resolving it can discard the
    # site-packages path on Thor's uv-managed Unsloth environment.
    assert 'PYTHON_BIN="${PYTHON_RESOLVED}"' in source
    assert 'asset_manifest_path.write_text(' in source
    assert 'export H3_ASSET_MANIFEST_SHA256=' in source


def test_thor_wrapper_pins_the_effective_dit_override_and_digest() -> None:
    source = THOR_WRAPPER.read_text(encoding="utf-8")
    assert 'export H3_DIT_CHECKPOINT="${PINNED_DIT_CHECKPOINT}"' in source
    assert 'H3_DIT_CHECKPOINT must resolve to the pinned Thor artifact' in source
    assert 'export H3_VERIFY_DIT_SHA256=1' in source
    assert (
        'export H3_EXPECTED_DIT_SHA256='
        '12944c1f7791637e7de12208aef04da82bd26b95271b1b47d817364315ade993'
    ) in source


def test_thor_python_shim_rejects_direct_invocation_before_torch_import() -> None:
    env = os.environ.copy()
    env.pop("H3_THOR_WRAPPER_ACTIVE", None)
    result = subprocess.run(
        [sys.executable, str(THOR_SHIM)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "direct gpu_infer.py invocation is unsupported" in result.stderr


def test_thor_python_shim_rejects_a_forged_active_flag() -> None:
    env = os.environ.copy()
    env["H3_THOR_WRAPPER_ACTIVE"] = "1"
    result = subprocess.run(
        [sys.executable, str(THOR_SHIM)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "direct gpu_infer.py invocation is unsupported" in result.stderr


def test_thor_wrapper_execs_inference_with_tee_process_substitution() -> None:
    source = THOR_WRAPPER.read_text(encoding="utf-8")
    assert 'exec > >(tee "${OUT_DIR}/run.log") 2>&1' in source
    assert 'exec "${UV_BIN}" run --no-project --python "${PYTHON_BIN}"' in source
    assert 'gpu_infer.py" "$@" 2>&1 | tee' not in source
    assert "H3_THOR_WRAPPER_TOKEN" in source
    assert "H3_THOR_WRAPPER_RECEIPT" in source


def test_thor_wrapper_pins_toolchain_cache_and_vae_autocast_identity() -> None:
    source = THOR_WRAPPER.read_text(encoding="utf-8")
    assert "PINNED_CUDA_BIN=/usr/local/cuda/bin" in source
    assert 'export TRITON_PTXAS_PATH="${CUDA_BIN}/ptxas"' in source
    assert 'CACHE_ROOT="${OUT_DIR}/.cache"' in source
    assert 'export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"' in source
    assert "decoder_autocast_dtypes" in source
    assert 'video_autocast != {"torch.float16"}' in source
    assert "if audio_autocast:" in source
    assert "export H3_VIDEO_VAE_AUTOCAST_DTYPE=torch.float16" in source
    assert "export H3_AUDIO_VAE_AUTOCAST_DTYPE=disabled" in source


def test_thor_source_snapshot_verifies_launcher_core() -> None:
    snapshot = json.loads(THOR_SNAPSHOT.read_text(encoding="utf-8"))
    assert snapshot["upstream_commit"] == "d0c0a4685ab5dc2336d18b7213d85f13def92418"
    assert snapshot["diffusers"]["commit"] == "abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc"
    for relative, expected in snapshot["core_sha256"].items():
        actual = hashlib.sha256((THOR_SNAPSHOT.parent / relative).read_bytes()).hexdigest()
        assert actual == expected, relative


def test_thor_benchmark_identity_includes_runtime_source_digest() -> None:
    source = THOR_WRAPPER.read_text(encoding="utf-8")
    assert "H3_RUNTIME_SOURCE_SHA256" in source
    assert 'H3_SANA_COMMIT="snapshot:${H3_SOURCE_SNAPSHOT_SHA256}:runtime:${H3_RUNTIME_SOURCE_SHA256}"' in source
    assert "LC_ALL=C sort -z" in source
