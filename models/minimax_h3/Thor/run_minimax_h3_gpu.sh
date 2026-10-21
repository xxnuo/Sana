#!/usr/bin/env bash
# MiniMax-H3 single-card Jetson Thor (SM110) entrypoint.
#
# The wrapper is deliberately strict: the Triton compiler must use the host
# CUDA 13 toolchain, and a sparse run must never be reported when the process is
# on a different architecture or silently using an untracked compiler.
set -euo pipefail

: "${OUT_DIR:?OUT_DIR must be set by scripts/launch_config.py}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANA_ROOT="$(cd "${HERE}/../../.." && pwd)"
GB10_ROOT="$(cd "${HERE}/../GB10" && pwd)"
if [[ -n "${H3_TOOL_ROOT:-}" ]]; then
  TOOL_ROOT="$(cd "${H3_TOOL_ROOT}" 2>/dev/null && pwd -P || true)"
  [[ -n "${TOOL_ROOT}" ]] || {
    printf '[thor] H3_TOOL_ROOT does not resolve to a directory: %s\n' "${H3_TOOL_ROOT}" >&2
    exit 2
  }
else
  TOOL_ROOT="$(cd "${SANA_ROOT}/../.." && pwd)"
fi
export H3_TOOL_ROOT="${TOOL_ROOT}"

mkdir -p "${OUT_DIR}"
export H3_OUTPUT_DIR="${OUT_DIR}"
# The driver is shared with GB10, so pin the Thor identity before it writes benchmark metadata.
export H3_TAG="thor_sm110"
export H3_VARIANT="diffusers_modular_pruned_fp8_scaled_single_thor_sm110"
H3_OFFICIAL="${H3_OFFICIAL:-0}"
[[ "${H3_OFFICIAL}" == "0" || "${H3_OFFICIAL}" == "1" ]] || {
  printf '[thor] H3_OFFICIAL must be 0 or 1, got %s\n' "${H3_OFFICIAL}" >&2
  exit 2
}
for argument in "$@"; do
  if [[ "${argument}" == "--official" ]]; then
    printf '[thor] set H3_OFFICIAL=1 instead of passing --official directly\n' >&2
    exit 2
  fi
done
export H3_OFFICIAL

# Jetson Thor's GPU and CPU share the same physical memory. Refuse a launch
# before importing CUDA or allocating weights when another resident service has
# already consumed the headroom needed by the roughly 74 GiB H3 peak.
MIN_AVAILABLE_MIB="${H3_MIN_AVAILABLE_MIB:-81920}"
[[ "${MIN_AVAILABLE_MIB}" =~ ^[0-9]+$ ]] || {
  printf '[thor] H3_MIN_AVAILABLE_MIB must be a non-negative integer, got %s\n' \
    "${MIN_AVAILABLE_MIB}" >&2
  exit 2
}
BENCH_RUNTIME_MIN_AVAILABLE_MIB="${H3_BENCH_RUNTIME_MIN_AVAILABLE_MIB:-16384}"
[[ "${BENCH_RUNTIME_MIN_AVAILABLE_MIB}" =~ ^[1-9][0-9]*$ ]] || {
  printf '[thor] H3_BENCH_RUNTIME_MIN_AVAILABLE_MIB must be a positive integer, got %s\n' \
    "${BENCH_RUNTIME_MIN_AVAILABLE_MIB}" >&2
  exit 2
}
MEM_AVAILABLE_KIB=""
while read -r key value _unit; do
  if [[ "${key}" == "MemAvailable:" ]]; then
    MEM_AVAILABLE_KIB="${value}"
    break
  fi
done < /proc/meminfo
[[ "${MEM_AVAILABLE_KIB}" =~ ^[0-9]+$ ]] || {
  printf '[thor] could not read MemAvailable from /proc/meminfo\n' >&2
  exit 1
}
MEM_AVAILABLE_MIB="$((MEM_AVAILABLE_KIB / 1024))"
export H3_MIN_AVAILABLE_MIB="${MIN_AVAILABLE_MIB}"
export H3_PREFLIGHT_MEMAVAILABLE_MIB="${MEM_AVAILABLE_MIB}"
export H3_BENCH_RUNTIME_MIN_AVAILABLE_MIB="${BENCH_RUNTIME_MIN_AVAILABLE_MIB}"
if [[ "${H3_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  printf '[thor] asset-only preflight: memory gate not required (MemAvailable=%s MiB)\n' \
    "${MEM_AVAILABLE_MIB}"
elif (( MIN_AVAILABLE_MIB > 0 && MEM_AVAILABLE_MIB < MIN_AVAILABLE_MIB )); then
  printf '[thor] insufficient UMA memory: MemAvailable=%s MiB, required=%s MiB\n' \
    "${MEM_AVAILABLE_MIB}" "${MIN_AVAILABLE_MIB}" >&2
  printf '[thor] stop competing workloads or explicitly lower H3_MIN_AVAILABLE_MIB for a diagnostic\n' >&2
  exit 1
fi
printf '[thor] UMA preflight: MemAvailable=%s MiB, required=%s MiB\n' \
  "${MEM_AVAILABLE_MIB}" "${MIN_AVAILABLE_MIB}"

SANA_GIT_IDENTITY=0
if command -v git >/dev/null 2>&1 && git -C "${SANA_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  export H3_SANA_COMMIT="$(git -C "${SANA_ROOT}" rev-parse HEAD)"
  if [[ -n "$(git -C "${SANA_ROOT}" status --porcelain --untracked-files=normal)" ]]; then
    export H3_SANA_DIRTY=1
  else
    export H3_SANA_DIRTY=0
  fi
  SANA_GIT_IDENTITY=1
fi

if [[ -n "${H3_PROMPT_FILE:-}" && "${H3_PROMPT_FILE}" != /* ]]; then
  export H3_PROMPT_FILE="${SANA_ROOT}/${H3_PROMPT_FILE}"
fi

PINNED_SOL_ATTN_ROOT="${SANA_ROOT}/techniques/sparse_backends"
if [[ -n "${H3_SOL_ATTN_ROOT:-}" ]]; then
  requested_sol_attn="$(readlink -f -- "${H3_SOL_ATTN_ROOT}" 2>/dev/null || true)"
  pinned_sol_attn="$(readlink -f -- "${PINNED_SOL_ATTN_ROOT}" 2>/dev/null || true)"
  if [[ -z "${requested_sol_attn}" || "${requested_sol_attn}" != "${pinned_sol_attn}" ]]; then
    printf '[thor] H3_SOL_ATTN_ROOT must resolve to the bundled source: %s\n' \
      "${PINNED_SOL_ATTN_ROOT}" >&2
    exit 2
  fi
fi
SOL_ATTN_ROOT="${PINNED_SOL_ATTN_ROOT}"
DIFFUSERS_SRC="${H3_DIFFUSERS_SRC:-${SANA_ROOT}/models/minimax_h3/GB200/diffusers_src/src}"
[[ -d "${SOL_ATTN_ROOT}/sol_attn" ]] || {
  printf '[thor] Sol-Attn package not found under %s\n' "${SOL_ATTN_ROOT}" >&2
  exit 1
}
export H3_SOL_ATTN_ROOT="${SOL_ATTN_ROOT}"

# The reference Thor hosts keep the downloaded assets and the pinned Diffusers
# source under /home/nvidia/Downloads.  Prefer explicit overrides, then discover
# those conventional locations so a config run does not depend on a long list of
# machine-specific exports.
if [[ -z "${H3_MODEL_ROOT:-}" && -d /home/nvidia/Downloads/MiniMaxAI--MiniMax-H3 ]]; then
  export H3_MODEL_ROOT=/home/nvidia/Downloads/MiniMaxAI--MiniMax-H3
fi
if [[ -z "${H3_QWEN_FP8:-}" && -d /home/nvidia/Downloads/Qwen--Qwen3-VL-32B-Instruct-FP8 ]]; then
  export H3_QWEN_FP8=/home/nvidia/Downloads/Qwen--Qwen3-VL-32B-Instruct-FP8
fi
if [[ -z "${H3_COMFY_ROOT:-}" && -d /home/nvidia/Downloads/Comfy-Org--MiniMax-H3 ]]; then
  export H3_COMFY_ROOT=/home/nvidia/Downloads/Comfy-Org--MiniMax-H3
fi

# The Thor profile is built around one exact pruned DiT artifact.  GB10's shared
# assembler historically accepted H3_DIT_CHECKPOINT as an escape hatch; keeping
# that variable here would let an inherited shell environment pass the fixed
# preflight while loading a different file.  Normalize it to the checked file
# and reject any override that does not resolve to the same inode.
[[ -n "${H3_COMFY_ROOT:-}" ]] || {
  printf '[thor] H3_COMFY_ROOT is required for the pinned DiT checkpoint\n' >&2
  exit 1
}
PINNED_DIT_CHECKPOINT="${H3_COMFY_ROOT}/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
if [[ -n "${H3_DIT_CHECKPOINT:-}" ]]; then
  requested_dit="$(readlink -f -- "${H3_DIT_CHECKPOINT}" 2>/dev/null || true)"
  pinned_dit="$(readlink -f -- "${PINNED_DIT_CHECKPOINT}" 2>/dev/null || true)"
  if [[ -z "${requested_dit}" || -z "${pinned_dit}" || "${requested_dit}" != "${pinned_dit}" ]]; then
    printf '[thor] H3_DIT_CHECKPOINT must resolve to the pinned Thor artifact: %s\n' \
      "${PINNED_DIT_CHECKPOINT}" >&2
    exit 2
  fi
fi
export H3_DIT_CHECKPOINT="${PINNED_DIT_CHECKPOINT}"
export H3_VERIFY_DIT_SHA256=1
export H3_EXPECTED_DIT_SHA256=12944c1f7791637e7de12208aef04da82bd26b95271b1b47d817364315ade993
if [[ "${DIFFUSERS_SRC}" == "${SANA_ROOT}/models/minimax_h3/GB200/diffusers_src/src" && ! -d "${DIFFUSERS_SRC}" ]]; then
  # The abc5e9bf checkout is the only pinned source known to retain latent
  # output for the Thor bring-up.  Do not silently select newer snapshots:
  # some register transformer_ref or remove latent decode and fail after a
  # large model allocation.
  for candidate in /home/nvidia/Downloads/diffusers-abc5e9bf/src; do
    if [[ -d "${candidate}/diffusers" ]]; then
      DIFFUSERS_SRC="${candidate}"
      break
    fi
  done
fi
if [[ -d "${DIFFUSERS_SRC}/diffusers" ]]; then
  export H3_DIFFUSERS_SRC="${DIFFUSERS_SRC}"
  DIFFUSERS_PATH="${DIFFUSERS_SRC}"
else
  printf '[thor] pinned latent-capable Diffusers source not found: %s\n' "${DIFFUSERS_SRC}" >&2
  printf '[thor] set H3_DIFFUSERS_SRC to the abc5e9bf src checkout\n' >&2
  exit 1
fi
export PYTHONPATH="${HERE}:${GB10_ROOT}:${DIFFUSERS_PATH}:${SOL_ATTN_ROOT}:${PYTHONPATH:-}"
export H3_DIFFUSERS_COMMIT="abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc"
export H3_DIFFUSERS_TREE_SHA256="009da31ca91b3737d70e954cec1008d92732334b104c144fe10c279f0e4121c1"
export H3_PREFLIGHT_VERIFY_SHA256=1

SOURCE_SNAPSHOT="${HERE}/SOURCE_SNAPSHOT.json"
[[ -f "${SOURCE_SNAPSHOT}" ]] || {
  printf '[thor] source snapshot not found: %s\n' "${SOURCE_SNAPSHOT}" >&2
  exit 1
}
export H3_SOURCE_SNAPSHOT_SHA256="$(sha256sum "${SOURCE_SNAPSHOT}" | awk '{print $1}')"
mapfile -d '' RUNTIME_SOURCE_FILES < <(
  {
    find \
      "${HERE}" \
      "${GB10_ROOT}" \
      "${SOL_ATTN_ROOT}/sol_attn" \
      -type f ! -type l \
      \( -name '*.py' -o -name '*.sh' -o -name 'SOURCE_SNAPSHOT.json' \) \
      ! -path '*/__pycache__/*' \
      ! -path '*/tests/*' \
      -print0
    find "${SANA_ROOT}/config/minimax_h3" -maxdepth 1 -type f ! -type l \
      -name 'thor_*.toml' -print0 2>/dev/null || true
    for relative in \
      scripts/launch_config.py \
      scripts/collect_run.py \
      scripts/run.py; do
      candidate="${SANA_ROOT}/${relative}"
      [[ -f "${candidate}" && ! -L "${candidate}" ]] && printf '%s\0' "${candidate}"
    done
    for relative in \
      scripts/benchmark_sol_engine_h3_thor.sh \
      scripts/h3_benchmark_gates.py \
      scripts/compare_h3_latents.py \
      scripts/compare_h3_vae_tiles.py \
      scripts/summarize_h3_benchmarks.py \
      scripts/h3_benchmark_process_group.sh; do
      candidate="${TOOL_ROOT}/${relative}"
      [[ -f "${candidate}" && ! -L "${candidate}" ]] && printf '%s\0' "${candidate}"
    done
  } | LC_ALL=C sort -zu
)
(( ${#RUNTIME_SOURCE_FILES[@]} > 0 )) || {
  printf '[thor] no runtime source files found for provenance hashing\n' >&2
  exit 1
}
for source_file in "${RUNTIME_SOURCE_FILES[@]}"; do
  if [[ "${source_file#${TOOL_ROOT}/}" == "${source_file}" ]]; then
    printf '[thor] provenance input is outside H3_TOOL_ROOT: %s\n' "${source_file}" >&2
    exit 1
  fi
done
export H3_RUNTIME_SOURCE_SHA256="$({
  for source_file in "${RUNTIME_SOURCE_FILES[@]}"; do
    relative="${source_file#${TOOL_ROOT}/}"
    printf '%s  %s\n' "$(sha256sum "${source_file}" | awk '{print $1}')" "${relative}"
  done
} | sha256sum | awk '{print $1}')"
if (( ! SANA_GIT_IDENTITY )); then
  export H3_SANA_COMMIT="snapshot:${H3_SOURCE_SNAPSHOT_SHA256}:runtime:${H3_RUNTIME_SOURCE_SHA256}"
  export H3_SANA_DIRTY=0
fi

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export SOL_ATTN_STRICT=1
# Transformers' model-sized allocator warmup is counterproductive on Thor UMA: it reserves
# the whole quantized conditioner before shard loading and can trigger a driver-level peak.
export HF_DEACTIVATE_ASYNC_LOAD=1
export H3_DISABLE_ALLOCATOR_WARMUP=1
export H3_LOAD_ORDER=qwen_first
# SM110 has no validated DeepGEMM path in this runtime; use the bundled Triton
# FP8 implementation for the Qwen conditioner as well as the DiT.
export TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1
export H3_REQUIRE_SM110=1

# Triton's bundled Blackwell ptxas may predate SM110. Use the host JetPack
# toolchain for every assembler/introspection operation and make the cache run
# local to the generated bundle so separate experiments cannot reuse a foreign
# architecture's code.
PINNED_CUDA_BIN=/usr/local/cuda/bin
if [[ -n "${CUDA_BIN:-}" ]]; then
  requested_cuda="$(readlink -f -- "${CUDA_BIN}" 2>/dev/null || true)"
  pinned_cuda="$(readlink -f -- "${PINNED_CUDA_BIN}" 2>/dev/null || true)"
  if [[ -z "${requested_cuda}" || "${requested_cuda}" != "${pinned_cuda}" ]]; then
    printf '[thor] CUDA_BIN must resolve to the JetPack toolchain: %s\n' \
      "${PINNED_CUDA_BIN}" >&2
    exit 2
  fi
fi
CUDA_BIN="${PINNED_CUDA_BIN}"
for tool in ptxas cuobjdump nvdisasm; do
  [[ -x "${CUDA_BIN}/${tool}" ]] || {
    printf '[thor] missing CUDA tool: %s/%s\n' "${CUDA_BIN}" "${tool}" >&2
    exit 1
  }
done
export TRITON_PTXAS_PATH="${CUDA_BIN}/ptxas"
export TRITON_PTXAS_BLACKWELL_PATH="${CUDA_BIN}/ptxas"
export TRITON_CUOBJDUMP_PATH="${CUDA_BIN}/cuobjdump"
export TRITON_NVDISASM_PATH="${CUDA_BIN}/nvdisasm"

CACHE_ROOT="${OUT_DIR}/.cache"
mkdir -p "${CACHE_ROOT}/triton" "${CACHE_ROOT}/torchinductor" "${CACHE_ROOT}/torch" "${CACHE_ROOT}/xdg"
export H3_CACHE_ROOT="${CACHE_ROOT}"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${CACHE_ROOT}/torchinductor"
export TORCH_HOME="${CACHE_ROOT}/torch"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  candidates=(
      "/opt/venv/bin/python"
      "/home/nvidia/.unsloth/studio/unsloth_studio/bin/python"
      "python3"
  )
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    candidates=("${VIRTUAL_ENV}/bin/python" "${candidates[@]}")
  fi
  for candidate in "${candidates[@]}"; do
    if [[ "${candidate}" == */* && -x "${candidate}" ]] || command -v "${candidate}" >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_RESOLVED="$(command -v "${PYTHON_BIN}" 2>/dev/null || true)"
[[ -n "${PYTHON_RESOLVED}" && -x "${PYTHON_RESOLVED}" ]] || {
  printf '[thor] Python interpreter is not executable: %s\n' "${PYTHON_BIN}" >&2
  exit 1
}
# Keep a virtual-environment entrypoint intact.  On the Thor hosts the Unsloth
# `bin/python` is a symlink into uv's interpreter store; resolving it here
# drops the venv site-packages when `uv --python` builds the run environment.
# `command -v` already gives an absolute path for PATH-resolved candidates,
# while an explicitly supplied path must retain its original venv wrapper.
PYTHON_BIN="${PYTHON_RESOLVED}"
if [[ -z "${UV_BIN:-}" ]]; then
  uv_candidates=(
      "/home/nvidia/.local/bin/uv"
      "/root/.local/bin/uv"
      "uv"
  )
  for candidate in "${uv_candidates[@]}"; do
    if [[ "${candidate}" == */* && -x "${candidate}" ]] || command -v "${candidate}" >/dev/null 2>&1; then
      UV_BIN="${candidate}"
      break
    fi
  done
fi
UV_BIN="${UV_BIN:-uv}"
command -v "${UV_BIN}" >/dev/null 2>&1 || {
  printf '[thor] uv is required; set UV_BIN to its absolute path\n' >&2
  exit 1
}

"${UV_BIN}" run --no-project --python "${PYTHON_BIN}" python - <<'PY'
import os
import sys
import ast
import inspect
import textwrap
from pathlib import Path

import diffusers
import torch

if not torch.cuda.is_available():
    raise SystemExit("[thor] CUDA is not available")
capability = tuple(torch.cuda.get_device_capability(0))
cuda_version = torch.version.cuda or ""
if capability != (11, 0):
    raise SystemExit(f"[thor] expected SM110, got SM{capability[0]}{capability[1]}")
if not cuda_version.startswith("13."):
    raise SystemExit(f"[thor] expected CUDA 13.x torch build, got {cuda_version!r}")
print(
    f"[thor] python={sys.executable} torch={torch.__version__} "
    f"cuda={cuda_version} gpu={torch.cuda.get_device_name(0)} "
    f"capability={capability}",
    flush=True,
)
from diffusers.models.transformers import transformer_minimax_h3  # noqa: F401
from diffusers import MiniMaxH3Blocks  # noqa: F401
from diffusers.modular_pipelines.minimax_h3.decoders import (
    MiniMaxH3AudioDecodeStep,
    MiniMaxH3VideoDecodeStep,
)

from hashlib import sha256
import json

diffusers_path = Path(diffusers.__file__).resolve()
expected_diffusers_src = os.environ.get("H3_DIFFUSERS_SRC")
if expected_diffusers_src:
    expected_package = (Path(expected_diffusers_src).resolve() / "diffusers")
    if not diffusers_path.is_relative_to(expected_package):
        raise SystemExit(
            f"[thor] Diffusers resolved outside H3_DIFFUSERS_SRC: "
            f"{diffusers_path} not under {expected_package}"
        )
expected_diffusers_tree = os.environ["H3_DIFFUSERS_TREE_SHA256"]
diffusers_package = diffusers_path.parent
tree_manifest = []
for source_path in sorted(diffusers_package.rglob("*.py")):
    relative = source_path.relative_to(diffusers_package).as_posix()
    tree_manifest.append(f"{sha256(source_path.read_bytes()).hexdigest()}  {relative}\n")
actual_diffusers_tree = sha256("".join(tree_manifest).encode()).hexdigest()
if actual_diffusers_tree != expected_diffusers_tree:
    raise SystemExit(
        f"[thor] Diffusers source digest mismatch: {actual_diffusers_tree} "
        f"!= {expected_diffusers_tree}"
    )
def has_latent_output_branch(decoder) -> bool:
    """Check decoder source semantics without assuming the local state variable name."""
    source = inspect.getsource(decoder.__call__)
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return False

    def is_output_type(node) -> bool:
        return (isinstance(node, ast.Name) and node.id == "output_type") or (
            isinstance(node, ast.Attribute) and node.attr == "output_type"
        )

    def is_latent(node) -> bool:
        return isinstance(node, ast.Constant) and node.value == "latent"

    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        for left, operator, right in zip(operands, node.ops, operands[1:]):
            if isinstance(operator, ast.Eq) and (
                (is_output_type(left) and is_latent(right))
                or (is_latent(left) and is_output_type(right))
            ):
                return True
    return False


def decoder_autocast_dtypes(decoder) -> set[str]:
    """Return CUDA autocast dtypes used directly by one pinned decoder step."""
    source = inspect.getsource(decoder.__call__)
    tree = ast.parse(textwrap.dedent(source))
    dtypes = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        for item in node.items:
            call = item.context_expr
            if not isinstance(call, ast.Call):
                continue
            name = call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", "")
            if name != "autocast":
                continue
            dtype = next((keyword.value for keyword in call.keywords if keyword.arg == "dtype"), None)
            if isinstance(dtype, ast.Attribute) and isinstance(dtype.value, ast.Name):
                dtypes.add(f"{dtype.value.id}.{dtype.attr}")
            elif isinstance(dtype, ast.Name):
                dtypes.add(dtype.id)
            else:
                dtypes.add(ast.dump(dtype) if dtype is not None else "default")
    return dtypes


for decoder in (MiniMaxH3VideoDecodeStep, MiniMaxH3AudioDecodeStep):
    if not has_latent_output_branch(decoder):
        raise SystemExit(f"[thor] Diffusers decoder lacks latent output: {decoder.__name__}")
video_autocast = decoder_autocast_dtypes(MiniMaxH3VideoDecodeStep)
audio_autocast = decoder_autocast_dtypes(MiniMaxH3AudioDecodeStep)
if video_autocast != {"torch.float16"}:
    raise SystemExit(f"[thor] unexpected video VAE autocast contract: {sorted(video_autocast)}")
if audio_autocast:
    raise SystemExit(f"[thor] unexpected audio VAE autocast contract: {sorted(audio_autocast)}")
print(
    f"[thor] diffusers={getattr(diffusers, '__version__', None)} source={diffusers_path} "
    f"video_vae_autocast=torch.float16 audio_vae_autocast=disabled",
    flush=True,
)

model_root = Path(os.environ.get("H3_MODEL_ROOT", ""))
qwen_root = Path(os.environ.get("H3_QWEN_FP8", ""))
comfy_root = Path(os.environ.get("H3_COMFY_ROOT", ""))

# These are the files used by the t2va block set at the pinned revisions.  The
# model index also contains transformer_ref, but loading that partition would
# allocate a second 33B DiT and is deliberately excluded from the Thor path.
required = {
    "H3_MODEL_ROOT/modular_model_index.json": model_root / "modular_model_index.json",
    "H3_MODEL_ROOT/transformer/config.json": model_root / "transformer" / "config.json",
    "H3_MODEL_ROOT/vae/config.json": model_root / "vae" / "config.json",
    "H3_MODEL_ROOT/vae/diffusion_pytorch_model.safetensors.index.json": (
        model_root / "vae" / "diffusion_pytorch_model.safetensors.index.json"
    ),
    "H3_MODEL_ROOT/audio_vae/config.json": model_root / "audio_vae" / "config.json",
    "H3_MODEL_ROOT/audio_vae/diffusion_pytorch_model.safetensors": (
        model_root / "audio_vae" / "diffusion_pytorch_model.safetensors"
    ),
    "H3_MODEL_ROOT/scheduler/scheduler_config.json": model_root / "scheduler" / "scheduler_config.json",
    "H3_MODEL_ROOT/audio_scheduler/scheduler_config.json": (
        model_root / "audio_scheduler" / "scheduler_config.json"
    ),
    "H3_QWEN_FP8/config.json": qwen_root / "config.json",
    "H3_QWEN_FP8/model.safetensors.index.json": qwen_root / "model.safetensors.index.json",
    "H3_COMFY_ROOT/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors": (
        comfy_root / "diffusion_models" / "minimax_h3_fl2va_pruned_fp8_scaled.safetensors"
    ),
}

# Tokenizer/processor files are not listed individually in the component index,
# but Qwen3VLProcessor reads all of them at construction time.
for relative in (
    "processor/chat_template.json",
    "processor/merges.txt",
    "processor/preprocessor_config.json",
    "processor/tokenizer_config.json",
    "processor/tokenizer.json",
    "processor/video_preprocessor_config.json",
    "processor/vocab.json",
    "tokenizer/merges.txt",
    "tokenizer/tokenizer_config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/vocab.json",
):
    required[f"H3_MODEL_ROOT/{relative}"] = model_root / relative

missing = [name for name, path in required.items() if not path.is_file()]
if missing:
    raise SystemExit("[thor] missing H3 assets: " + ", ".join(missing))

metadata_expected = {
    "H3_MODEL_ROOT/modular_model_index.json": (2935, "a2b6a210e482ffb78e613b553f570c44e101afce6741bd4ed91429d0559af031"),
    "H3_MODEL_ROOT/transformer/config.json": (546, "74c11bff524336576096993cbfcdcdc2ef4fa2fa4409df693bdcbc6c666282ae"),
    "H3_MODEL_ROOT/vae/config.json": (2011, "78f67deec3d63aae807f2bfe7154bc1e26f6372cb20b63265fcbae1b62bb5745"),
    "H3_MODEL_ROOT/vae/diffusion_pytorch_model.safetensors.index.json": (74228, "15f6d44553c3c616b0dc999920aa784f92ecee7e4201f1f99ac405cfbf3061ca"),
    "H3_MODEL_ROOT/audio_vae/config.json": (2271, "9a3c645ff892b376c6f5f4c8685964cd75474731af594ff058492a0000caabb6"),
    "H3_MODEL_ROOT/scheduler/scheduler_config.json": (97, "8fa6c3aa70dc9e691e1a6df899fd1b6f75f70481a27cee6e18a303817075c304"),
    "H3_MODEL_ROOT/audio_scheduler/scheduler_config.json": (96, "804780f7133477067bd6bbfbc02dc8b3cf9feeb400f97c08f5b1d5f6cbab3840"),
    "H3_QWEN_FP8/config.json": (11929, "2329a2c9022d94906d5af9bafe93ca97cba0eedb5426fc42ca64f2bfcd65e949"),
    "H3_QWEN_FP8/model.safetensors.index.json": (145407, "b1aef15fda23adc0cec1d4b699d5a9dfb6f68b59cca744392d447a730b9e64f4"),
    "H3_MODEL_ROOT/processor/chat_template.json": (5499, "5c72a170d2a4a1a3bc5adad2e689ae28138a9700e5b8c96c0266331e86c0acce"),
    "H3_MODEL_ROOT/processor/merges.txt": (1671839, "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"),
    "H3_MODEL_ROOT/processor/preprocessor_config.json": (390, "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"),
    "H3_MODEL_ROOT/processor/tokenizer_config.json": (11003, "a07e942ac874baa13758de8d1fbdb186683cc03416b5589e1b6671c6b3057c68"),
    "H3_MODEL_ROOT/processor/tokenizer.json": (7032403, "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7"),
    "H3_MODEL_ROOT/processor/video_preprocessor_config.json": (385, "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"),
    "H3_MODEL_ROOT/processor/vocab.json": (2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
    "H3_MODEL_ROOT/tokenizer/merges.txt": (1671839, "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3"),
    "H3_MODEL_ROOT/tokenizer/tokenizer_config.json": (11003, "a07e942ac874baa13758de8d1fbdb186683cc03416b5589e1b6671c6b3057c68"),
    "H3_MODEL_ROOT/tokenizer/tokenizer.json": (7032403, "a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7"),
    "H3_MODEL_ROOT/tokenizer/vocab.json": (2776833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
}
verified_files = []
for name, (expected_size, expected_sha) in metadata_expected.items():
    path = required[name]
    if path.stat().st_size != expected_size:
        raise SystemExit(
            f"[thor] pinned metadata size mismatch: {path} "
            f"{path.stat().st_size} != {expected_size}"
        )
    actual_sha = sha256(path.read_bytes()).hexdigest()
    if actual_sha != expected_sha:
        raise SystemExit(
            f"[thor] pinned metadata SHA-256 mismatch: {path} {actual_sha} != {expected_sha}"
        )
    verified_files.append(
        {"path": name, "size_bytes": expected_size, "sha256": actual_sha}
    )

try:
    component_index = json.loads((model_root / "modular_model_index.json").read_text())
except (OSError, TypeError, ValueError) as exc:
    raise SystemExit(f"[thor] invalid H3 component index: {exc}") from exc
expected_components = {
    spec.name for spec in MiniMaxH3Blocks().get_workflow("t2va").expected_components
}
if "transformer" not in expected_components:
    raise SystemExit(
        f"[thor] MiniMax-H3 block set lacks the FL2VA transformer: {sorted(expected_components)}"
    )
if "transformer_ref" in expected_components:
    raise SystemExit("[thor] t2va workflow unexpectedly includes transformer_ref")
indexed_components = set(component_index)
missing_components = expected_components - indexed_components - {"video_processor"}
if missing_components:
    raise SystemExit(
        "[thor] H3 component index lacks t2va components: "
        + ", ".join(sorted(missing_components))
    )

def indexed_shards(index_path: Path) -> list[str]:
    try:
        index = json.loads(index_path.read_text())
        names = sorted(set(index["weight_map"].values()))
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"[thor] invalid safetensors index {index_path}: {exc}") from exc
    if not names:
        raise SystemExit(f"[thor] empty safetensors index: {index_path}")
    return names

def check_files(
    root: Path,
    names: list[str],
    label: str,
    path_prefix: str,
    expected: dict[str, tuple[int, str]],
) -> None:
    if set(names) != set(expected):
        raise SystemExit(
            f"[thor] {label} shard set does not match the pinned revision: "
            f"actual={names}, expected={sorted(expected)}"
        )
    for name in names:
        path = root / name
        if not path.is_file():
            raise SystemExit(f"[thor] missing {label} shard: {path}")
        stat = path.stat()
        expected_size, expected_sha = expected.get(name, (0, ""))
        if expected_size and stat.st_size != expected_size:
            raise SystemExit(
                f"[thor] {label} shard size mismatch: {path} {stat.st_size} != {expected_size}"
            )
        # A sparse/incomplete download can report the target logical size while
        # occupying only a few megabytes on disk.  Refuse it before model load.
        if expected_size and stat.st_blocks * 512 < expected_size:
            raise SystemExit(f"[thor] {label} shard is sparse/incomplete: {path}")
        if os.environ.get("H3_PREFLIGHT_VERIFY_SHA256", "1") not in ("", "0", "false", "False"):
            if not expected_sha:
                raise SystemExit(f"[thor] no pinned SHA-256 for {label} shard: {path.name}")
            digest = sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                    digest.update(chunk)
            actual = digest.hexdigest()
            if actual != expected_sha:
                raise SystemExit(f"[thor] {label} shard SHA-256 mismatch: {path} {actual} != {expected_sha}")
        else:
            actual = None
        verified_files.append(
            {
                "path": f"{path_prefix}/{name}",
                "size_bytes": stat.st_size,
                "sha256": actual,
            }
        )

qwen_expected = {
    "model-00001-of-00007.safetensors": (5324789312, "6241a0b1084cdd18437b1fbf746035b0ccdbf2ec1d09bf44766de2b82bfa4999"),
    "model-00002-of-00007.safetensors": (5365032312, "34e2fe44c670c5b76bd41e19fdcbe27b081ac1d6ded689b14022d3b65fd429fe"),
    "model-00003-of-00007.safetensors": (5365032312, "7c3ca8e912329fa8c1d640fea91b81799a0555c51192dcb910a8af915b783852"),
    "model-00004-of-00007.safetensors": (5365032312, "e20133f4dd94d191d16799e7d8d3808e8d6dfcfe39e5d596ba8e2ec7453673fb"),
    "model-00005-of-00007.safetensors": (5365032312, "f9012aafe848e03d7f54130eb72b768fe9366f557d39b667bc08d601bb90aff5"),
    "model-00006-of-00007.safetensors": (5365032312, "5e5eeb6324f79c91226a7b12cccc5a0635b5402676b35121727d402a8d17b118"),
    "model-00007-of-00007.safetensors": (3367015312, "82e3a1183df4b8deece6141ebac7eb560d0b1279f83f660359d3d4389fb90a87"),
}
vae_expected = {
    "diffusion_pytorch_model-00001-of-00003.safetensors": (5061033024, "72f4c6be84ac0674f27398cde991dd9d719762f3952c4921aa66b2ce542f6374"),
    "diffusion_pytorch_model-00002-of-00003.safetensors": (4955986528, "2e05e8bc23fa4071043e17fd242be8acd0685e781a43987432b2eae925be4198"),
    "diffusion_pytorch_model-00003-of-00003.safetensors": (398539336, "c05d6ac4b1a33de372799d708531da6320f6a3ce6d1ce6d895e770988e004a39"),
}
check_files(
    qwen_root,
    indexed_shards(qwen_root / "model.safetensors.index.json"),
    "Qwen",
    "H3_QWEN_FP8",
    qwen_expected,
)
check_files(
    model_root / "vae",
    indexed_shards(model_root / "vae" / "diffusion_pytorch_model.safetensors.index.json"),
    "video VAE",
    "H3_MODEL_ROOT/vae",
    vae_expected,
)
check_files(model_root / "audio_vae", ["diffusion_pytorch_model.safetensors"], "audio VAE", "H3_MODEL_ROOT/audio_vae", {
    "diffusion_pytorch_model.safetensors": (605429340, "52c59e67ba8de5477c81bfbced0327aabf500f1bfdeefd5ee754529241cb26cb"),
})
check_files(comfy_root / "diffusion_models", ["minimax_h3_fl2va_pruned_fp8_scaled.safetensors"], "DiT", "H3_COMFY_ROOT/diffusion_models", {
    "minimax_h3_fl2va_pruned_fp8_scaled.safetensors": (20958205608, "12944c1f7791637e7de12208aef04da82bd26b95271b1b47d817364315ade993"),
})

for root in (model_root, qwen_root, comfy_root):
    leftovers = sorted(
        path for path in root.rglob("*")
        if path.is_file() and (path.name.endswith(".incomplete") or path.name.endswith(".aria2"))
    )
    if leftovers:
        raise SystemExit("[thor] incomplete download artifacts remain: " + ", ".join(map(str, leftovers[:8])))

quarantined = sorted(
    path for path in model_root.rglob("*.hf-stalled") if path.is_file()
)
if quarantined:
    print(
        "[thor] ignored quarantined stalled downloads: "
        + ", ".join(map(str, quarantined[:8])),
        flush=True,
    )

asset_manifest = {
    "schema_version": 1,
    "models": {
        "minimax_h3": {
            "repo_id": "MiniMaxAI/MiniMax-H3",
            "revision": "bfc8ed0353f5a9733be73e6b2c98ec0948195b86",
        },
        "transformer": {
            "repo_id": "Comfy-Org/MiniMax-H3",
            "revision": "4cc1d817b6184899b41293954329f576cb5ae86b",
        },
        "text_encoder": {
            "repo_id": "Qwen/Qwen3-VL-32B-Instruct-FP8",
            "revision": "4bf2c2f39c37c0fede78bede4056e1f18cdf8109",
        },
    },
    "files": sorted(verified_files, key=lambda item: item["path"]),
}
canonical = json.dumps(asset_manifest, sort_keys=True, separators=(",", ":")).encode()
asset_manifest["payload_sha256"] = sha256(canonical).hexdigest()
asset_manifest_path = Path(os.environ["OUT_DIR"]) / "asset-manifest.json"
asset_manifest_path.write_text(
    json.dumps(asset_manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.chmod(asset_manifest_path, 0o644)

print(
    f"[thor] assets complete: qwen={len(indexed_shards(qwen_root / 'model.safetensors.index.json'))} "
    f"video_vae={len(indexed_shards(model_root / 'vae' / 'diffusion_pytorch_model.safetensors.index.json'))} "
    f"sha256={os.environ.get('H3_PREFLIGHT_VERIFY_SHA256', '1')} "
    f"diffusers_commit={os.environ['H3_DIFFUSERS_COMMIT']} "
    f"diffusers_tree={actual_diffusers_tree}",
    flush=True,
)
PY

ASSET_MANIFEST="${OUT_DIR}/asset-manifest.json"
[[ -s "${ASSET_MANIFEST}" ]] || {
  printf '[thor] asset manifest was not produced: %s\n' "${ASSET_MANIFEST}" >&2
  exit 1
}
export H3_ASSET_MANIFEST_PATH="${ASSET_MANIFEST}"
export H3_ASSET_MANIFEST_SHA256="$(sha256sum "${ASSET_MANIFEST}" | awk '{print $1}')"

if [[ "${H3_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  printf '[thor] preflight passed\n'
  exit 0
fi

exec > >(tee "${OUT_DIR}/run.log") 2>&1
export H3_VIDEO_VAE_AUTOCAST_DTYPE=torch.float16
export H3_AUDIO_VAE_AUTOCAST_DTYPE=disabled
export H3_THOR_WRAPPER_ACTIVE=1
export H3_THOR_WRAPPER_PID="$$"
H3_THOR_WRAPPER_TOKEN="$(od -An -N32 -tx1 /dev/urandom | tr -d '[:space:]')"
[[ "${H3_THOR_WRAPPER_TOKEN}" =~ ^[0-9a-f]{64}$ ]] || {
  printf '[thor] could not create wrapper context token\n' >&2
  exit 1
}
export H3_THOR_WRAPPER_TOKEN
export H3_THOR_WRAPPER_RECEIPT="${OUT_DIR}/.thor-wrapper-context-${H3_THOR_WRAPPER_PID}"
(umask 077; printf '%s\n' "${H3_THOR_WRAPPER_TOKEN}" > "${H3_THOR_WRAPPER_RECEIPT}")
INFER_ARGS=("$@")
if [[ "${H3_OFFICIAL}" == "1" ]]; then
  INFER_ARGS+=(--official)
fi
exec "${UV_BIN}" run --no-project --python "${PYTHON_BIN}" python -u \
  "${HERE}/gpu_infer.py" "${INFER_ARGS[@]}"
