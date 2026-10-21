# MiniMax-H3 on Jetson Thor

This profile targets one NVIDIA Jetson Thor T5000 (SM110, 128 GiB unified
memory) with the MiniMax-H3 FL2VA pipeline. It reuses the single-card GB10
assembler, but the Thor wrapper adds SM110, CUDA 13, JetPack toolchain, UMA
headroom, and source/asset integrity gates before importing the model.

The runtime is pinned to the Sana `sol-engine` checkout recorded in
[`SOURCE_SNAPSHOT.json`](SOURCE_SNAPSHOT.json). This is a bring-up and
benchmark profile; it is independent of the other Sana hardware targets.

## Pinned inputs

The launcher validates the following inputs and records their identities in
`outputs/asset-manifest.json` and `outputs/benchmark.json`:

- MiniMax-H3: `MiniMaxAI/MiniMax-H3`, revision
  `bfc8ed0353f5a9733be73e6b2c98ec0948195b86`.
- Pruned FP8 DiT: `Comfy-Org/MiniMax-H3`, revision
  `4cc1d817b6184899b41293954329f576cb5ae86b`, file
  `diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors`.
- Qwen conditioner: `Qwen/Qwen3-VL-32B-Instruct-FP8`, revision
  `4bf2c2f39c37c0fede78bede4056e1f18cdf8109`.
- Diffusers source: commit
  `abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc` (Python tree digest is pinned in
  `SOURCE_SNAPSHOT.json`).

By default the wrapper discovers the three model roots under
`/home/nvidia/Downloads/` using the names shown below. Set the variables
explicitly when the assets live elsewhere:

```bash
export H3_MODEL_ROOT=/path/to/MiniMaxAI--MiniMax-H3
export H3_QWEN_FP8=/path/to/Qwen--Qwen3-VL-32B-Instruct-FP8
export H3_COMFY_ROOT=/path/to/Comfy-Org--MiniMax-H3
export H3_DIFFUSERS_SRC=/path/to/diffusers-abc5e9bf/src
```

The DiT file is fixed to the path below and its SHA-256 is checked at launch:

```text
$H3_COMFY_ROOT/diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors
12944c1f7791637e7de12208aef04da82bd26b95271b1b47d817364315ade993
```

## Host prerequisites

- Jetson Thor T5000 with `torch.cuda.get_device_capability(0) == (11, 0)`.
- CUDA 13.x PyTorch build and an executable `uv`.
- JetPack `/usr/local/cuda/bin/ptxas`, `cuobjdump`, and `nvdisasm`.
- At least 80 GiB `MemAvailable` for the launch preflight. Runtime monitoring
  also requires 16 GiB remaining UMA headroom (`H3_BENCH_RUNTIME_MIN_AVAILABLE_MIB`).
- Offline access to all pinned assets; the wrapper sets `HF_HUB_OFFLINE=1`.

Use the wrapper through `launch_config.py`; direct execution of `gpu_infer.py`
is rejected. From the repository root on the Thor host:

```bash
uv run --no-project \
  --python /home/nvidia/.unsloth/studio/unsloth_studio/bin/python python \
  thirdparty/Sana/scripts/launch_config.py \
  thirdparty/Sana/config/minimax_h3/thor_fullopt.toml \
  --mode local --strict-commit \
  --run-root runs/thor-manual
```

For a preparation-only check, replace `--mode local` with `--mode dry-run`.
The wrapper's asset/import preflight can be run separately with an explicit
temporary output directory:

```bash
preflight_dir="$(mktemp -d -p /tmp h3-thor-preflight.XXXXXX)"
OUT_DIR="$preflight_dir" H3_PREFLIGHT_ONLY=1 \
  bash thirdparty/Sana/models/minimax_h3/Thor/run_minimax_h3_gpu.sh
rm -rf -- "$preflight_dir"
```

## Profiles

All profiles use the smoke shape by default (448x256, 124 frames, 8 steps,
seed 0). The matching manifests are in
[`config/minimax_h3`](../../../config/minimax_h3/):

| Manifest | Scope | Important switches |
|---|---|---|
| `thor_baseline.toml` | Latent control | No fused kernels, latent output |
| `thor_kernel.toml` | Lossless latent arm | Triton fusions only |
| `thor_baseline_video.toml` | Video/audio control | No fusions or VAE tile batching |
| `thor_fullopt.toml` | Delivery candidate | Triton fusions plus VAE tile batch 4 |
| `thor_kernel_sol.toml` | Experimental | 8-step SM110 Sol-Attn A/B only |

`thor_fullopt` deliberately leaves Sol-Attn and cache disabled until a new
isolated quality/latency result justifies enabling them. Its VAE contract is
FP32 weights, FP16 video autocast, and disabled audio autocast.

## Kernel gates

Run the standalone SM110 Sol-Attn and fusion compile/correctness gates on one
or more Thor hosts (`gpu2` and `gpu5` are the available hosts):

```bash
bash scripts/run_sol_engine_h3_thor.sh --host gpu2 --host gpu5
```

This gate checks Triton architecture/PTX artifacts and numerical contracts; it
does not run MiniMax-H3 end to end.

## Reproducible benchmark

The benchmark driver enforces a single-flight idle gate, bounded runtime
monitoring, process-group cleanup, UMA sampling, and provenance capture. It
supports these workload suites:

- `smoke`: 448x256, 124 frames, 8 steps (default; Sol-Attn candidate allowed).
- `standard`: 832x480, 124 frames, 50 steps.
- `official`: 1344x768, 124 frames, 50 steps and `H3_OFFICIAL=1`.

Run the complete smoke matrix on an idle Thor host:

```bash
bash scripts/benchmark_sol_engine_h3_thor.sh \
  --suite smoke --phase all \
  --run-root runs/thor-smoke-$(date -u +%Y%m%d-%H%M%S)
```

For a standard or official end-to-end comparison (three measured requests):

```bash
bash scripts/benchmark_sol_engine_h3_thor.sh \
  --suite standard --phase e2e --e2e-samples 3 \
  --run-root runs/thor-standard-$(date -u +%Y%m%d-%H%M%S)

bash scripts/benchmark_sol_engine_h3_thor.sh \
  --suite official --phase e2e --e2e-samples 3 \
  --run-root runs/thor-official-$(date -u +%Y%m%d-%H%M%S)
```

The run root is relative to `thirdparty/Sana/`. Each generated run contains
`outputs/benchmark.json`, launch and driver logs, host snapshots, runtime
monitor samples, `asset-manifest.json`, and the Triton artifact report. Latent
runs additionally produce `latents.safetensors` and `quality.json`; the suite
root contains `summary-latent.{json,md}`, `summary-e2e.{json,md}`, and (for the
complete matrix) `e2e-validation-scope.json`.

## Promotion boundary

Promotion requires a current isolated run with passing provenance, SM110
artifact, UMA/runtime monitor, exact latent, and exact VAE gates. Complete
request latency by itself is performance evidence only; decoded video/audio
quality is promoted only when `e2e-validation-scope.json` reports
`quality_and_performance`.

The current production candidate is `thor_fullopt` (lossless Triton fusions
and VAE tile batching). `thor_kernel_sol` is an experimental 8-step policy and
must not be promoted from a speedup or kernel compile pass alone. It is only
eligible for the smoke A/B and requires latent quality thresholds against
`thor_kernel`; standard and official suites intentionally skip it.
