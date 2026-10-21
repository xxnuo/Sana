#!/usr/bin/env python3
"""Thor entrypoint for the single-card pruned-FP8 MiniMax-H3 runtime.

The model assembly is shared with the GB10 profile because both are single-card
CUDA 13 systems using the same pruned FP8 checkpoint layout.  This shim is an
internal target of ``run_minimax_h3_gpu.sh``; direct invocation is rejected so
the CUDA toolchain, UMA, asset, and source-integrity gates cannot be bypassed.
"""

from __future__ import annotations

import os
import runpy
import stat
from pathlib import Path


def _read_process_values(path: Path) -> dict[str, str]:
    values = {}
    for item in path.read_bytes().split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        values[key.decode(errors="replace")] = value.decode(errors="replace")
    return values


def _validate_wrapper_context() -> None:
    """Reject accidental direct invocation before importing CUDA or model code."""
    try:
        wrapper_pid = int(os.environ["H3_THOR_WRAPPER_PID"])
        token = os.environ["H3_THOR_WRAPPER_TOKEN"]
        receipt = Path(os.environ["H3_THOR_WRAPPER_RECEIPT"]).resolve()
        output_dir = Path(os.environ["H3_OUTPUT_DIR"]).resolve()
        parent_pid = os.getppid()
        parent_env = _read_process_values(Path(f"/proc/{parent_pid}/environ"))
        parent_argv = Path(f"/proc/{parent_pid}/cmdline").read_bytes().split(b"\0")
        receipt_stat = receipt.stat()
        receipt_token = receipt.read_text(encoding="ascii").strip()
    except (KeyError, OSError, UnicodeError, ValueError):
        raise SystemExit(
            "[thor] direct gpu_infer.py invocation is unsupported; "
            "run models/minimax_h3/Thor/run_minimax_h3_gpu.sh"
        ) from None

    valid = (
        os.environ.get("H3_THOR_WRAPPER_ACTIVE") == "1"
        and wrapper_pid == parent_pid
        and bool(token)
        and parent_env.get("H3_THOR_WRAPPER_ACTIVE") == "1"
        and parent_env.get("H3_THOR_WRAPPER_PID") == str(wrapper_pid)
        and parent_env.get("H3_THOR_WRAPPER_TOKEN") == token
        and parent_env.get("H3_THOR_WRAPPER_RECEIPT") == str(receipt)
        and bool(parent_argv)
        and Path(parent_argv[0].decode(errors="replace")).name == "uv"
        and receipt.parent == output_dir
        and receipt_stat.st_uid == os.getuid()
        and stat.S_IMODE(receipt_stat.st_mode) == 0o600
        and receipt_token == token
    )
    if not valid:
        raise SystemExit(
            "[thor] direct gpu_infer.py invocation is unsupported; "
            "run models/minimax_h3/Thor/run_minimax_h3_gpu.sh"
        )
    receipt.unlink()


_validate_wrapper_context()

if os.environ.get("H3_THOR_WRAPPER_ACTIVE") != "1":
    raise SystemExit(
        "[thor] direct gpu_infer.py invocation is unsupported; "
        "run models/minimax_h3/Thor/run_minimax_h3_gpu.sh"
    )

import torch


if not torch.cuda.is_available():
    raise SystemExit("[thor] CUDA is not available")
capability = tuple(torch.cuda.get_device_capability(0))
if capability != (11, 0):
    raise SystemExit(f"[thor] expected SM110, got SM{capability[0]}{capability[1]}")
cuda_version = torch.version.cuda or ""
if not cuda_version.startswith("13."):
    raise SystemExit(f"[thor] expected CUDA 13.x torch build, got {cuda_version!r}")


# Keep direct invocation and the strict shell launcher on the same profile identity.  The shared
# GB10 assembler reads these values when it writes benchmark metadata.
os.environ["H3_TAG"] = "thor_sm110"
os.environ["H3_VARIANT"] = "diffusers_modular_pruned_fp8_scaled_single_thor_sm110"
os.environ["H3_REQUIRE_SM110"] = "1"

_GB10_ENTRYPOINT = Path(__file__).resolve().parent.parent / "GB10" / "gpu_infer.py"
runpy.run_path(str(_GB10_ENTRYPOINT), run_name="__main__")
