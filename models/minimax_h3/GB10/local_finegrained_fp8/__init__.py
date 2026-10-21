"""Local fine-grained FP8 kernels with lazy public imports.

The Transformers 5.5/5.8 compatibility path imports ``legacy`` submodules. Python
executes this parent package first, so eager imports here would unnecessarily bind
the newer kernel bundle and its Triton ABI before the legacy code is selected.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "fp8_act_quant": (".utils", "fp8_act_quant"),
    "matmul_2d": (".matmul", "matmul_2d"),
    "matmul_batched": (".batched", "matmul_batched"),
    "matmul_grouped": (".grouped", "matmul_grouped"),
    "moe_fused_batched": (".fused_batched", "moe_fused_batched"),
    "moe_fused_grouped": (".fused_grouped", "moe_fused_grouped"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value

__all__ = [
    "fp8_act_quant",
    # 2D matmul
    "matmul_2d",
    # Batched matmul
    "matmul_batched",
    # Grouped matmul
    "matmul_grouped",
    # Fused batched matmul
    "moe_fused_batched",
    # Fused grouped matmul
    "moe_fused_grouped",
]
