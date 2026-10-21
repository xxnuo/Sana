from __future__ import annotations

import pytest

from sol_attn.interface import _backend_for_arch


@pytest.mark.parametrize(
    ("arch", "expected"),
    [
        ((8, 9), "cute_sm89"),
        ((9, 0), "cute_sm90"),
        ((10, 0), "cute_sm100"),
        ((12, 0), "cute_sm120"),
    ],
)
def test_registered_cuda_architectures_select_cute(arch, expected):
    assert _backend_for_arch(arch, cute_available=True) == expected


@pytest.mark.parametrize("arch", [(8, 0), (11, 0), (12, 1)])
def test_unregistered_cuda_architectures_fall_back_to_triton(arch):
    assert _backend_for_arch(arch, cute_available=True) == "triton"


def test_registered_cuda_architecture_falls_back_without_cute_runtime():
    assert _backend_for_arch((10, 0), cute_available=False) == "triton"


def test_cuda_architecture_older_than_ampere_is_rejected():
    with pytest.raises(RuntimeError, match="compute capability >= 8.0"):
        _backend_for_arch((7, 5), cute_available=True)
