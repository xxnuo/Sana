from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

GB10_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GB10_ROOT))

from local_finegrained_fp8 import bayesian_autotuner


class _FakeKernel:
    def __init__(self) -> None:
        self.fn = lambda value: value
        self.calls = 0

    def run(self, *args, **kwargs):
        self.calls += 1
        return args, kwargs


def _make_tuner(*, cache_results: bool = True, n_trials: int = 1):
    configs = [
        bayesian_autotuner.Config({"BLOCK": 32}, num_warps=4, num_stages=2),
        bayesian_autotuner.Config({"BLOCK": 64}, num_warps=8, num_stages=3),
    ]
    tuner = bayesian_autotuner.BayesianAutotuner(
        _FakeKernel(),
        ["size"],
        configs,
        ["size"],
        None,
        None,
        n_trials=n_trials,
        cache_results=cache_results,
        do_bench=lambda *_args, **_kwargs: (1.0, 1.0, 1.0),
    )
    return tuner, configs


def test_triton_33_constructor_accepts_cache_results() -> None:
    tuner, _ = _make_tuner(cache_results=True)

    assert tuner.cache_results is True


def test_runtime_fingerprint_records_sm110_identity(
    monkeypatch,
) -> None:
    properties = SimpleNamespace(name="NVIDIA Thor", multi_processor_count=14)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _index: properties)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: (11, 0))
    monkeypatch.setattr(
        bayesian_autotuner,
        "_nvidia_driver_identity",
        lambda: "NVIDIA UNIX Open Kernel Module 580.00",
    )
    monkeypatch.setattr(
        bayesian_autotuner,
        "_ptxas_identity",
        lambda arch: {
            "path": "/opt/cuda/bin/ptxas",
            "release": "13.2",
            "version": str(arch),
        },
    )
    target = SimpleNamespace(backend="cuda", arch=110, warp_size=32)
    monkeypatch.setattr(
        bayesian_autotuner,
        "driver",
        SimpleNamespace(active=SimpleNamespace(get_current_target=lambda: target)),
    )
    bayesian_autotuner._runtime_fingerprint.cache_clear()

    identity = json.loads(bayesian_autotuner._runtime_fingerprint(0))

    assert identity["gpu"] == {
        "name": "NVIDIA Thor",
        "compute_capability": "sm_110",
        "sm_count": 14,
    }
    assert identity["ptxas"]["release"] == "13.2"
    assert identity["torch"] == torch.__version__
    assert identity["triton"]


def test_disk_cache_key_changes_with_runtime_identity() -> None:
    common = {
        "triton_identity": "triton-key",
        "backend_hash": "backend",
        "function_hash": "function",
        "env_vars": {"TRITON_FOO": "1"},
        "tuning_key": (1024, 4096),
        "configs": ["config-a"],
    }

    thor_key = bayesian_autotuner._disk_cache_key(
        runtime_fingerprint='{"gpu":{"compute_capability":"sm_110","sm_count":14}}',
        **common,
    )
    other_key = bayesian_autotuner._disk_cache_key(
        runtime_fingerprint='{"gpu":{"compute_capability":"sm_110","sm_count":18}}',
        **common,
    )

    assert thor_key != other_key


def test_exhaustive_search_cache_is_scoped_by_runtime(
    monkeypatch,
) -> None:
    tuner, configs = _make_tuner(cache_results=False, n_trials=100)
    runtime = {"value": "thor-t5000-a"}
    seen_keys = []
    monkeypatch.setattr(
        bayesian_autotuner,
        "_runtime_fingerprint_for_args",
        lambda _arguments: runtime["value"],
    )

    def select_first(_configs, _args, _kwargs, key):
        seen_keys.append(key)
        tuner.configs_timings = {configs[0]: 1.0}
        return configs[0]

    monkeypatch.setattr(tuner, "_bayesian_search", select_first)

    tuner.run(1024)
    runtime["value"] = "thor-t5000-b"
    tuner.run(1024)

    assert len(tuner.cache) == 2
    assert len(seen_keys) == 2
    assert seen_keys[0][-1] == (
        bayesian_autotuner._RUNTIME_KEY,
        "thor-t5000-a",
    )
    assert seen_keys[1][-1] == (
        bayesian_autotuner._RUNTIME_KEY,
        "thor-t5000-b",
    )


def test_single_trial_budget_benchmarks_one_config(monkeypatch) -> None:
    tuner, configs = _make_tuner(cache_results=False, n_trials=1)
    benched = []

    def bench(*_args, config, **_kwargs):
        benched.append(config)
        return 1.0

    monkeypatch.setattr(tuner, "_bench", bench)

    best = tuner._bayesian_search(configs, (1024,), {}, ())

    assert best in configs
    assert len(benched) == 1
