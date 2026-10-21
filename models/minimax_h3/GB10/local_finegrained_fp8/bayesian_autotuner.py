# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bayesian-optimization autotuner: benches a budgeted sample of the config grid instead of
the full grid, via a Tree-structured Parzen Estimator (TPE) over the config *dimensions*
(``num_warps`` / ``num_stages`` / tile sizes / flags). After a short random seed phase it
models the good (top-``gamma`` by measured time) vs bad per-dimension value densities and
benches the unmeasured config that maximizes ``l(x)/g(x)`` — the Expected-Improvement proxy
— then a coordinate-descent pass polishes the best. Subclasses Triton's ``Autotuner`` and
uses only the Python stdlib (no external optimizer dependency).

Each new key's search is warm-started from the most recently tuned key's best config on the same
GPU/runtime identity (nearby workloads share tile-shape preferences). Grids smaller than
``n_trials`` are exhaustively benchmarked while retaining the persistent cache.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import math
import os
import random
import subprocess
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import triton
from triton.compiler.compiler import get_cache_invalidating_env_vars
from triton.runtime.autotuner import Autotuner, Config
from triton.runtime.cache import get_cache_manager
from triton.runtime.driver import driver
from triton.runtime.jit import JITFunction

try:
    from triton.runtime.cache import triton_key
except ImportError:  # Triton 3.3
    from triton.compiler.compiler import triton_key


_triton_knobs = getattr(triton, "knobs", None)


_AUTOTUNER_SUPPORTS_CACHE_RESULTS = (
    "cache_results" in inspect.signature(Autotuner.__init__).parameters
)
_RUNTIME_KEY = "__sol_h3_runtime__"


def _autotune_print_enabled() -> bool:
    if _triton_knobs is not None:
        try:
            return bool(_triton_knobs.autotuning.print)
        except AttributeError:
            pass
    return os.environ.get("TRITON_PRINT_AUTOTUNING") == "1"


def _cuda_device_index(arguments: Iterable[object]) -> int | None:
    for value in arguments:
        device = getattr(value, "device", None)
        if getattr(device, "type", None) != "cuda":
            continue
        index = getattr(device, "index", None)
        if index is not None:
            return int(index)
        break
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.current_device())
    except (ImportError, RuntimeError):
        pass
    return None


@functools.lru_cache(maxsize=1)
def _nvidia_driver_identity() -> str | None:
    path = Path("/proc/driver/nvidia/version")
    try:
        return " ".join(path.read_text(encoding="utf-8").split())
    except OSError:
        return None


@functools.cache
def _ptxas_identity(arch: int) -> dict[str, str] | None:
    try:
        from triton.backends.nvidia.compiler import get_ptxas, get_ptxas_version

        tool = get_ptxas(arch)
        if hasattr(tool, "path"):
            path = tool.path
            release = getattr(tool, "version", None)
        else:
            path = tool[0]
            release = tool[1] if len(tool) > 1 else None
        version = " ".join(str(get_ptxas_version(arch)).split())
        return {
            "path": os.path.realpath(str(path)),
            "release": str(release or ""),
            "version": version,
        }
    except (ImportError, OSError, RuntimeError, subprocess.SubprocessError, TypeError):
        return None


def _build_runtime_fingerprint(device_index: int | None) -> dict[str, object]:
    fingerprint: dict[str, object] = {
        "schema": 1,
        "triton": str(getattr(triton, "__version__", "unknown")),
        "nvidia_driver": _nvidia_driver_identity(),
    }
    try:
        import torch
    except ImportError:
        fingerprint["torch"] = None
        return fingerprint

    fingerprint["torch"] = str(getattr(torch, "__version__", "unknown"))
    fingerprint["torch_cuda"] = getattr(getattr(torch, "version", None), "cuda", None)
    if device_index is None:
        return fingerprint

    try:
        properties = torch.cuda.get_device_properties(device_index)
        capability = tuple(torch.cuda.get_device_capability(device_index))
        arch = int(capability[0]) * 10 + int(capability[1])
        fingerprint["gpu"] = {
            "name": str(properties.name),
            "compute_capability": f"sm_{arch}",
            "sm_count": int(properties.multi_processor_count),
        }
        fingerprint["ptxas"] = _ptxas_identity(arch)
        try:
            target = driver.active.get_current_target()
            fingerprint["triton_target"] = {
                "backend": getattr(target, "backend", None),
                "arch": getattr(target, "arch", None),
                "warp_size": getattr(target, "warp_size", None),
            }
        except (AttributeError, RuntimeError):
            fingerprint["triton_target"] = None
    except (AttributeError, RuntimeError):
        fingerprint["gpu"] = None
    return fingerprint


@functools.cache
def _runtime_fingerprint(device_index: int | None) -> str:
    return json.dumps(
        _build_runtime_fingerprint(device_index),
        sort_keys=True,
        separators=(",", ":"),
    )


def _runtime_fingerprint_for_args(arguments: Iterable[object]) -> str:
    return _runtime_fingerprint(_cuda_device_index(arguments))


def _runtime_from_tuning_key(tuning_key) -> str | None:
    for item in reversed(tuning_key):
        if isinstance(item, tuple) and len(item) == 2 and item[0] == _RUNTIME_KEY:
            return item[1]
    return None


def _disk_cache_key(
    *,
    triton_identity: str,
    backend_hash: str,
    function_hash: str,
    env_vars,
    tuning_key,
    runtime_fingerprint: str,
    configs,
) -> str:
    parts = [
        triton_identity,
        backend_hash,
        function_hash,
        str(sorted(env_vars.items())),
        str(tuning_key),
        runtime_fingerprint,
    ] + [str(config) for config in configs]
    return hashlib.sha256("-".join(parts).encode("utf-8")).hexdigest()


class BayesianAutotuner(Autotuner):
    """Drop-in replacement for ``triton.runtime.autotuner.Autotuner`` that
    benches ~``n_trials`` configs per key via TPE Bayesian optimization +
    coordinate-descent refinement, instead of the full grid."""

    def __init__(
        self,
        *args,
        n_trials: int = 80,
        n_startup_trials: int = 12,
        gamma: float = 0.25,
        refine: bool = True,
        max_refine_iters: int = 5,
        log_path: str | None = None,
        cache_results: bool = True,
        **kwargs,
    ):
        parent_kwargs = dict(kwargs)
        if _AUTOTUNER_SUPPORTS_CACHE_RESULTS:
            parent_kwargs["cache_results"] = cache_results
        super().__init__(*args, **parent_kwargs)
        if not _AUTOTUNER_SUPPORTS_CACHE_RESULTS:
            self.cache_results = (
                bool(cache_results) and os.environ.get("TRITON_INTERPRET") != "1"
            )
        # Bayesian trial budget — the per-decorator default, overridable via the
        # FINEGRAINED_AUTOTUNE_TRIALS env var (quick sweeps / exhaustive runs without touching
        # the decorators; set it >= grid size to benchmark and cache the full grid).
        self.n_trials = max(
            1, int(os.environ.get("FINEGRAINED_AUTOTUNE_TRIALS") or n_trials)
        )
        self.n_startup_trials = n_startup_trials
        # top fraction of measured configs the TPE treats as "good"
        self.gamma = gamma
        self.refine = refine
        self.max_refine_iters = max_refine_iters
        # JSONL log of every benched (key, config, ms) — set here or via the
        # FINEGRAINED_AUTOTUNE_LOG env var. Analyse offline to prune bad configs.
        self.log_path = log_path or os.environ.get("FINEGRAINED_AUTOTUNE_LOG")

    def run(self, *args, **kwargs):
        # A single config needs no search. Multi-config exhaustive searches still go through
        # this implementation so their result gets the same runtime-scoped disk cache.
        if len(self.configs) <= 1:
            return super().run(*args, **kwargs)

        self.nargs = dict(zip(self.arg_names, args))
        all_args = {**self.nargs, **kwargs}
        _args = {k: v for k, v in all_args.items() if k in self.arg_names}
        key = [_args[k] for k in self.keys if k in _args]
        key.extend(str(v.dtype) for v in _args.values() if hasattr(v, "dtype"))
        key.append((_RUNTIME_KEY, _runtime_fingerprint_for_args(_args.values())))
        key = tuple(key)

        if key not in self.cache:
            pruned = self.prune_configs(kwargs)

            def benchmark():
                t0 = time.time()
                self.cache[key] = self._bayesian_search(pruned, args, kwargs, key)
                self.bench_time = time.time() - t0
                if _autotune_print_enabled():
                    fn_name = getattr(self.fn, "__name__", str(self.fn))
                    print(
                        f"[bayesian-autotune] {fn_name} tuned "
                        f"{len(self.configs_timings)} configs in {self.bench_time:.1f}s — "
                        f"key={key}, best={self.cache[key].all_kwargs()}"
                    )
                full_nargs = {**self.nargs, **kwargs, **self.cache[key].all_kwargs()}
                self.pre_hook(full_nargs, reset_only=True)

            if self.cache_results:
                self.check_disk_cache(key, pruned, benchmark)
            else:
                benchmark()

        config = self.cache[key]
        self.best_config = config
        if config.pre_hook is not None:
            config.pre_hook({**self.nargs, **kwargs, **config.all_kwargs()})
        ret = self.fn.run(*args, **kwargs, **config.all_kwargs())
        self.nargs = None
        return ret

    def _bayesian_search(self, configs: list[Config], args, kwargs, key) -> Config:
        timings: dict[int, float] = {}
        sigs = [tuple(sorted(c.all_kwargs().items())) for c in configs]

        def bench_idx(idx: int) -> float:
            if idx in timings:
                return timings[idx]
            try:
                ms = self._bench(*args, config=configs[idx], **kwargs)
                if isinstance(ms, (tuple, list)):
                    ms = ms[0]
                timings[idx] = float(ms)
            except Exception:
                timings[idx] = float("inf")
            self._log_result(key, configs[idx], timings[idx])
            return timings[idx]

        # Distinct values per config dimension, for the TPE's Laplace smoothing.
        dim_vals: dict = defaultdict(set)
        for sig in sigs:
            for d, v in sig:
                dim_vals[d].add(v)

        # Seed phase: a few random configs (seeded → deterministic), warm-started from the
        # most recent cached key's best, to give the TPE an initial good/bad split.
        n_startup = min(
            len(configs), self.n_trials, max(1, self.n_startup_trials)
        )
        order = list(range(len(configs)))
        random.Random(0).shuffle(order)
        warm_idx = self._warm_start_index(configs, _runtime_from_tuning_key(key))
        if warm_idx is not None:
            order = [warm_idx] + [i for i in order if i != warm_idx]
        for idx in order[:n_startup]:
            bench_idx(idx)

        # TPE: split measured configs into good (top-gamma) / bad, build per-dimension value
        # densities for each, and bench the unmeasured config maximizing log l(x) - log g(x)
        # (Expected-Improvement proxy), updating the model after each measurement.
        while len(timings) < self.n_trials:
            ranked = sorted(timings, key=timings.get)
            n_good = max(1, round(self.gamma * len(ranked)))
            good_c: dict = defaultdict(lambda: defaultdict(int))
            bad_c: dict = defaultdict(lambda: defaultdict(int))
            for j, i in enumerate(ranked):
                tgt = good_c if j < n_good else bad_c
                for d, v in sigs[i]:
                    tgt[d][v] += 1
            n_bad = len(ranked) - n_good
            best_i, best_score = None, -math.inf
            for i in range(len(configs)):
                if i in timings:
                    continue
                score = 0.0
                for d, v in sigs[i]:
                    V = len(dim_vals[d])
                    lp = (good_c[d][v] + 1.0) / (
                        n_good + V
                    )  # P(v | good), Laplace-smoothed
                    gp = (bad_c[d][v] + 1.0) / (n_bad + V)  # P(v | bad)
                    score += math.log(lp) - math.log(gp)
                if score > best_score:
                    best_score, best_i = score, i
            if best_i is None:
                break
            bench_idx(best_i)

        # Coordinate-descent refinement: try single-dim perturbations around
        # the current best until no neighbor improves.
        if self.refine:
            for _ in range(self.max_refine_iters):
                best_idx = min(timings, key=timings.get)
                best_sig = sigs[best_idx]
                best_ms = timings[best_idx]
                improved = False
                for i, s in enumerate(sigs):
                    if i in timings:
                        continue
                    diff = sum(1 for (_, a), (_, b) in zip(best_sig, s) if a != b)
                    if diff != 1:
                        continue
                    if bench_idx(i) < best_ms:
                        improved = True
                if not improved:
                    break

        self.configs_timings = {configs[i]: t for i, t in timings.items()}
        return configs[min(timings, key=timings.get)]

    def _log_result(self, key, config: Config, ms: float):
        """Append one ``(key, config, ms)`` record as JSONL for offline analysis —
        e.g. pruning configs that are consistently far off the per-key best. ``inf`` ms
        marks a config that failed to compile/run (out of resources, etc.)."""
        if not self.log_path:
            return
        try:
            rec = {
                "timestamp": time.time(),
                "fn_name": getattr(self.fn, "__name__", str(self.fn)),
                "key": list(key),
                "kwargs": config.kwargs,
                "num_warps": config.num_warps,
                "num_stages": config.num_stages,
                "ms": ms,
            }
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except Exception:
            pass

    def _warm_start_index(
        self, configs: list[Config], runtime_fingerprint: str | None = None
    ):
        """Return the index in ``configs`` matching the most recently cached
        key's best config (or ``None`` if no prior tune or no match in the
        current pruned list)."""
        if not self.cache:
            return None
        # Python 3.7+ dicts preserve insertion order. Do not warm-start Thor from a result
        # selected on another SM110 SKU or runtime even when the shape key happens to match.
        for cached_key, prev_best in reversed(self.cache.items()):
            if _runtime_from_tuning_key(cached_key) != runtime_fingerprint:
                continue
            prev_kwargs = prev_best.all_kwargs()
            for i, config in enumerate(configs):
                if config.all_kwargs() == prev_kwargs:
                    return i
        return None

    def check_disk_cache(self, tuning_key, configs, bench_fn):
        """Persist the tuned best config to Triton's on-disk cache so a later run (or process)
        skips the whole minutes-long search+compile — most of which is Triton codegen, not
        benching. Keyed like Triton's own autotune cache (triton version + backend + kernel source
        hash + invalidating env + key + the full config grid), plus the GPU/runtime fingerprint,
        so any of those changing re-tunes.

        Unlike Triton's stock version we do NOT bail when configs carry a ``pre_hook`` (ours
        always do): we persist each config's ``all_kwargs()`` — serializable and pre_hook-free —
        and on a hit re-match it to the live ``Config`` object, so the pre_hook is preserved from
        code rather than lost in (de)serialization. A hit launches only the winning config."""
        if not tuning_key:
            bench_fn()
            return False

        from triton.compiler.compiler import make_backend

        fn = self.fn
        while not isinstance(fn, JITFunction):
            fn = fn.fn
        env_vars = get_cache_invalidating_env_vars()
        runtime_fingerprint = _runtime_from_tuning_key(tuning_key)
        if runtime_fingerprint is None:
            runtime_fingerprint = _runtime_fingerprint_for_args(self.nargs.values())
        cache_key = _disk_cache_key(
            triton_identity=triton_key(),
            backend_hash=make_backend(driver.active.get_current_target()).hash(),
            function_hash=fn.cache_key,
            env_vars=env_vars,
            tuning_key=tuning_key,
            runtime_fingerprint=runtime_fingerprint,
            configs=configs,
        )
        cache = get_cache_manager(cache_key)
        file_name = f"{fn.__name__[:150]}.bayes_autotune.json"

        # signature -> live Config (carries the pre_hook); used to re-match the cached winner
        by_sig = {tuple(sorted(c.all_kwargs().items())): c for c in configs}
        path = cache.get_file(file_name)
        if path:
            try:
                with open(path) as f:
                    data = json.load(f)
                if data.get("runtime_fingerprint") != runtime_fingerprint:
                    raise ValueError("autotune cache runtime fingerprint mismatch")
                best = by_sig.get(tuple(sorted(data["best"].items())))
                if best is not None:
                    self.cache[tuning_key] = best
                    self.configs_timings = {
                        by_sig[s]: t
                        for kw, t in data["timings"]
                        for s in (tuple(sorted(kw.items())),)
                        if s in by_sig
                    }
                    return True
            except Exception:
                pass  # corrupt/stale cache file → fall through and re-tune

        bench_fn()
        try:
            cache.put(
                json.dumps(
                    {
                        "schema": 2,
                        "runtime_fingerprint": runtime_fingerprint,
                        "best": self.cache[tuning_key].all_kwargs(),
                        "timings": [
                            (c.all_kwargs(), t) for c, t in self.configs_timings.items()
                        ],
                    }
                ),
                file_name,
                binary=False,
            )
        except Exception:
            pass
        return False


def bayesian_autotune(
    configs,
    key,
    *,
    n_trials: int = 80,
    n_startup_trials: int = 12,
    gamma: float = 0.25,
    refine: bool = True,
    max_refine_iters: int = 5,
    log_path: str | None = None,
    cache_results: bool = True,
    reset_to_zero=None,
    restore_value=None,
    **kwargs,
):
    """Decorator mirroring ``@triton.autotune``. Extra kwargs:
    n_trials:                 total configs benched per key (TPE budget)
    n_startup_trials:         random seed configs before the TPE model kicks in
    gamma:                    top fraction of measured configs treated as "good"
    refine, max_refine_iters: coordinate-descent refinement after the TPE
    log_path:                 JSONL of benched configs (or FINEGRAINED_AUTOTUNE_LOG)
    cache_results:            persist the tuned best config to disk (on by default) so later
                              runs skip the search+compile — see BayesianAutotuner.check_disk_cache"""

    def decorator(fn):
        return BayesianAutotuner(
            fn,
            fn.arg_names,
            configs,
            key,
            reset_to_zero,
            restore_value,
            n_trials=n_trials,
            n_startup_trials=n_startup_trials,
            gamma=gamma,
            refine=refine,
            max_refine_iters=max_refine_iters,
            log_path=log_path,
            cache_results=cache_results,
            **kwargs,
        )

    return decorator
