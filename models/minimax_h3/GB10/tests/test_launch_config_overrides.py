from __future__ import annotations

import sys
from pathlib import Path
import tomllib


SANA_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(SANA_ROOT / "scripts"))

import launch_config


THOR_BASELINE = SANA_ROOT / "config" / "minimax_h3" / "thor_baseline.toml"


def test_official_config_overrides_preserve_types_and_source() -> None:
    source = {
        "id": "test",
        "official_config": {"width": 448, "task": "t2va"},
    }

    result = launch_config.apply_official_config_overrides(
        source,
        ["width=1344", "height=768", "seed=0", "guidance_distilled=true"],
    )

    assert source["official_config"] == {"width": 448, "task": "t2va"}
    assert result["official_config"] == {
        "width": 1344,
        "height": 768,
        "seed": 0,
        "guidance_distilled": True,
        "task": "t2va",
    }


def test_official_config_override_accepts_an_unquoted_string() -> None:
    result = launch_config.apply_official_config_overrides(
        {"official_config": {}}, ["suite=official"]
    )
    assert result["official_config"]["suite"] == "official"


def test_official_config_override_rejects_invalid_items() -> None:
    for item in ("width", "=1", "nested.width=1"):
        try:
            launch_config.apply_official_config_overrides({}, [item])
        except SystemExit as exc:
            assert "--official-config" in str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"invalid override was accepted: {item}")


def _thor_profile() -> dict:
    with THOR_BASELINE.open("rb") as handle:
        return tomllib.load(handle)


def test_thor_official_shape_overrides_are_bound_to_runtime_env() -> None:
    data = launch_config.apply_official_config_overrides(
        _thor_profile(), ["width=832", "height=480", "steps=50", "seed=7", "suite=standard"]
    )
    bound = launch_config.bind_minimax_h3_thor_official_config(data)

    assert bound["env"]["H3_WIDTH"] == "832"
    assert bound["env"]["H3_HEIGHT"] == "480"
    assert bound["env"]["H3_STEPS"] == "50"
    assert bound["env"]["H3_SEED"] == "7"
    assert bound["env"]["H3_OFFICIAL"] == "0"
    assert bound["env"]["H3_WARMUP"] == "1"
    assert bound["env"]["H3_MEASURE"] == "3"
    launch_config.validate_minimax_h3_thor_runtime_contract(bound, bound["env"])


def test_thor_formal_request_counts_are_bound_for_standard_and_official() -> None:
    for suite, width, height in (("standard", 832, 480), ("official", 1344, 768)):
        data = launch_config.apply_official_config_overrides(
            _thor_profile(),
            [
                f"suite={suite}",
                f"width={width}",
                f"height={height}",
                "frames=124",
                "steps=50",
            ],
        )
        bound = launch_config.bind_minimax_h3_thor_official_config(data)
        assert bound["env"]["H3_WARMUP"] == "1"
        assert bound["env"]["H3_MEASURE"] == "3"
        launch_config.validate_minimax_h3_thor_runtime_contract(bound, bound["env"])


def test_thor_formal_request_counts_reject_explicit_env_override() -> None:
    data = launch_config.apply_official_config_overrides(
        _thor_profile(),
        ["suite=standard", "width=832", "height=480", "frames=124", "steps=50"],
    )
    bound = launch_config.bind_minimax_h3_thor_official_config(data)
    for name, value in (("H3_WARMUP", "2"), ("H3_MEASURE", "1")):
        env = dict(bound["env"])
        env[name] = value
        try:
            launch_config.validate_minimax_h3_thor_runtime_contract(bound, env)
        except SystemExit as exc:
            assert name in str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"formal request count override was accepted: {name}")


def test_thor_runtime_env_cannot_disagree_with_official_shape() -> None:
    data = launch_config.bind_minimax_h3_thor_official_config(_thor_profile())
    env = dict(data["env"])
    env["H3_WIDTH"] = "1344"

    try:
        launch_config.validate_minimax_h3_thor_runtime_contract(data, env)
    except SystemExit as exc:
        assert "official/runtime contract mismatch" in str(exc)
        assert "width" in str(exc)
    else:
        raise AssertionError("Thor runtime/metadata mismatch was accepted")


def test_thor_official_identity_cannot_claim_an_unpinned_revision() -> None:
    data = launch_config.bind_minimax_h3_thor_official_config(_thor_profile())
    data["official_config"] = dict(data["official_config"], revision="different")

    try:
        launch_config.validate_minimax_h3_thor_runtime_contract(data, data["env"])
    except SystemExit as exc:
        assert "official/runtime contract mismatch" in str(exc)
        assert "revision" in str(exc)
    else:
        raise AssertionError("unpinned Thor revision was accepted")


def test_thor_suite_cannot_claim_a_different_workload_shape() -> None:
    data = launch_config.apply_official_config_overrides(
        _thor_profile(), ["suite=official"]
    )
    data = launch_config.bind_minimax_h3_thor_official_config(data)

    try:
        launch_config.validate_minimax_h3_thor_runtime_contract(data, data["env"])
    except SystemExit as exc:
        assert "official/runtime contract mismatch" in str(exc)
        assert "suite.width" in str(exc)
    else:
        raise AssertionError("official suite with smoke shape was accepted")


def test_thor_official_switch_cannot_be_overridden_by_runtime_env() -> None:
    data = launch_config.bind_minimax_h3_thor_official_config(_thor_profile())
    env = dict(data["env"])
    env["H3_OFFICIAL"] = "1"

    try:
        launch_config.validate_minimax_h3_thor_runtime_contract(data, env)
    except SystemExit as exc:
        assert "H3_OFFICIAL" in str(exc)
    else:
        raise AssertionError("H3_OFFICIAL override was accepted")


def test_thor_runtime_memory_floor_must_be_positive() -> None:
    data = launch_config.bind_minimax_h3_thor_official_config(_thor_profile())
    env = dict(data["env"])
    env["H3_BENCH_RUNTIME_MIN_AVAILABLE_MIB"] = "0"

    try:
        launch_config.validate_minimax_h3_thor_runtime_contract(data, env)
    except SystemExit as exc:
        assert "H3_BENCH_RUNTIME_MIN_AVAILABLE_MIB" in str(exc)
    else:
        raise AssertionError("invalid Thor runtime memory floor was accepted")


def test_thor_runtime_memory_floor_requires_formal_uma_headroom() -> None:
    data = launch_config.bind_minimax_h3_thor_official_config(_thor_profile())
    for value in ("16383", "8192"):
        env = dict(data["env"])
        env["H3_BENCH_RUNTIME_MIN_AVAILABLE_MIB"] = value
        try:
            launch_config.validate_minimax_h3_thor_runtime_contract(data, env)
        except SystemExit as exc:
            assert "16384" in str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"runtime UMA floor was accepted: {value}")

    for value in ("16384", "20000"):
        env = dict(data["env"])
        env["H3_BENCH_RUNTIME_MIN_AVAILABLE_MIB"] = value
        launch_config.validate_minimax_h3_thor_runtime_contract(data, env)


def test_thor_profile_requires_a_suite() -> None:
    data = _thor_profile()
    data["official_config"] = dict(data["official_config"])
    data["official_config"].pop("suite")

    try:
        launch_config.bind_minimax_h3_thor_official_config(data)
    except SystemExit as exc:
        assert "official_config.suite" in str(exc)
    else:
        raise AssertionError("Thor profile without suite was accepted")
