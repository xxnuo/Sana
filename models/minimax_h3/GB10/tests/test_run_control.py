from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SANA_ROOT = Path(__file__).resolve().parents[4]
SCRIPTS_ROOT = SANA_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_ROOT))

import collect_run
import launch_config


def _load_run_module():
    spec = importlib.util.spec_from_file_location("h3_run_control", SCRIPTS_ROOT / "run.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_script = _load_run_module()


def _write_metadata(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "config_id": "minimax_h3_thor_test",
                "status": "prepared",
                "status_history": [
                    {"status": "prepared", "at_utc": "test", "reason": "bundle_created"}
                ],
            }
        )
    )


@pytest.mark.parametrize(
    ("exit_code", "expected_status"),
    ((0, "completed"), (7, "failed")),
)
def test_run_local_persists_terminal_lifecycle(
    tmp_path: Path, exit_code: int, expected_status: str
) -> None:
    run_dir = tmp_path / "run"
    _write_metadata(run_dir)
    launcher = tmp_path / "launch.sh"
    launcher.write_text(f"#!/usr/bin/env bash\nexit {exit_code}\n")
    launcher.chmod(0o755)

    assert launch_config.run_local(run_dir, launcher) == exit_code

    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["status"] == expected_status
    assert metadata["returncode"] == exit_code
    assert metadata["started_at_utc"]
    assert metadata["finished_at_utc"]
    assert [item["status"] for item in metadata["status_history"]] == [
        "prepared",
        "running",
        expected_status,
    ]


def test_run_local_records_spawn_failure(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_metadata(run_dir)

    assert launch_config.run_local(run_dir, tmp_path / "missing-launch.sh") == 126

    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["returncode"] == 126
    assert metadata["status_reason"].startswith("local_launch_error:")
    assert [item["status"] for item in metadata["status_history"]] == [
        "prepared",
        "running",
        "failed",
    ]


def test_run_local_records_an_interrupted_launcher(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    _write_metadata(run_dir)

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(launch_config.subprocess, "run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        launch_config.run_local(run_dir, tmp_path / "launch.sh")

    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["returncode"] == 130
    assert metadata["status_reason"] == "local_launch_interrupted: KeyboardInterrupt"


def test_recorded_running_run_remains_a_single_flight_blocker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runs_root = tmp_path / "runs"
    active_run = runs_root / "active"
    active_run.mkdir(parents=True)
    (active_run / "metadata.json").write_text(
        json.dumps(
            {
                "config_id": "already_recorded",
                "kind": "baseline",
                "purpose": "control",
                "run_dir": "runs/active",
                "status": "running",
            }
        )
    )
    (tmp_path / "AGENT-STATUS.json").write_text(
        json.dumps({"config": [{"run_dir": "runs/active"}]})
    )
    monkeypatch.setattr(launch_config, "repo_root", lambda: tmp_path)
    monkeypatch.delenv("AUTO_VIDEO_DISABLE_SINGLE_FLIGHT_GUARD", raising=False)
    args = argparse.Namespace(mode="local", run_root="runs", confirm_submit=False)

    with pytest.raises(SystemExit, match=r"active.*status=running"):
        launch_config.enforce_single_flight_or_exit(
            args,
            {
                "id": "minimax_h3_thor_new_control",
                "kind": "baseline",
                "purpose": "control",
            },
        )


def test_run_frontend_checks_single_flight_before_preparing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    def reject(_args, _data):
        events.append("guard")
        raise SystemExit("blocked")

    def prepare(_args, _data):  # pragma: no cover - must not be reached
        events.append("prepare")
        raise AssertionError("bundle was created before the guard")

    monkeypatch.setattr(run_script.launch_config, "enforce_single_flight_or_exit", reject)
    monkeypatch.setattr(run_script.launch_config, "prepare_run", prepare)

    with pytest.raises(SystemExit, match="blocked"):
        run_script.prepare_guarded_local_run(
            argparse.Namespace(mode="local", run_root="runs"),
            {"id": "minimax_h3_thor_test"},
        )

    assert events == ["guard"]


@pytest.mark.parametrize(
    ("contents", "message"),
    (
        (None, "is missing"),
        ("{", "is malformed"),
        ("[]", "JSON object"),
        ("{}", "must not be empty"),
    ),
)
def test_required_json_rejects_missing_or_invalid_evidence(
    tmp_path: Path, contents: str | None, message: str
) -> None:
    path = tmp_path / "evidence.json"
    if contents is not None:
        path.write_text(contents)

    with pytest.raises(SystemExit, match=message):
        collect_run.load_required_json(path, "evidence.json")


@pytest.mark.parametrize(
    ("contents", "message"),
    ((None, "is missing"), ("id =", "is malformed"), ("", "must not be empty")),
)
def test_required_toml_rejects_missing_or_invalid_evidence(
    tmp_path: Path, contents: str | None, message: str
) -> None:
    path = tmp_path / "manifest.resolved.toml"
    if contents is not None:
        path.write_text(contents)

    with pytest.raises(SystemExit, match=message):
        collect_run.load_required_toml(path, "manifest.resolved.toml")


@pytest.mark.parametrize("benchmark_contents", (None, "{", "{}"))
def test_collector_does_not_rebuild_invalid_runner_benchmark(
    tmp_path: Path, benchmark_contents: str | None
) -> None:
    run_dir = tmp_path / "run"
    outputs = run_dir / "outputs"
    outputs.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps({"config_id": "minimax_h3_thor_test", "status": "running"})
    )
    (run_dir / "manifest.resolved.toml").write_text(
        '[artifacts]\noutput_dir = "outputs"\nbenchmark = "benchmark.json"\n'
    )
    benchmark = outputs / "benchmark.json"
    if benchmark_contents is not None:
        benchmark.write_text(benchmark_contents)

    with pytest.raises(SystemExit):
        collect_run.collect(SimpleNamespace(run_dir=str(run_dir)))

    if benchmark_contents is None:
        assert not benchmark.exists()
    else:
        assert benchmark.read_text() == benchmark_contents
    for name in ("quality.json", "risk_notes.md", "collection.json", "patch_summary.md"):
        assert not (outputs / name).exists()


@pytest.mark.parametrize("metadata_contents", (None, "{", "[]", "{}"))
def test_collector_writes_nothing_for_invalid_metadata(
    tmp_path: Path, metadata_contents: str | None
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metadata = run_dir / "metadata.json"
    if metadata_contents is not None:
        metadata.write_text(metadata_contents)

    with pytest.raises(SystemExit):
        collect_run.collect(SimpleNamespace(run_dir=str(run_dir)))

    assert not (run_dir / "outputs").exists()
    if metadata_contents is None:
        assert not metadata.exists()
    else:
        assert metadata.read_text() == metadata_contents


@pytest.mark.parametrize("manifest_contents", (None, "id =", ""))
def test_collector_writes_nothing_for_invalid_manifest(
    tmp_path: Path, manifest_contents: str | None
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps({"config_id": "minimax_h3_thor_test", "status": "running"})
    )
    manifest = run_dir / "manifest.resolved.toml"
    if manifest_contents is not None:
        manifest.write_text(manifest_contents)

    with pytest.raises(SystemExit):
        collect_run.collect(SimpleNamespace(run_dir=str(run_dir)))

    assert not (run_dir / "outputs").exists()
