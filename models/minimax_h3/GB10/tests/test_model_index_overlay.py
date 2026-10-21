from __future__ import annotations

import json
from pathlib import Path

from model_index_overlay import localise_component_index


def test_localise_component_index_does_not_modify_snapshot(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "transformer").mkdir()
    (snapshot / "transformer" / "config.json").write_text("{}\n")
    original = {
        "transformer": [
            "diffusers",
            "MiniMaxH3Transformer3DModel",
            {"pretrained_model_name_or_path": "MiniMaxAI/MiniMax-H3", "subfolder": "transformer"},
        ]
    }
    index = snapshot / "modular_model_index.json"
    index.write_text(json.dumps(original) + "\n")
    output = tmp_path / "output"
    monkeypatch.setenv("H3_OUTPUT_DIR", str(output))

    overlay = Path(localise_component_index(str(snapshot)))
    assert overlay != snapshot
    assert json.loads(index.read_text()) == original
    localized = json.loads((overlay / "modular_model_index.json").read_text())
    assert localized["transformer"][2]["pretrained_model_name_or_path"] == str(snapshot.resolve())
    assert (overlay / "transformer").is_symlink()
    assert (overlay / "transformer" / "config.json").read_text() == "{}\n"
    assert localise_component_index(str(snapshot)) == str(overlay)


def test_localise_component_index_leaves_absolute_index_untouched(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    index = snapshot / "modular_model_index.json"
    spec = {
        "transformer": [
            "diffusers",
            "MiniMaxH3Transformer3DModel",
            {"pretrained_model_name_or_path": str(snapshot), "subfolder": "transformer"},
        ]
    }
    index.write_text(json.dumps(spec))
    assert localise_component_index(str(snapshot)) == str(snapshot.resolve())
