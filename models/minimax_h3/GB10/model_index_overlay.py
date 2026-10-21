"""Build a read-only local-path overlay for a MiniMax-H3 component index."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def localise_component_index(snapshot: str) -> str:
    """Return a model root whose component entries resolve to ``snapshot`` locally.

    The source snapshot is never modified. When localization is needed, the returned directory
    contains symlinks to every source entry plus an atomically-written
    ``modular_model_index.json`` under ``H3_OUTPUT_DIR``.
    """
    source = Path(snapshot).resolve()
    index = source / "modular_model_index.json"
    with index.open() as handle:
        spec = json.load(handle)

    changed = False
    for entry in spec.values():
        if isinstance(entry, list) and len(entry) == 3 and isinstance(entry[2], dict):
            path = entry[2].get("pretrained_model_name_or_path")
            if path and not os.path.isabs(path):
                entry[2]["pretrained_model_name_or_path"] = str(source)
                changed = True
    if not changed:
        return str(source)

    digest = hashlib.sha256(
        str(source).encode("utf-8") + b"\0" + index.read_bytes()
    ).hexdigest()[:20]
    output_root = Path(
        os.environ.get(
            "H3_OUTPUT_DIR",
            str(Path(tempfile.gettempdir()) / "minimax-h3-runtime"),
        )
    )
    overlay = output_root / ".localized-model-index" / digest
    overlay.mkdir(parents=True, exist_ok=True)

    for child in source.iterdir():
        if child.name == index.name:
            continue
        destination = overlay / child.name
        target = str(child)
        if destination.is_symlink() or destination.exists():
            if destination.is_symlink() and os.readlink(destination) == target:
                continue
            if destination.is_symlink():
                destination.unlink()
            else:
                raise RuntimeError(f"localized model overlay has unexpected path: {destination}")
        os.symlink(target, destination, target_is_directory=child.is_dir())

    payload = (json.dumps(spec, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=".modular_model_index.", dir=overlay)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, overlay / index.name)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    return str(overlay)
