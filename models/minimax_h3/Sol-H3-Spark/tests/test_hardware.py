from types import SimpleNamespace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from runtime.hardware import bsa_kernel_module, require_supported_gpu


@pytest.mark.parametrize("capability,module", [
    ((11, 0), "csrc.fwd.sm120_blk64.bsa_fwd_sm120"),
    ((12, 1), "csrc.fwd.sm120_blk64.bsa_fwd_sm120"),
])
def test_supported_device_selects_its_kernel(capability, module):
    torch = SimpleNamespace(cuda=SimpleNamespace(
        get_device_capability=lambda: capability, device_count=lambda: 1))
    assert bsa_kernel_module(require_supported_gpu(torch)) == module


@pytest.mark.parametrize("count,capability", [(0, (11, 0)), (2, (11, 0)), (1, (10, 0)), (1, (12, 0))])
def test_rejects_other_topologies(count, capability):
    torch = SimpleNamespace(cuda=SimpleNamespace(
        get_device_capability=lambda: capability, device_count=lambda: count))
    with pytest.raises(RuntimeError):
        require_supported_gpu(torch)


def test_thor_prepare_adds_bsa_source_without_changing_spark_inputs():
    prepare = Path(__file__).resolve().parents[1] / "prepare.py"
    plans = [json.loads(subprocess.check_output(
        [sys.executable, str(prepare), "--plan", "--hardware", hardware], text=True))
        for hardware in ("spark", "thor")]
    assert "cudnn_frontend_root" not in plans[0]["paths"]
    assert "cudnn_frontend_root" in plans[1]["paths"]
    assert plans[1]["source_patches"] == plans[0]["source_patches"]
