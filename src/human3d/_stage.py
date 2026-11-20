"""Initialize the local runtime, then execute one allowed pipeline stage."""

from pathlib import Path
import runpy
import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from human3d.offline import install_guard

install_guard()
if os.environ.get("HUMAN3D_LEGACY_USER_SITE") == "1":
    import numpy  # noqa: F401 -- preload the base NumPy ABI before user packages
    import site

    sys.path.insert(0, site.getusersitepackages())

stage = sys.argv[1]
allowed = {
    "prepare",
    "fit",
    "update_canonical",
    "geometry",
    "export_local",
    "validate_delivery",
    "validate_revision",
    "make_review",
    "check_core",
    "doctor",
}
if stage not in allowed:
    raise SystemExit(f"Unknown stage: {stage}")
runpy.run_module("human3d.stages." + stage, run_name="__main__")
