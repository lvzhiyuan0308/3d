"""Read-only runtime and input diagnostics for an actual pipeline worker."""

import importlib
import json
import sys

from human3d.settings import load_settings, model_path, compute_device

settings = load_settings()
report = {
    "python": sys.version.split()[0],
    "config": str(settings.config),
    "packages": {},
}
for name in (
    "numpy",
    "torch",
    "cv2",
    "scipy",
    "numba",
    "trimesh",
    "skimage",
    "PIL",
    "smplx",
):
    module = importlib.import_module(name)
    report["packages"][name] = getattr(module, "__version__", "imported")
import numpy as np
import torch

report["device"] = compute_device()
report["cuda_available"] = torch.cuda.is_available()
if report["cuda_available"]:
    report["gpu"] = torch.cuda.get_device_name(0)
report["inputs_exist"] = {
    "photo": settings.photo.is_file(),
    "turnaround": settings.turnaround.is_file(),
}
with np.load(model_path(), allow_pickle=True) as model:
    required = [
        "v_template",
        "f",
        "weights",
        "shapedirs",
        "posedirs",
        "lmk_faces_idx",
        "lmk_bary_coords",
    ]
    report["model_arrays"] = {key: list(model[key].shape) for key in required}
    if model["weights"].shape != (10475, 55):
        raise ValueError("This pipeline expects SMPL-X 10475 vertices / 55 joints")
try:
    import mediapipe as mp

    report["mediapipe_solutions"] = hasattr(mp, "solutions")
except ImportError:
    report["mediapipe_solutions"] = False
report["python_socket_guard"] = True
print(json.dumps(report, indent=2))
if not all(report["inputs_exist"].values()):
    raise SystemExit(
        "Missing input images; configure paths or bootstrap the previous local run"
    )
