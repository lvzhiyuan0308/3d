"""Meaningful checks against the existing independent NumPy implementation."""

from human3d.settings import (
    work_dir,
    model_path,
)
import json, numpy as np, torch
from human3d.core import Body, project, rasterize

ROOT = work_dir()
from human3d.reference_numpy import SMPLX

path = model_path()
model = Body(path, "cpu")
reference = SMPLX(path)
beta = np.zeros(10)
beta[0] = 0.25
pose = np.zeros(165)
pose[3 + 45 + 2] = -1.2
pose[3 + 48 + 2] = 1.2
pose[3 + 9] = 0.12
with torch.no_grad():
    v, j, A = model.forward(
        torch.tensor(beta, dtype=torch.float32),
        torch.tensor(pose[None], dtype=torch.float32),
    )
    v2, j2, A2 = model.forward(
        torch.tensor(beta, dtype=torch.float32),
        torch.tensor(pose[None], dtype=torch.float32),
        torch.ones((10475, 3)) * 0.001,
    )
refv, refj = reference.forward(beta, pose)
ve = float(np.max(np.abs(v[0].numpy() - refv)))
je = float(np.max(np.abs(j[0].numpy() - refj)))
assert ve < 1e-5, (ve, je)
assert je < 1e-5, (ve, je)
assert torch.equal(j, j2), "Displacements must not change joint centers"
assert torch.equal(A, A2), "Displacements must not change bone transforms"
uv, z = project(np.array([[0, 0, 0], [0, 1, 0]], dtype="float32"), 0, 100, 50, 150)
assert uv[1, 1] < uv[0, 1], "Image row zero must be top"
points = np.array([[1, 1], [6, 1], [1, 6], [1, 1], [6, 1], [1, 6]], dtype="float32")
ids, bc, depth = rasterize(
    points,
    np.array([0, 0, 0, 1, 1, 1], dtype="float32"),
    np.array([[0, 1, 2], [3, 4, 5]], dtype="int32"),
    8,
    8,
)
assert (
    ids[2, 2] == 1 and abs(float(depth[2, 2]) - 1) < 1e-6
), "Z-buffer near-side convention"
report = {
    "smplx_vertex_max_error_m": ve,
    "joint_max_error_m": je,
    "displacements_leave_joints_unchanged": True,
    "image_y_down": True,
    "zbuffer_near_surface": True,
}
(ROOT / "core_validation.json").write_text(
    json.dumps(report, indent=2), encoding="utf8"
)
print("CORE_VALIDATION", json.dumps(report), flush=True)
