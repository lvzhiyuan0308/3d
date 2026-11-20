"""Export a neutral standing pose, leaving the independently fitted views intact."""

from human3d.settings import (
    work_dir,
    model_path,
    compute_device,
)
import numpy as np, torch
from human3d.core import Body

ROOT = work_dir()
p = ROOT / "fit/fit_result.npz"
d = dict(np.load(p))
device = compute_device()
b = Body(model_path(), device)
pose = torch.zeros((1, 165), device=device)
pose[0, 3 + 45 + 2] = -1.40
pose[0, 3 + 48 + 2] = 1.40
pose[0, 3 + 51 + 1] = -0.03
pose[0, 3 + 54 + 1] = 0.03
with torch.no_grad():
    v, j, A = b.forward(
        torch.tensor(d["betas"], device=device),
        pose,
        torch.tensor(d["offset"], device=device),
    )
d.update(
    canonical_pose=pose.cpu().numpy(),
    canonical_vertices=v[0].cpu().numpy(),
    canonical_joints=j[0].cpu().numpy(),
    canonical_skinning=A[0].cpu().numpy(),
)
np.savez_compressed(p, **d)
print("CANONICAL_STANDING_POSE_UPDATED")
