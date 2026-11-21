"""Stages 2/3: shared-shape fitting and bounded normal displacement.

Uses silhouette distance fields and foreground coverage, with explicit pose and
shape priors. No Sapiens normal loss is claimed when its weights are absent.
"""

from human3d.settings import (
    work_dir,
    model_path,
    compute_device,
    load_settings,
)
import os, json, time

os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, cv2, torch
from scipy.ndimage import distance_transform_edt
from human3d.core import Body, torch_project, project, normals, rasterize
from human3d.fit_terms import keypoint_term, view_weight_vector

ROOT = work_dir()
OUT = ROOT / "fit"
OUT.mkdir(exist_ok=True)
torch.set_num_threads(4)
torch.manual_seed(17)
np.random.seed(17)
MODEL = model_path()
device = compute_device()
body = Body(MODEL, device)
views = json.loads((ROOT / "prepared/views.json").read_text())
CFG = load_settings()
HEIGHT_M = CFG.height_m  # one source of truth for the height prior (config: subject.height_m)
HEIGHT_WEIGHT = CFG.fit_height_weight
# Camera scale is initialised from mask height / this value (mask includes hair and shoes).
CAMERA_INIT_HEIGHT_M = 1.77
jointmap = {
    "l_hip": 1,
    "r_hip": 2,
    "l_knee": 4,
    "r_knee": 5,
    "l_ankle": 7,
    "r_ankle": 8,
    "l_shoulder": 16,
    "r_shoulder": 17,
    "l_elbow": 18,
    "r_elbow": 19,
    "l_wrist": 20,
    "r_wrist": 21,
}
kpnames = list(jointmap) + ["nose", "l_eye", "r_eye"]
vids = [9120, 9448, 9929]
gt = []
conf = []
sdfs = []
targets = []
labelmaps = []
samples = np.random.choice(len(body.template), 2400, replace=False)
samples = torch.tensor(samples, device=device)
for view in views:
    name = view["name"]
    w, h = view["size"]
    kp = view["keypoints"]
    gt.append([kp[n][:2] for n in kpnames])
    weight = np.array([kp[n][2] for n in kpnames], np.float32)
    if name == "back":
        weight[-3:] = 0
    if name in ["left", "right"]:
        weight[:] *= 0.6
    for n in ["l_hip", "r_hip", "l_shoulder", "r_shoulder"]:
        weight[kpnames.index(n)] *= 0.55
    conf.append(weight)
    mask = cv2.imread(str(ROOT / "prepared" / f"{name}_mask.png"), 0) > 0
    labels = cv2.imread(str(ROOT / "prepared" / f"{name}_labels.png"), 0)
    sdf = distance_transform_edt(~mask) - distance_transform_edt(mask)
    sdfs.append(torch.tensor(sdf, dtype=torch.float32, device=device)[None, None])
    # Only skin/clothing pixels guide coverage; hair and shoes are separate layers.
    ys, xs = np.where((labels > 0) & (labels < 4))
    ii = np.random.choice(len(xs), min(500, len(xs)), replace=False)
    xy = np.stack([xs[ii], ys[ii]], 1).astype("float32")
    weights = np.where(labels[ys[ii], xs[ii]] == 1, 1.0, 0.10).astype("float32")
    targets.append(
        (torch.tensor(xy, device=device), torch.tensor(weights, device=device))
    )
gt = torch.tensor(np.array(gt), device=device)
# Per-view weights from views.json (all ones unless [fit] use_view_weights = true).
view_w = view_weight_vector(views, "geometry_weight", CFG.fit_use_view_weights, device)
conf = torch.tensor(np.array(conf), device=device) * view_w[:, None]
image_heights = torch.tensor([v["size"][1] for v in views], dtype=torch.float32, device=device)
pose0 = np.zeros((4, 165), np.float32)
pose0[:, 3 + (16 - 1) * 3 + 2] = -1.40
pose0[:, 3 + (17 - 1) * 3 + 2] = 1.40
pose0[:, 3 + (18 - 1) * 3 + 1] = -0.03
pose0[:, 3 + (19 - 1) * 3 + 1] = 0.03
basepose = torch.tensor(pose0, device=device)
beta = torch.nn.Parameter(torch.zeros(10, device=device))
delta = torch.nn.Parameter(torch.zeros((4, 63), device=device))
cameras = []
with torch.no_grad():
    v, _, _ = body.forward(beta, basepose)
for i, view in enumerate(views):
    w, h = view["size"]
    mask = cv2.imread(str(ROOT / "prepared" / (view["name"] + "_mask.png")), 0) > 0
    y, x = np.where(mask)
    scale = (y.max() - y.min()) / CAMERA_INIT_HEIGHT_M
    cy = (y.max() + y.min()) / 2 + float(
        (v[i, :, 1].max() + v[i, :, 1].min()) / 2
    ) * scale
    cameras.append(
        [np.deg2rad(view["yaw_deg"]), np.log(scale), float(np.median(x)), cy]
    )
cameras = torch.nn.Parameter(torch.tensor(cameras, dtype=torch.float32, device=device))
cam0 = cameras.detach().clone()


def poses():
    return torch.cat([basepose[:, :3], basepose[:, 3:66] + delta, basepose[:, 66:]], 1)


def terms(v, j):
    points = torch.cat([j[:, list(jointmap.values())], v[:, vids]], 1)
    pred = torch_project(points, cameras)
    # Pixel-space Geman-McClure prevents a bad detector point from dominating.
    kp = keypoint_term(pred, gt, conf, image_heights)
    pp = torch_project(v[:, samples], cameras)
    outside = kp * 0
    cover = kp * 0
    for i, view in enumerate(views):
        w, h = view["size"]
        grid = pp[i].clone()
        grid[:, 0] = grid[:, 0] / (w - 1) * 2 - 1
        grid[:, 1] = grid[:, 1] / (h - 1) * 2 - 1
        sdf = torch.nn.functional.grid_sample(
            sdfs[i], grid[None, None], align_corners=True, padding_mode="border"
        ).reshape(-1)
        outside = outside + (sdf.relu() / h).square().mean() * view_w[i]
        xy, ww = targets[i]
        dist = torch.cdist(xy[None] / h, pp[i : i + 1] / h).squeeze(0).amin(1)
        cover = cover + (dist.square() * ww).mean() * view_w[i]
    return kp, outside / 4, cover / 4


log = []
start = time.time()
opt = torch.optim.Adam(
    [
        {"params": [cameras], "lr": 0.015},
        {"params": [beta], "lr": 0.025},
        {"params": [delta], "lr": 0.004},
    ]
)
for step in range(300):
    opt.zero_grad()
    v, j, _ = body.forward(beta, poses())
    kp, outer, coverage = terms(v, j)
    shaped = body.template + torch.einsum("vcn,n->vc", body.shapedirs, beta)
    height = shaped[:, 1].max() - shaped[:, 1].min()
    prior = (
        0.003 * beta.square().mean()
        + 0.07 * delta.square().mean()
        + 0.02 * (delta - delta.mean(0)).square().mean()
    )
    # Config subject.height_m; unless subject.height_measured is set it is only a prior.
    loss = (
        kp
        + outer * 12
        + coverage * 3
        + prior
        + (height - HEIGHT_M).square() * HEIGHT_WEIGHT
    )
    if step < 60:
        loss = kp + outer * 4 + coverage + prior + (beta * 0).sum()
    loss.backward()
    if step < 60:
        beta.grad.zero_()
        delta.grad.zero_()
    opt.step()
    with torch.no_grad():
        beta.clamp_(-2.5, 2.5)
        delta.clamp_(-0.35, 0.35)
        cameras[:, 0].copy_(
            torch.maximum(
                torch.minimum(cameras[:, 0], cam0[:, 0] + 0.18), cam0[:, 0] - 0.18
            )
        )
    if step % 25 == 0 or step == 299:
        row = {
            "stage": 2,
            "step": step,
            "loss": float(loss),
            "kp": float(kp),
            "outside": float(outer),
            "coverage": float(coverage),
        }
        log.append(row)
        print(json.dumps(row), flush=True)

with torch.no_grad():
    p = poses().detach()
    vv, jj, _ = body.forward(beta, p)
    canon = p[0:1].clone()
    canon[:, :3] = 0
    canonical, _, _ = body.forward(beta, canon)
    zero = torch.zeros_like(canon)
    rest, _, _ = body.forward(beta, zero)
    basev = vv.detach()
    basen = normals(rest[0].cpu().numpy(), body.faces)

# Stage 3: conservative bounded displacements on skin/tight garments only.
u = torch.nn.Parameter(torch.zeros(len(body.template), device=device))
n = torch.tensor(basen, dtype=torch.float32, device=device)
weights = body.weights.detach().cpu().numpy()
dom = weights.argmax(1)
dmax = np.full(len(u), 0.008, np.float32)
dmax[np.isin(dom, [0, 3, 6, 9, 13, 14])] = 0.018
dmax[np.isin(dom, [12, 15, 22, 23, 24])] = 0.003
dmax = torch.tensor(dmax, device=device)
edge = np.unique(
    np.sort(
        np.concatenate(
            [body.faces[:, [0, 1]], body.faces[:, [1, 2]], body.faces[:, [2, 0]]]
        ),
        axis=1,
    ),
    axis=0,
)
edge = torch.tensor(edge, device=device).long()
optim = torch.optim.Adam([u], lr=0.025)
for step in range(160):
    optim.zero_grad()
    d = dmax * torch.tanh(u)
    offset = n * d[:, None]
    v, j, _ = body.forward(beta.detach(), p, offset)
    kp, outer, coverage = terms(v, j)
    smooth = (d[edge[:, 0]] - d[edge[:, 1]]).square().mean()
    loss = outer * 20 + coverage * 5 + smooth * 15 + d.square().mean() * 0.3
    loss.backward()
    optim.step()
    if step % 40 == 0 or step == 159:
        row = {
            "stage": 3,
            "step": step,
            "loss": float(loss),
            "mean_abs_displacement_mm": float(d.abs().mean() * 1000),
        }
        log.append(row)
        print(json.dumps(row), flush=True)
with torch.no_grad():
    offset = n * (dmax * torch.tanh(u))[:, None]
    v, j, A = body.forward(beta, p, offset)
    cv, cj, cA = body.forward(beta, canon, offset)
    rv, rj, rA = body.forward(beta, zero, offset)
np.savez_compressed(
    OUT / "fit_result.npz",
    betas=beta.detach().cpu().numpy(),
    poses=p.cpu().numpy(),
    canonical_pose=canon.cpu().numpy(),
    cameras=cameras.detach().cpu().numpy(),
    view_vertices=v.cpu().numpy(),
    view_joints=j.cpu().numpy(),
    canonical_vertices=cv[0].cpu().numpy(),
    canonical_joints=cj[0].cpu().numpy(),
    rest_vertices=rv[0].cpu().numpy(),
    offset=offset.cpu().numpy(),
    faces=body.faces,
    uv=body.uv,
    uv_faces=body.uv_faces,
    weights=weights,
    parents=body.parents,
    canonical_skinning=cA[0].cpu().numpy(),
    view_skinning=A.cpu().numpy(),
)
diagnostics = []
for i, view in enumerate(views):
    name = view["name"]
    im = cv2.imread(str(ROOT / "prepared" / (name + ".png")))
    h, w = im.shape[:2]
    ca = cameras[i].detach().cpu().numpy()
    uv, z = project(v[i].cpu().numpy(), ca[0], np.exp(ca[1]), ca[2], ca[3])
    ids, bc, depth = rasterize(uv, z, body.faces, h, w)
    mask = ids >= 0
    target = cv2.imread(str(ROOT / "prepared" / (name + "_mask.png")), 0) > 0
    over = im.copy()
    over[mask] = (over[mask] * 0.55 + np.array([190, 180, 30]) * 0.45).astype("uint8")
    cv2.imwrite(str(OUT / (name + "_body_overlay.png")), over)
    pts = torch.cat([j[:, list(jointmap.values())], v[:, vids]], 1)
    pred = torch_project(pts, cameras)[i].detach().cpu().numpy()
    cf = conf[i].cpu().numpy()
    rmse = np.sqrt((((pred - gt[i].cpu().numpy()) ** 2).sum(1) * cf).sum() / cf.sum())
    diagnostics.append(
        {
            "view": name,
            "joint_rmse_px": float(rmse),
            "body_vs_clothed_mask_iou": float(
                (mask & target).sum() / (mask | target).sum()
            ),
        }
    )
report = {
    "device": device,
    "torch": torch.__version__,
    "model_path": str(MODEL),
    "runtime_s": time.time() - start,
    "shared_betas": beta.detach().cpu().tolist(),
    "views": diagnostics,
    "normal_loss_used": False,
    "normal_loss_reason": "No Sapiens weights installed. Image-derived surface normals were not fabricated.",
    "height_prior_m": HEIGHT_M,
    "height_is_measured": CFG.height_is_measured,
    "height_prior_weight": HEIGHT_WEIGHT,
    "view_weights_used": bool(CFG.fit_use_view_weights),
    "view_weights": view_w.cpu().tolist(),
    "generated_views_are_ground_truth": False,
    "log": log,
}
(OUT / "fit_report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
print("FIT_COMPLETE", json.dumps(diagnostics), flush=True)
