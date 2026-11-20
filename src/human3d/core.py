"""SMPL-X LBS, orthographic cameras and a deterministic CPU z-buffer.

Displacements do not enter the joint regressor. Tensor differentiation stays on
the selected device; the z-buffer is used for visibility and diagnostic renders.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def rasterize(uv, z, faces, h, w):
    depth = np.full((h, w), -1e10, np.float32)
    ids = np.full((h, w), -1, np.int32)
    bary = np.zeros((h, w, 3), np.float32)
    for fi in range(len(faces)):
        a, b, c = faces[fi]
        x0, y0 = uv[a]
        x1, y1 = uv[b]
        x2, y2 = uv[c]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-9:
            continue
        xmin = max(0, int(np.floor(min(x0, x1, x2))))
        xmax = min(w - 1, int(np.ceil(max(x0, x1, x2))))
        ymin = max(0, int(np.floor(min(y0, y1, y2))))
        ymax = min(h - 1, int(np.ceil(max(y0, y1, y2))))
        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                p = ((y1 - y2) * (x + 0.5 - x2) + (x2 - x1) * (y + 0.5 - y2)) / den
                q = ((y2 - y0) * (x + 0.5 - x2) + (x0 - x2) * (y + 0.5 - y2)) / den
                r = 1 - p - q
                if min(p, q, r) < -1e-5:
                    continue
                d = p * z[a] + q * z[b] + r * z[c]
                if d > depth[y, x]:
                    depth[y, x] = d
                    ids[y, x] = fi
                    bary[y, x, 0] = p
                    bary[y, x, 1] = q
                    bary[y, x, 2] = r
    return ids, bary, depth


def normals(v, f):
    n = np.zeros_like(v)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    for j in range(3):
        np.add.at(n, f[:, j], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)


def project(v, yaw, scale, cx, cy):
    c, s = np.cos(yaw), np.sin(yaw)
    vv = np.stack([v[:, 0] * c + v[:, 2] * s, v[:, 1], -v[:, 0] * s + v[:, 2] * c], 1)
    return np.stack([vv[:, 0] * scale + cx, -vv[:, 1] * scale + cy], 1).astype(
        "float32"
    ), vv[:, 2].astype("float32")


class Body:
    def __init__(self, path, device="cuda"):
        import torch
        from smplx.lbs import batch_rodrigues

        self.torch = torch
        self.rodrigues = batch_rodrigues
        self.device = device
        d = np.load(path, allow_pickle=True)
        self.faces = d["f"].astype("int32")
        self.uv = d["vt"]
        self.uv_faces = d["ft"]
        self.parents = d["kintree_table"][0].astype(int)
        self.parents[0] = -1
        for name, key in [
            ("template", "v_template"),
            ("shapedirs", "shapedirs"),
            ("posedirs", "posedirs"),
            ("regressor", "J_regressor"),
            ("weights", "weights"),
        ]:
            value = d[key]
            if name == "shapedirs":
                value = value[:, :, :10]
            setattr(self, name, torch.tensor(value, dtype=torch.float32, device=device))
        mean = np.zeros(165, np.float32)
        mean[75:120] = d["hands_meanl"]
        mean[120:165] = d["hands_meanr"]
        self.mean = torch.tensor(mean, device=device)

    def forward(self, beta, pose, offset=None):
        t = self.torch
        B = pose.shape[0]
        shaped = self.template + t.einsum("vcn,n->vc", self.shapedirs, beta)
        J = self.regressor @ shaped
        rot = self.rodrigues((pose + self.mean).reshape(-1, 3)).reshape(B, 55, 3, 3)
        feat = (rot[:, 1:] - t.eye(3, device=self.device)).reshape(B, -1)
        posed = shaped[None] + t.einsum("vcp,bp->bvc", self.posedirs, feat)
        if offset is not None:
            posed = posed + offset[None]
        G = []
        for j in range(55):
            rel = J[j] if j == 0 else J[j] - J[self.parents[j]]
            top = t.cat([rot[:, j], rel[None, :, None].expand(B, -1, -1)], 2)
            bottom = t.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)[
                None, None
            ].expand(B, 1, 4)
            m = t.cat([top, bottom], 1)
            G.append(m if j == 0 else G[self.parents[j]] @ m)
        G = t.stack(G, 1)
        A = G.clone()
        A[:, :, :3, 3] = G[:, :, :3, 3] - t.einsum("bjkl,jl->bjk", G[:, :, :3, :3], J)
        T = t.einsum("vj,bjkl->bvkl", self.weights, A)
        v = t.einsum("bvkl,bvl->bvk", T[:, :, :3, :3], posed) + T[:, :, :3, 3]
        return v, G[:, :, :3, 3], A


def torch_project(v, cameras):
    import torch

    yaw, logscale, cx, cy = cameras.unbind(1)
    x = v[:, :, 0] * yaw.cos()[:, None] + v[:, :, 2] * yaw.sin()[:, None]
    return torch.stack(
        [
            x * logscale.exp()[:, None] + cx[:, None],
            -v[:, :, 1] * logscale.exp()[:, None] + cy[:, None],
        ],
        2,
    )
