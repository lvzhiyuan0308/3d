"""Synthetic SMPL-X-shaped model + four-view inputs for end-to-end regression runs.

The body is a tall ellipsoid with 55 joints, NOT a human: it only has the same array layout as an
SMPL-X .npz (10242 vertices >= the landmark vertex ids the fit stage uses), so the fit stage can be
executed and compared between code versions without the real weights or photographs.
"""

import json
from pathlib import Path

import cv2
import numpy as np

JOINT_TARGETS = {  # SMPL-X joint index -> rough position (metres) on the ellipsoid
    0: (0, 0.95, 0),
    1: (0.09, 0.9, 0),
    2: (-0.09, 0.9, 0),
    4: (0.09, 0.5, 0),
    5: (-0.09, 0.5, 0),
    7: (0.09, 0.08, 0),
    8: (-0.09, 0.08, 0),
    12: (0, 1.5, 0),
    15: (0, 1.65, 0),
    16: (0.18, 1.42, 0),
    17: (-0.18, 1.42, 0),
    18: (0.2, 1.15, 0),
    19: (-0.2, 1.15, 0),
    20: (0.21, 0.9, 0),
    21: (-0.21, 0.9, 0),
}


def _subdivide(v, f):
    """One level of midpoint subdivision, new vertices projected back onto the unit sphere."""
    cache, verts, faces = {}, list(v), []

    def mid(a, b):
        key = (min(a, b), max(a, b))
        if key not in cache:
            m = (verts[a] + verts[b]) / 2
            verts.append(m / np.linalg.norm(m))
            cache[key] = len(verts) - 1
        return cache[key]

    for a, b, c in f:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
    return np.array(verts), np.array(faces)


def icosphere(levels=5):
    t = (1 + 5**0.5) / 2
    v = np.array(
        [
            [-1, t, 0],
            [1, t, 0],
            [-1, -t, 0],
            [1, -t, 0],
            [0, -1, t],
            [0, 1, t],
            [0, -1, -t],
            [0, 1, -t],
            [t, 0, -1],
            [t, 0, 1],
            [-t, 0, -1],
            [-t, 0, 1],
        ],
        float,
    )
    f = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ]
    )
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    for _ in range(levels):
        v, f = _subdivide(v, f)
    return v, f.astype("int32")


def make_fake_smplx(path, seed=0):
    rng = np.random.default_rng(seed)
    unit, faces = icosphere(5)
    V = len(unit)
    template = unit * np.array([0.22, 0.85, 0.13]) + np.array([0, 0.85, 0])
    J = 55
    centers = np.zeros((J, 3))
    centers[:, 1] = np.linspace(0.3, 1.6, J)
    for j, p in JOINT_TARGETS.items():
        centers[j] = p
    d2 = ((template[None] - centers[:, None]) ** 2).sum(-1)  # (J,V)
    regressor = np.zeros((J, V))
    for j in range(J):
        regressor[j, np.argsort(d2[j])[:20]] = 1 / 20
    weights = np.exp(-d2.T / 0.04)
    weights /= weights.sum(1, keepdims=True)
    shapedirs = rng.normal(size=(V, 3, 10)) * 0.002
    shapedirs[:, :, 0] = (template - template.mean(0)) * 0.1  # beta0 ~ overall scale
    parents = np.zeros(J, int)
    parents[1:] = np.maximum(np.arange(1, J) // 3, 0)
    parents[0] = -1
    np.savez(
        path,
        f=faces,
        vt=rng.random((V, 2)),
        ft=faces,
        kintree_table=np.stack([parents, np.arange(J)]),
        v_template=template,
        shapedirs=shapedirs,
        posedirs=rng.normal(size=(V, 3, 486)) * 5e-4,
        J_regressor=regressor,
        weights=weights,
        hands_meanl=np.zeros(45),
        hands_meanr=np.zeros(45),
    )
    return template, faces, regressor


def make_inputs(root, panel_h=534, panel_w=200, heights=None, extra_toml=""):
    """Write model, config.toml and prepared/ (masks, labels, images, views.json) under root."""
    from human3d.core import project, rasterize

    root = Path(root)
    (root / "work/prepared").mkdir(parents=True, exist_ok=True)
    template, faces, regressor = make_fake_smplx(root / "model.npz")
    (root / "config.toml").write_text(
        '[paths]\nsmplx_model = "model.npz"\noriginal_photo = "p.jpg"\nturnaround = "t.png"\nwork_dir = "work"\n'
        '[runtime]\ndevice = "cpu"\n' + extra_toml,
        encoding="utf8",
    )
    names = [("front", 0), ("left", 90), ("right", -90), ("back", 180)]
    jmap = {
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
    vids = {"nose": 9120, "l_eye": 9448, "r_eye": 9929}
    views = []
    heights = heights or [panel_h] * 4
    for (name, yaw), h in zip(names, heights):
        w = panel_w
        scale = h * 0.9 / 1.7
        uv, z = project(
            template.astype("float32"),
            np.deg2rad(yaw),
            scale,
            w / 2,
            h * 0.5 + 0.85 * scale,
        )
        ids, _, _ = rasterize(uv, z, faces, h, w)
        mask = (ids >= 0).astype("uint8")
        joints = regressor @ template
        juv, _ = project(
            joints.astype("float32"),
            np.deg2rad(yaw),
            scale,
            w / 2,
            h * 0.5 + 0.85 * scale,
        )
        kp = {n: [float(juv[i, 0]), float(juv[i, 1]), 0.9] for n, i in jmap.items()}
        for n, i in vids.items():
            kp[n] = [float(uv[i, 0]), float(uv[i, 1]), 0.9]
        labels = mask.copy()
        labels[: int(h * 0.12)] *= 4
        cv2.imwrite(str(root / "work/prepared" / f"{name}_mask.png"), mask * 255)
        cv2.imwrite(str(root / "work/prepared" / f"{name}_labels.png"), labels)
        cv2.imwrite(
            str(root / "work/prepared" / f"{name}.png"), np.full((h, w, 3), 90, "uint8")
        )
        views.append(
            {
                "name": name,
                "yaw_deg": yaw,
                "crop": [0, 0, w, h],
                "size": [w, h],
                "keypoints": kp,
                "source": "generated_reference",
                "geometry_weight": 0.6,
                "texture_weight": 0.7,
            }
        )
    (root / "work/prepared/views.json").write_text(json.dumps(views), encoding="utf8")
    return root / "config.toml"
