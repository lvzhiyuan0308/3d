from human3d.settings import (
    model_path,
)

"""Anatomical skin plus fitted separate garment/hair shells.

This uses parametric cloth construction for the silhouette-only branch. It
does not claim learned normal-map reconstruction of folds or hair strands.
"""
import numpy as np, trimesh
from scipy.spatial import cKDTree, ConvexHull


def scalp_cap_mask(can, faces, in_head, head_center):
    """Faces of the scalp cap: all vertices on the head and above a tilted plane (vectorised)."""
    p = can[faces].mean(1)
    front = p[:, 2] - head_center[2]
    threshold = head_center[1] - 0.055 + np.clip((front + 0.045) / 0.10, 0, 1) * 0.13
    return in_head[faces].all(1) & (p[:, 1] > threshold)


def boundary_cap_faces(boundary, n):
    """Two triangles per boundary edge (a, b) joining a shell to its offset copy (vectorised)."""
    boundary = np.asarray(boundary, dtype=int).reshape(-1, 2)
    a, b = boundary[:, 0], boundary[:, 1]
    return np.stack([np.stack([b, a, a + n], 1), np.stack([b, a + n, b + n], 1)], 1).reshape(
        -1, 3
    )


def construct(d):
    rest = d["rest_vertices"]
    base = trimesh.Trimesh(rest, d["faces"], process=False)
    raw = np.load(model_path(), allow_pickle=True)
    shaped = raw["v_template"] + raw["shapedirs"][:, :, :10] @ d["betas"]
    J = raw["J_regressor"] @ shaped
    tree = cKDTree(rest)
    A = d["canonical_skinning"]
    pieces = []
    names = []

    def add(name, v, f, posed=False, solid=True, thickness=0.003):
        v = np.asarray(v, float)
        f = np.asarray(f, int)
        if not posed:
            dist, idx = tree.query(v, k=4)
            w = 1 / np.maximum(dist, 0.001) ** 2
            w /= w.sum(1, keepdims=True)
            W = np.einsum("nk,nkj->nj", w, d["weights"][idx])
            T = np.einsum("vj,jab->vab", W, A)
            v = np.einsum("vab,vb->va", T[:, :3, :3], v) + T[:, :3, 3]
        m = trimesh.Trimesh(v, f, process=False)
        if solid:
            n = m.vertex_normals
            vv = np.concatenate([v, v - n * thickness])
            N = len(v)
            ff = np.concatenate([f, f[:, ::-1] + N])
            edges = m.edges
            ii = trimesh.grouping.group_rows(m.edges_sorted, require_count=1)
            boundary = edges[ii]
            ff = np.concatenate([ff, boundary_cap_faces(boundary, N)])
            m = trimesh.Trimesh(vv, ff, process=False)
        pieces.append(m)
        names.append(name)

    add("SMPL-X anatomy", d["canonical_vertices"], d["faces"], posed=True, solid=False)

    def section(y):
        seg = trimesh.intersections.mesh_plane(base, [0, 1, 0], [0, y, 0])
        p = seg.reshape(-1, 3)
        p = p[np.abs(p[:, 0]) < 0.235]
        poly = p[ConvexHull(p[:, [0, 2]]).vertices][:, [0, 2]]
        center = (poly.min(0) + poly.max(0)) / 2
        return poly, center

    theta = np.linspace(-np.pi, np.pi, 97)

    def radial(poly, center, t):
        direction = np.array([np.sin(t), np.cos(t)])
        a = poly - center
        e = np.roll(poly, -1, axis=0) - poly
        den = direction[0] * e[:, 1] - direction[1] * e[:, 0]
        tt = np.divide(
            a[:, 0] * e[:, 1] - a[:, 1] * e[:, 0],
            den,
            out=np.full(len(a), np.inf),
            where=np.abs(den) > 1e-9,
        )
        ss = np.divide(
            a[:, 0] * direction[1] - a[:, 1] * direction[0],
            den,
            out=np.full(len(a), np.inf),
            where=np.abs(den) > 1e-9,
        )
        good = (tt > 0) & (ss >= -1e-5) & (ss <= 1 + 1e-5)
        return tt[good].min() if good.any() else 0.15

    def rings(name, rings):
        rr = np.asarray(rings)
        nr, ns, _ = rr.shape
        ff = []
        for j in range(nr - 1):
            for i in range(ns - 1):
                a = j * ns + i
                b = a + 1
                c = a + ns
                dd = c + 1
                ff.extend([(a, b, c), (b, dd, c)])
        vv = rr.reshape(-1, 3)
        ff = np.array(ff)
        # Check outward orientation against each ring's local radial direction.
        m = trimesh.Trimesh(vv, ff, process=False)
        radialv = vv - np.repeat(rr.mean(1), ns, axis=0)
        if np.median(np.einsum("ij,ij->i", m.vertex_normals, radialv)) < 0:
            ff = ff[:, ::-1]
        add(name, vv, ff)

    waist = J[0, 1] + 0.10
    rr = []
    for y in np.linspace(waist, J[12, 1] - 0.075, 36):
        poly, c = section(y)
        rr.append(
            [
                [
                    c[0] + (radial(poly, c, t) + 0.010) * np.sin(t),
                    y,
                    c[1] + (radial(poly, c, t) + 0.010) * np.cos(t),
                ]
                for t in theta
            ]
        )
    last = np.asarray(rr[-1])
    neck = J[12]
    for a in np.linspace(0, 1, 10)[1:]:
        inner = np.stack(
            [
                neck[0] + 0.052 * np.sin(theta),
                np.full_like(theta, neck[1] - 0.012),
                neck[2] + 0.044 * np.cos(theta),
            ],
            1,
        )
        ring = last * (1 - a) + inner * a
        ring[:, 1] += 0.045 * np.sin(np.pi * a) * np.abs(np.sin(theta)) ** 4
        rr.append(ring)
    rings("Blouse", rr)
    for side, s, e, wj in [("left", 16, 18, 20), ("right", 17, 19, 21)]:
        start = J[s]
        end = J[e] + (J[wj] - J[e]) * 0.13
        axis = end - start
        axis /= np.linalg.norm(axis)
        up = np.array([0.0, 1.0, 0.0])
        up -= axis * np.dot(up, axis)
        up /= np.linalg.norm(up)
        front = np.cross(axis, up)
        rr = []
        for t in np.linspace(0, 1, 24):
            center = start * (1 - t) + end * t
            r = 0.060 * (1 - t) + 0.046 * t + 0.006 * np.sin(np.pi * t)
            rr.append(
                [center + r * (np.cos(a) * up + np.sin(a) * front) for a in theta]
            )
        rings(side + " sleeve", rr)
        rr = []
        for t in np.linspace(0, 1, 5):
            c = end - axis * 0.027 * (1 - t)
            rr.append([c + 0.050 * (np.cos(a) * up + np.sin(a) * front) for a in theta])
        rings(side + " cuff", rr)
    rr = []
    for t in np.linspace(0, 1, 55):
        y = waist - 0.405 * t
        poly, c = section(y)
        row = []
        for a in theta:
            r = radial(poly, c, a) + 0.013
            r += (
                0.0022
                * np.sin(43 * t + 3 * np.sin(a))
                * max(0, np.cos(a)) ** 4
                * np.sin(np.pi * t)
            )
            row.append(
                [
                    c[0] + r * np.sin(a),
                    y + 0.040 * np.sin(a) * t**3,
                    c[1] + r * np.cos(a),
                ]
            )
        rr.append(row)
    rings("Asymmetric wrap skirt", rr)
    # Hair is explicitly modeled geometry, not part of the bounded SMPL-X+D.
    # The forehead cap overlaps the skull by 12 mm; no hanging front strips.
    can = d["canonical_vertices"]
    canmesh = trimesh.Trimesh(can, d["faces"], process=False)
    hg = d["weights"][:, 15] > 0.4
    head = can[hg]
    hc = (head.min(0) + head.max(0)) / 2
    capf = d["faces"][scalp_cap_mask(can, d["faces"], hg, hc)]
    used = np.unique(capf)
    mapping = np.full(len(rest), -1)
    mapping[used] = np.arange(len(used))
    add(
        "Solid scalp cap",
        can[used] + canmesh.vertex_normals[used] * 0.006,
        mapping[capf],
        posed=True,
        thickness=0.020,
    )
    # Closed filled loft with a front wall embedded in head/neck/upper back.
    # Cross sections are 70-130 mm deep at body scale, not 3 mm shell walls.
    neck = d["canonical_joints"][12]
    rr = []
    angles = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    for t in np.linspace(0, 1, 55):
        y = (hc[1] + 0.080) * (1 - t) + (neck[1] - 0.215) * t
        band = can[
            (np.abs(can[:, 1] - y) < 0.020) & (np.abs(can[:, 0] - hc[0]) < 0.055)
        ]
        # Facial topology is much denser than the occiput. A percentile can
        # incorrectly select front-face vertices; use the rear envelope.
        back = band[:, 2].min() if len(band) else hc[2] - 0.075
        center_z = back + 0.012
        radius_x = np.interp(
            t,
            [0, 0.12, 0.40, 0.78, 0.93, 1],
            [0.035, 0.085, 0.100, 0.132, 0.120, 0.075],
        )
        radius_z = np.interp(t, [0, 0.16, 0.60, 1], [0.035, 0.065, 0.050, 0.035])
        # The inner wall is always at least 20 mm inside the existing anatomy.
        center_z = back + 0.024 - radius_z
        rr.append(
            [
                [hc[0] + radius_x * np.sin(a), y, center_z + radius_z * np.cos(a)]
                for a in angles
            ]
        )
    rr = np.asarray(rr)
    nr, ns, _ = rr.shape
    vf = rr.reshape(-1, 3).tolist()
    hf = []
    for j in range(nr - 1):
        for i in range(ns):
            a = j * ns + i
            b = j * ns + (i + 1) % ns
            c = a + ns
            dd = b + ns
            hf.extend([(a, c, b), (b, c, dd)])
    for j, reverse in [(0, False), (nr - 1, True)]:
        center = len(vf)
        vf.append(rr[j].mean(0).tolist())
        for i in range(ns):
            tri = (center, j * ns + i, j * ns + (i + 1) % ns)
            hf.append(tri[::-1] if reverse else tri)
    hair = trimesh.Trimesh(vf, hf, process=True)
    hair.fix_normals()
    if not (hair.is_watertight and hair.volume > 0):
        raise RuntimeError(
            "Back-hair loft is not a closed positive-volume solid (watertight="
            f"{hair.is_watertight}, volume={hair.volume:.3e}); check head landmarks and fit."
        )
    add(
        "Solid back hair embedded in anatomy",
        hair.vertices,
        hair.faces,
        posed=True,
        solid=False,
    )
    result = trimesh.util.concatenate(pieces)
    ranges = []
    start = 0
    for name, part in zip(names, pieces):
        ranges.append({"name": name, "start": start, "count": len(part.faces)})
        start += len(part.faces)
    return result, ranges
