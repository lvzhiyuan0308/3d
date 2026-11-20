"""Independent NumPy SMPL-X LBS reference for numerical cross-checks.

Extracted from the earlier local single-photo prototype; no detector or export CLI.
"""

import numpy as np


def rodrigues(r):
    r = np.asarray(r, dtype=np.float64).reshape(-1, 3)
    th = np.linalg.norm(r, axis=1, keepdims=True) + 1e-12
    k = r / th
    K = np.zeros((len(r), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    t = th[:, :, None]
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


class SMPLX:
    def __init__(self, npz_path, n_betas=10):
        d = np.load(npz_path, allow_pickle=True)
        self.nb = n_betas
        self.v_template = d["v_template"].astype(np.float64)
        self.faces = d["f"].astype(np.int64)
        self.shapedirs = d["shapedirs"][:, :, :n_betas].astype(np.float64)
        self.posedirs = d["posedirs"].astype(np.float64)  # (V,3,486)
        self.J_reg = d["J_regressor"].astype(np.float64)
        self.weights = d["weights"].astype(np.float64)
        self.parents = d["kintree_table"][0].astype(np.int64)
        self.parents[0] = -1
        self.vt = d["vt"].astype(np.float64)
        self.ft = d["ft"].astype(np.int64)
        self.pose_mean = np.zeros(165)
        self.pose_mean[75:120] = d["hands_meanl"]
        self.pose_mean[120:165] = d["hands_meanr"]
        self.JT = self.J_reg @ self.v_template
        self.JD = np.einsum("jv,vcb->jcb", self.J_reg, self.shapedirs)
        self.top = int(np.argmax(self.v_template[:, 1]))
        self.bot = int(np.argmin(self.v_template[:, 1]))

    def rest_joints(self, betas):
        return self.JT + self.JD @ betas

    def height(self, betas):
        d = self.shapedirs[[self.top, self.bot], 1, :] @ betas
        return (self.v_template[self.top, 1] + d[0]) - (
            self.v_template[self.bot, 1] + d[1]
        )

    def fk(self, rot, J):
        rel = J.copy()
        rel[1:] -= J[self.parents[1:]]
        T = np.zeros((55, 4, 4))
        T[:, :3, :3], T[:, :3, 3], T[:, 3, 3] = rot, rel, 1.0
        G = np.empty_like(T)
        G[0] = T[0]
        for i in range(1, 55):
            G[i] = G[self.parents[i]] @ T[i]
        return G

    def skin_matrices(self, G, J):
        A = G.copy()
        A[:, :3, 3] -= np.einsum("jab,jb->ja", G[:, :3, :3], J)
        return A

    def pose_to_G(self, betas, pose165):
        J = self.rest_joints(betas)
        rot = rodrigues(pose165 + self.pose_mean)
        return J, rot, self.fk(rot, J)

    def posed_vertex_subset(self, vids, betas, G, J):
        """Evaluate selected shaped vertices without pose blend shapes."""
        v = self.v_template[vids] + self.shapedirs[vids] @ betas
        A = self.skin_matrices(G, J).reshape(55, 16)
        T = (self.weights[vids] @ A).reshape(-1, 4, 4)
        return np.einsum("vab,vb->va", T[:, :3, :3], v) + T[:, :3, 3]

    def forward(self, betas, pose165):
        v_shaped = self.v_template + self.shapedirs @ betas
        J = self.J_reg @ v_shaped
        rot = rodrigues(pose165 + self.pose_mean)
        feat = (rot[1:] - np.eye(3)).reshape(-1)
        v_posed = v_shaped + np.tensordot(self.posedirs, feat, axes=([2], [0]))
        G = self.fk(rot, J)
        A = self.skin_matrices(G, J)
        T = (self.weights @ A.reshape(55, 16)).reshape(-1, 4, 4)
        verts = np.einsum("vab,vb->va", T[:, :3, :3], v_posed) + T[:, :3, 3]
        return verts, G[:, :3, 3]
