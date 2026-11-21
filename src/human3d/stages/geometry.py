"""Anatomy-anchored layered reconstruction and multi-view texture projection."""

from human3d.settings import (
    work_dir,
    model_path,
    input_file,
    load_settings,
)
import time, json
import numpy as np, cv2
from scipy.spatial import cKDTree, Delaunay
from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_fill_holes
from human3d.core import project, normals, rasterize
from human3d.geom2d import points_in_convex_polygon
from human3d.layered import construct
from human3d.texture_semantics import (
    masks_and_clean,
    skin_tone,
    neutral_face,
    FACE_MP51,
    OVAL,
)

ROOT = work_dir()
OUT = ROOT / "surface"
OUT.mkdir(exist_ok=True)
d = np.load(ROOT / "fit/fit_result.npz")
views = json.loads((ROOT / "prepared/views.json").read_text())
vcan = d["canonical_vertices"]
faces = d["faces"]
weights = d["weights"]
cams = d["cameras"]
Tc = np.einsum("vj,jab->vab", weights, d["canonical_skinning"])
Tv = np.einsum("vj,ijab->ivab", weights, d["view_skinning"])
warp = Tv @ np.linalg.inv(Tc)[None]
tree = cKDTree(vcan)
start = time.time()
mesh, ranges = construct(d)
verts = np.array(mesh.vertices, dtype="float32")
ff = np.array(mesh.faces, dtype="int32")
(OUT / "layer_ranges.json").write_text(json.dumps(ranges, indent=2), encoding="utf8")
keep = mesh.split(only_watertight=False)
masks = []
images = []
label_images = []
valid_masks = []
clean_images = []
for view in views:
    name = view["name"]
    masks.append(cv2.imread(str(ROOT / "prepared" / (name + "_mask.png")), 0) > 0)
    images.append(cv2.imread(str(ROOT / "prepared" / (name + ".png"))))
    label_images.append(cv2.imread(str(ROOT / "prepared" / (name + "_labels.png")), 0))
    valid, clean = masks_and_clean(images[-1], masks[-1], label_images[-1])
    valid_masks.append(valid)
    clean_images.append(clean)
face_labels = np.ones(len(ff), dtype="uint8")
vertex_labels = np.ones(len(verts), dtype="uint8")
for part in ranges:
    sl = slice(part["start"], part["start"] + part["count"])
    name = part["name"].lower()
    label = (
        4
        if "hair" in name or "scalp" in name
        else (3 if "skirt" in name else (1 if "anatomy" in name else 2))
    )
    face_labels[sl] = label
    vertex_labels[np.unique(ff[sl])] = label
# A solid-field checkpoint is generated only for the separate print version.
lo = verts.min(0) - [0.075, 0.04, 0.08]
hi = verts.max(0) + [0.075, 0.05, 0.08]
shape = np.array([160, 430, 128])
spacing = (hi - lo) / (shape - 1)
axes = [np.linspace(lo[k], hi[k], shape[k], dtype="float32") for k in range(3)]
uvprint = np.stack(
    [(verts[:, 0] - lo[0]) / spacing[0], (verts[:, 1] - lo[1]) / spacing[1]], 1
).astype("float32")
ids, _, zmax = rasterize(
    uvprint, verts[:, 2].astype("float32"), ff, int(shape[1]), int(shape[0])
)
_, _, znegative = rasterize(
    uvprint, -verts[:, 2].astype("float32"), ff, int(shape[1]), int(shape[0])
)
inside = (
    (axes[2][None, None, :] <= zmax.T[:, :, None])
    & (axes[2][None, None, :] >= -znegative.T[:, :, None])
    & (ids.T[:, :, None] >= 0)
)
inside = binary_fill_holes(inside)
field = distance_transform_edt(~inside, sampling=spacing) - distance_transform_edt(
    inside, sampling=spacing
)
field = gaussian_filter(field, 0.5).astype("float32")
np.savez_compressed(
    OUT / "surface_field.npz", field=field, lo=lo, spacing=spacing, shape=shape
)
_, nn = tree.query(verts, workers=4)
W = weights[nn].copy()
# Keep the skirt envelope attached primarily to pelvis, not independently to legs.
pelvis = d["canonical_joints"][0]
yy = verts[:, 1] - pelvis[1]
skirt = (yy < -0.12) & (yy > -0.48) & (np.abs(verts[:, 0] - pelvis[0]) < 0.24)
W[skirt] *= 0.15
W[skirt, 0] += 0.85
W /= W.sum(1, keepdims=True)
vnorm = normals(verts, ff)
sources = []
source_meta = []
uvs = []
zbufs = []
scores = []
projected = []
viewverts = []
source_samples = []
for i, view in enumerate(views):
    T = warp[i, nn]
    q = np.einsum("vab,vb->va", T[:, :3, :3], verts) + T[:, :3, 3]
    uv, z = project(q, cams[i, 0], np.exp(cams[i, 1]), cams[i, 2], cams[i, 3])
    h, w = images[i].shape[:2]
    ids, bc, zb = rasterize(uv, z, ff, h, w)
    n = np.einsum("vab,vb->va", T[:, :3, :3], vnorm)
    yaw = cams[i, 0]
    cos = n[:, 2] * np.cos(yaw) - n[:, 0] * np.sin(yaw)
    px = np.clip(np.rint(uv[:, 0]).astype(int), 0, w - 1)
    py = np.clip(np.rint(uv[:, 1]).astype(int), 0, h - 1)
    vis = (z >= zb[py, px] - 0.018) & (masks[i][py, px])
    inside = (uv[:, 0] >= 1) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 1) & (uv[:, 1] < h - 1)
    edge = distance_transform_edt(masks[i])[py, px]
    projected.append(uv)
    zbufs.append(zb)
    viewverts.append(q)
    # Test vertices, edge midpoints and centroid; use the full layered mesh
    # z-buffer, including hair. Pixel labels additionally protect interiors.
    sample_bary = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0.5, 0.5, 0],
            [0, 0.5, 0.5],
            [0.5, 0, 0.5],
            [1 / 3, 1 / 3, 1 / 3],
        ]
    )
    samples = np.einsum("sk,fkc->fsc", sample_bary, uv[ff])
    sz = np.einsum("sk,fk->fs", sample_bary, z[ff])
    sx = np.clip(np.floor(samples[:, :, 0]).astype(int), 0, w - 1)
    sy = np.clip(np.floor(samples[:, :, 1]).astype(int), 0, h - 1)
    in_image = (
        (samples[:, :, 0] >= 2)
        & (samples[:, :, 0] < w - 2)
        & (samples[:, :, 1] >= 2)
        & (samples[:, :, 1] < h - 2)
    )
    visible = (sz >= zb[sy, sx] - 0.008) & in_image
    facing = np.maximum(cos[ff].mean(1), 0) ** 4
    for label in (2, 3):
        semantic = valid_masks[i][label][sy, sx]
        agreement = (semantic & visible).mean(1)
        # Invalid camera evidence scores zero. Any fallback still reads a
        # class-cleaned source, never the occluding raw pixels.
        score = np.where(face_labels == label, agreement**3 * facing, -1.0)
        scores.append(score)
        source = OUT / (view["name"] + "_class" + str(label) + "_source.png")
        cv2.imwrite(str(source), clean_images[i][label])
        sources.append(str(source))
        source_meta.append({"view": view["name"], "label": label, "eroded_pixels": 2})
        source_samples.append(agreement)
face_scores = np.array(scores)
face_material = np.argmax(face_scores, axis=0)
unseen = face_scores.max(0) < 1e-5
loopuv = np.empty((len(ff), 3, 2), "float32")
for k in range(8):
    i = k // 2
    use = face_material == k
    h, w = images[i].shape[:2]
    uv = projected[i][ff[use]]
    loopuv[use, :, 0] = np.clip(uv[:, :, 0] / w, 0, 1)
    loopuv[use, :, 1] = np.clip(1 - uv[:, :, 1] / h, 0, 1)

# Original-photo facial appearance via an explicit landmark warp, not a claim
# of recovered identity geometry. The real photograph is not a hold-out here.
face_count = 0
try:
    original = cv2.imread(str(input_file("photo")))
    ref = images[0][:145]
    cache = OUT / "face_landmarks.json"
    if cache.exists():
        cached = json.loads(cache.read_text())
        a = np.array(cached["reference"])
        b = np.array(cached["original"])
    else:
        import mediapipe as mp

        detector = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.4,
        )

        def landmarks(im, box):
            x0, y0, x1, y1 = box
            crop = im[y0:y1, x0:x1]
            res = detector.process(
                cv2.cvtColor(cv2.resize(crop, None, fx=3, fy=3), cv2.COLOR_BGR2RGB)
            )
            if not res.multi_face_landmarks:
                raise RuntimeError("Face landmark detection failed")
            return np.array(
                [
                    [p.x * crop.shape[1] + x0, p.y * crop.shape[0] + y0]
                    for p in res.multi_face_landmarks[0].landmark
                ]
            )

        a = landmarks(ref, load_settings().face_reference_crop)
        b = landmarks(original, load_settings().face_photo_crop)
        detector.close()
    # Align actual SMPL-X facial landmarks to the single real-photo face.
    # The old reference-to-photo warp ignored the model's feature locations.
    raw = np.load(model_path(), allow_pickle=True)
    body_uv, _ = project(
        d["view_vertices"][0], cams[0, 0], np.exp(cams[0, 1]), cams[0, 2], cams[0, 3]
    )
    model51 = np.einsum(
        "ni,nij->nj", raw["lmk_bary_coords"], body_uv[faces[raw["lmk_faces_idx"]]]
    )
    affine, _ = cv2.estimateAffine2D(a[FACE_MP51], model51, method=cv2.LMEDS)
    border = np.c_[a[OVAL], np.ones(len(OVAL))] @ affine.T
    anchors = np.concatenate([model51, border])
    targets = np.concatenate([b[FACE_MP51], b[OVAL]])
    tri = Delaunay(anchors)
    points = projected[0][ff].reshape(-1, 2)
    simp = tri.find_simplex(points)
    safe = np.maximum(simp, 0)
    T = tri.transform[safe]
    bary = np.einsum("nij,nj->ni", T[:, :2], points - T[:, 2])
    bary = np.concatenate([bary, 1 - bary.sum(1, keepdims=True)], 1)
    mapped = np.einsum("ni,nij->nj", bary, targets[tri.simplices[safe]])
    mapped = mapped.reshape(-1, 3, 2)
    valid = (simp.reshape(-1, 3) >= 0).all(1)
    front = vnorm[ff].mean(1)[:, 2] > 0.10
    head_weight = weights[:, [15, 22, 23, 24]].sum(1)
    valid &= (
        front
        & (np.arange(len(ff)) < ranges[0]["count"])
        & (head_weight[nn[ff]].mean(1) > 0.30)
    )
    # Keep the contour/eyebrows, excluding synthetic hair around the face.
    centers = projected[0][ff].mean(1)
    faceoval = cv2.convexHull(anchors.astype("float32"))
    valid &= points_in_convex_polygon(centers, faceoval)
    face_count = int(valid.sum())
    if face_count:
        h, w = original.shape[:2]
        loopuv[valid, :, 0] = np.clip(mapped[valid, :, 0] / w, 0, 1)
        loopuv[valid, :, 1] = np.clip(1 - mapped[valid, :, 1] / h, 0, 1)
        face_material[valid] = len(sources)
        source = OUT / "original_face_source.png"
        support = np.zeros(original.shape[:2], "uint8")
        for triangle in mapped[valid]:
            cv2.fillConvexPoly(support, np.rint(triangle).astype("int32"), 1)
        cv2.imwrite(
            str(source),
            neutral_face(original, b, skin_tone(images[0], label_images[0]), support),
        )
        sources.append(str(source))
        source_meta.append(
            {
                "view": "original_photo",
                "label": 1,
                "face_landmarks": 51,
                "blend_views": False,
            }
        )
        (OUT / "face_alignment.json").write_text(
            json.dumps(
                {
                    "model_landmarks": model51.tolist(),
                    "photo_landmarks": b[FACE_MP51].tolist(),
                    "mesh_space_reference_border": border.tolist(),
                    "mapping": "SMPL-X 51 static landmarks + affine border -> single original photo",
                    "identity_geometry_reconstructed": False,
                }
            ),
            encoding="utf8",
        )
    (OUT / "face_landmarks.json").write_text(
        json.dumps(
            {
                "reference": a.tolist(),
                "original": b.tolist(),
                "triangles_retextured": face_count,
            }
        ),
        encoding="utf8",
    )
except Exception as exc:
    (OUT / "face_projection_error.txt").write_text(str(exc), encoding="utf8")
    raise RuntimeError("Single-photo face alignment failed") from exc

np.savez_compressed(
    OUT / "character_surface.npz",
    vertices=verts,
    faces=ff,
    weights=W,
    loop_uv=loopuv,
    face_material=face_material,
    body_vertices=vcan,
    body_faces=faces,
    body_weights=weights,
    joints=d["canonical_joints"],
    parents=d["parents"],
    face_labels=face_labels,
    vertex_labels=vertex_labels,
)
(OUT / "texture_sources.json").write_text(
    json.dumps(sources, indent=2), encoding="utf8"
)
(OUT / "source_meta.json").write_text(
    json.dumps(source_meta, indent=2), encoding="utf8"
)
cloth = (face_labels == 2) | (face_labels == 3)
chosen = np.array(source_samples)[face_material.clip(0, 7), np.arange(len(ff))]
audit = {
    "labels_source": "Conservative colour/region heuristics, not learned Sapiens segmentation",
    "mask_erosion_px": 2,
    "depth_tolerance_m": 0.008,
    "triangle_visibility_samples": 7,
    "cloth_triangles": int(cloth.sum()),
    "occluded_or_unseen_cloth_triangles": int((cloth & unseen).sum()),
    "measured_source_agreement_mean_cloth": float(chosen[cloth].mean()),
    "all_cloth_sources_have_matching_class": bool(
        all(
            source_meta[int(face_material[k])]["label"] == int(face_labels[k])
            for k in np.where(cloth)[0]
        )
    ),
    "raw_unfiltered_camera_fallback": False,
    "single_original_photo_face": True,
}
(OUT / "texture_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf8")
diagnostics = []
for i, view in enumerate(views):
    q = viewverts[i]
    uv, z = project(q, cams[i, 0], np.exp(cams[i, 1]), cams[i, 2], cams[i, 3])
    h, w = images[i].shape[:2]
    ids, bc, zb = rasterize(uv, z, ff, h, w)
    mask = ids >= 0
    target = masks[i]
    overlay = images[i].copy()
    overlay[mask] = (overlay[mask] * 0.55 + np.array([70, 200, 30]) * 0.45).astype(
        "uint8"
    )
    cv2.imwrite(str(OUT / (view["name"] + "_surface_overlay.png")), overlay)
    diagnostics.append(
        {
            "view": view["name"],
            "iou": float((mask & target).sum() / (mask | target).sum()),
        }
    )
report = {
    "method": "Shared-shape SMPL-X+D, separate clothes, closed embedded back hair, class-filtered projective textures",
    "vertices": len(verts),
    "triangles": len(ff),
    "grid": shape.tolist(),
    "watertight": bool(mesh.is_watertight),
    "volume_m3": float(mesh.volume),
    "components": len(keep),
    "layers": ranges,
    "runtime_s": time.time() - start,
    "original_photo_face_triangles": face_count,
    "original_photo_is_holdout": False,
    "views": diagnostics,
    "normal_maps_used": False,
    "detail_limit": "Input turnaround subject approximately 150 pixels wide; no learned normal/detail synthesis was run.",
}
(OUT / "surface_report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
print("SURFACE_COMPLETE", json.dumps(report), flush=True)
