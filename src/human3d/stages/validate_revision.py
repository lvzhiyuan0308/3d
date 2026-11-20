"""Targeted geometric and source-label checks for the reported defects."""

from human3d.settings import (
    work_dir,
)
import json, numpy as np, trimesh

R = work_dir()
O = R / "delivery"
d = np.load(R / "surface/character_surface.npz")
fit = np.load(R / "fit/fit_result.npz")
ranges = json.loads((R / "surface/layer_ranges.json").read_text())
construction = json.loads((O / "print_construction.json").read_text())
scale = construction["scale_mm_per_body_m"]
origin = np.array(construction["origin_body_m"])
assert not any("front lock" in p["name"].lower() for p in ranges)
back = next(p for p in ranges if "Solid back hair" in p["name"])
hair = trimesh.Trimesh(
    d["vertices"],
    d["faces"][back["start"] : back["start"] + back["count"]],
    process=True,
)
hair.remove_unreferenced_vertices()
hair.merge_vertices()
hair.fix_normals()
assert hair.is_watertight and hair.volume > 0
body = trimesh.Trimesh(d["body_vertices"], d["body_faces"], process=False)


def interval(mesh, height, x):
    seg = trimesh.intersections.mesh_plane(mesh, [0, 1, 0], [0, height, 0])
    values = []
    for a, b in seg:
        den = b[0] - a[0]
        if abs(den) < 1e-9:
            continue
        t = (x - a[0]) / den
        if 0 <= t <= 1:
            values.append(float(a[2] + t * (b[2] - a[2])))
    if len(values) < 2:
        raise ValueError("No closed cross-section at " + str((height, x)))
    return min(values), max(values)


neck = fit["canonical_joints"][12]
xc = np.median(hair.vertices[:, 0])
sections = []
for height in np.linspace(neck[1] - 0.18, neck[1] + 0.20, 7):
    hz = interval(hair, float(height), xc)
    bz = interval(body, float(height), xc)
    overlap = min(hz[1], bz[1]) - max(hz[0], bz[0])
    sections.append(
        {
            "height_body_m": float(height),
            "hair_depth_mm": float((hz[1] - hz[0]) * scale),
            "overlap_into_anatomy_mm": float(overlap * scale),
            "inner_wall_offset_from_body_back_mm": float((hz[1] - bz[0]) * scale),
        }
    )
    # Regression guard: dense facial vertices must not drive hair in front
    # of the cheeks/nose as a rear-envelope percentile did initially.
    assert hz[1] < bz[0] + 0.032
assert min(s["overlap_into_anatomy_mm"] for s in sections) > 0
assert min(s["hair_depth_mm"] for s in sections) > 5
stl = trimesh.load(O / "character_print_200mm.stl", force="mesh", process=True)
ankles = []
split_x = float((fit["canonical_joints"][[7, 8], 0].mean() - origin[0]) * scale)
for index in (7, 8):
    joint = (fit["canonical_joints"][index] - origin) * scale
    seg = trimesh.intersections.mesh_plane(stl, [0, 0, 1], [0, 0, joint[1]])
    mid = seg.mean(1)
    same = (mid[:, 0] > split_x) == (joint[0] > split_x)
    p = seg[same].reshape(-1, 3)
    size = np.ptp(p, axis=0)
    ankles.append(
        {
            "joint": index,
            "height_mm": float(joint[1]),
            "width_x_mm": float(size[0]),
            "depth_y_mm": float(size[1]),
        }
    )
assert min(a["width_x_mm"] for a in ankles) > 4.5
audit = json.loads((R / "surface/texture_audit.json").read_text())
assert (
    audit["all_cloth_sources_have_matching_class"]
    and not audit["raw_unfiltered_camera_fallback"]
)
meta = json.loads((R / "surface/source_meta.json").read_text())
face_index = next(i for i, p in enumerate(meta) if p["view"] == "original_photo")
assert (d["face_material"] == face_index).sum() > 500
assert np.isfinite(d["loop_uv"]).all()
report = {
    "front_strips_removed": True,
    "back_hair_closed_solid": bool(hair.is_watertight),
    "back_hair_cross_sections": sections,
    "measured_ankle_cross_sections": ankles,
    "class_filtered_sources": True,
    "single_photo_face": True,
    "global_minimum_wall_thickness_verified": False,
    "physical_print_tested": False,
    "scope": "Selected hair centre sections and ankle horizontal sections, not global minimum feature analysis",
}
(O / "revision_validation.json").write_text(
    json.dumps(report, indent=2), encoding="utf8"
)
print("REVISION_VALIDATED", json.dumps(report))
