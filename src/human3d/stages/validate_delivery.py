from human3d.settings import (
    work_dir,
)
import json, struct, numpy as np, trimesh

ROOT = work_dir()
OUT = ROOT / "delivery"
b = (OUT / "character_multiview.glb").read_bytes()
magic, version, length = struct.unpack_from("<4sII", b)
n, t = struct.unpack_from("<II", b, 12)
doc = json.loads(b[20 : 20 + n])
assert magic == b"glTF" and version == 2 and length == len(b)
assert len(doc["skins"][0]["joints"]) == 55
assert all(
    "JOINTS_0" in p["attributes"] and "WEIGHTS_0" in p["attributes"]
    for m in doc["meshes"]
    for p in m["primitives"]
)
assert all("uri" not in x for x in doc.get("images", []))
regular = trimesh.load(OUT / "character_multiview.glb", force="scene", process=False)
assert len(regular.geometry) > 0
assert all(np.isfinite(m.vertices).all() for m in regular.geometry.values())
stl = trimesh.load(OUT / "character_print_200mm.stl", force="mesh", process=True)
assert stl.is_watertight
assert stl.is_winding_consistent
assert len(stl.split(only_watertight=False)) == 1
assert abs(stl.extents[2] - 200) < 0.01
report = {
    "glb_version": version,
    "glb_length_verified": True,
    "glb_embedded_images": len(doc.get("images", [])),
    "glb_joints": 55,
    "glb_roundtrip_geometry": len(regular.geometry),
    "stl_watertight": bool(stl.is_watertight),
    "stl_winding_consistent": bool(stl.is_winding_consistent),
    "stl_volume_mm3": float(stl.volume),
    "stl_height_mm": float(stl.extents[2]),
    "physical_print_tested": False,
    "minimum_feature_thickness_verified": False,
}
(OUT / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf8")
print("DELIVERY_VALIDATED", json.dumps(report))
