"""Local exports without requiring a working Blender executable.

One projective UV atlas, GLB with a transferred 55-joint skeleton, OBJ/MTL,
solid STL, actual-mesh colour/clay renders, and geometric validation.
"""

from human3d.settings import (
    work_dir,
    load_settings,
)
from pathlib import Path
import json, struct, math
import numpy as np, cv2, trimesh
from PIL import Image
from skimage.measure import marching_cubes
from human3d.core import normals, project, rasterize
from human3d.texture_semantics import skin_tone

ROOT = work_dir()
OUT = ROOT / "delivery"
OUT.mkdir(exist_ok=True)
d = np.load(ROOT / "surface/character_surface.npz")
v = d["vertices"].astype("float64")
f = d["faces"].astype("int32")
floor = v[:, 1].min()
v[:, 1] -= floor
source_paths = json.loads((ROOT / "surface/texture_sources.json").read_text())
source_paths = [
    str(Path(p) if Path(p).is_file() else ROOT / "surface" / Path(p).name)
    for p in source_paths
]
sources = [cv2.imread(p) for p in source_paths]
atlas = np.full((4096, 4096, 3), 128, "uint8")
uv = d["loop_uv"].copy()
mi = d["face_material"].copy()
hair_index = len(sources)
xx = np.arange(512)[None, :] / 512
strands = 5 * np.sin(xx * math.tau * 67) + 2 * np.sin(xx * math.tau * 151)
hair = (
    np.broadcast_to(
        np.array([38, 51, 67])[None, None, :] + strands[:, :, None], (1024, 512, 3)
    )
    .clip(0, 255)
    .astype("uint8")
)
sources.append(hair)
labels = cv2.imread(str(ROOT / "prepared/front_labels.png"), 0)
yy = np.indices(labels.shape)[0]
front_original = cv2.imread(str(ROOT / "prepared/front.png"))
skin_colour = skin_tone(front_original, labels)
# A continuous, low-contrast skin chart across the entire body, including
# skirt-covered thighs. No material or colour boundary at the skirt hem.
sh, sw = 1024, 512
row = np.linspace(0, 1, sh)[:, None, None]
col = np.linspace(0, 1, sw)[None, :, None]
skin = (
    skin_colour[None, None, :]
    + 2.4 * np.sin(row * np.pi * 2)
    + 0.8 * np.sin(col * np.pi * 2)
)
skin = np.broadcast_to(skin, (sh, sw, 3)).clip(0, 255).astype("uint8")
skin_index = len(sources)
sources.append(skin)
# Opaque lace: visual relief only; no alpha and no projected skin pixels.
ssy, ssx = np.indices((1024, 512))
lace = ((np.cos(ssx * 0.21) * np.cos(ssy * 0.17)) > 0.50).astype(float)
opaque = np.array([211, 215, 219])[None, None, :] - lace[:, :, None] * 14
sleeve_index = len(sources)
sources.append(opaque.clip(0, 255).astype("uint8"))
ranges = json.loads((ROOT / "surface/layer_ranges.json").read_text())
for part in ranges:
    sl = slice(part["start"], part["start"] + part["count"])
    if "hair" in part["name"].lower() or "scalp" in part["name"].lower():
        mi[sl] = hair_index
        pos = v[f[sl]]
        head = d["joints"][15]
        uv[sl, :, 0] = (
            np.arctan2(pos[:, :, 0] - head[0], pos[:, :, 2] - head[2]) + math.pi
        ) / math.tau
        uv[sl, :, 1] = (pos[:, :, 1] - pos[:, :, 1].min()) / max(
            np.ptp(pos[:, :, 1]), 0.001
        )
    if "sleeve" in part["name"].lower() or "cuff" in part["name"].lower():
        mi[sl] = sleeve_index
        pos = v[f[sl]]
        uv[sl, :, 0] = (pos[:, :, 2] - pos[:, :, 2].min()) / max(
            np.ptp(pos[:, :, 2]), 0.001
        )
        uv[sl, :, 1] = (pos[:, :, 1] - pos[:, :, 1].min()) / max(
            np.ptp(pos[:, :, 1]), 0.001
        )
body_end = ranges[0]["count"]
face_index = next(i for i, p in enumerate(source_paths) if "original_face" in p)
skin_faces = (np.arange(len(f)) < body_end) & (mi != face_index)
mi[skin_faces] = skin_index
pos = v[f[skin_faces]]
uv[skin_faces, :, 0] = 0.5 + np.arctan2(pos[:, :, 0], pos[:, :, 2]) / (2 * np.pi)
uv[skin_faces, :, 1] = pos[:, :, 1] / max(np.ptp(v[:, 1]), 0.001)
cols = 4
rows = math.ceil(len(sources) / cols)
tw = 4096 // cols
th = 4096 // rows
pad = 8
slots = [((i % cols) * tw, (i // cols) * th, tw, th) for i in range(len(sources))]
for i, im in enumerate(sources):
    use = mi == i
    if i < len(source_paths) and "original_face" in source_paths[i]:
        # Retain original pixel detail efficiently, cropping unused real-photo background.
        oh, ow = im.shape[:2]
        x0, y0, x1, y1 = load_settings().face_photo_crop
        uv[use, :, 0] = (uv[use, :, 0] * ow - x0) / (x1 - x0)
        uv[use, :, 1] = 1 - ((1 - uv[use, :, 1]) * oh - y0) / (y1 - y0)
        im = im[y0:y1, x0:x1]
    x, y, w, h = slots[i]
    tile = cv2.resize(im, (w - 2 * pad, h - 2 * pad), interpolation=cv2.INTER_CUBIC)
    atlas[y : y + h, x : x + w] = cv2.copyMakeBorder(
        tile, pad, pad, pad, pad, cv2.BORDER_REPLICATE
    )
    uv[use, :, 0] = (x + pad + np.clip(uv[use, :, 0], 0, 1) * (w - 2 * pad - 1)) / 4096
    uv[use, :, 1] = (
        1 - (y + pad + (1 - np.clip(uv[use, :, 1], 0, 1)) * (h - 2 * pad - 1)) / 4096
    )
cv2.imwrite(str(OUT / "character_albedo.png"), atlas)
keys = np.concatenate(
    [f.reshape(-1, 1), np.rint(uv.reshape(-1, 2) * 1e6).astype("int32")], 1
)
uniq, inv = np.unique(keys, axis=0, return_inverse=True)
splitv = v[uniq[:, 0]]
splituv = uniq[:, 1:].astype(float) / 1e6
splitf = inv.reshape(-1, 3)
mesh = trimesh.Trimesh(splitv, splitf, process=False)
mat = trimesh.visual.material.PBRMaterial(
    baseColorTexture=Image.fromarray(atlas[:, :, ::-1]),
    roughnessFactor=0.7,
    metallicFactor=0.0,
    doubleSided=False,
)
mesh.visual = trimesh.visual.TextureVisuals(uv=splituv, material=mat)
glb = mesh.export(file_type="glb")

# Add real JOINTS_0/WEIGHTS_0 and inverse bind matrices to the exported GLB.
jl, jt = struct.unpack_from("<II", glb, 12)
doc = json.loads(glb[20 : 20 + jl])
off = 20 + jl
bl, bt = struct.unpack_from("<II", glb, off)
binary = bytearray(glb[off + 8 : off + 8 + bl])


def accessor(array, component, kind, target=None):
    while len(binary) % 4:
        binary.append(0)
    offset = len(binary)
    data = array.tobytes()
    binary.extend(data)
    view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
    if target:
        view["target"] = target
    doc.setdefault("bufferViews", []).append(view)
    item = {
        "bufferView": len(doc["bufferViews"]) - 1,
        "componentType": component,
        "count": len(array),
        "type": kind,
    }
    doc.setdefault("accessors", []).append(item)
    return len(doc["accessors"]) - 1


W = d["weights"][uniq[:, 0]]
jidx = np.argsort(W, axis=1)[:, -4:][:, ::-1]
ww = np.take_along_axis(W, jidx, axis=1)
ww /= ww.sum(1, keepdims=True)
ja = accessor(jidx.astype("<u2"), 5123, "VEC4", 34962)
wa = accessor(ww.astype("<f4"), 5126, "VEC4", 34962)
joint = d["joints"].copy()
joint[:, 1] -= floor
parents = d["parents"]
nodes = doc["nodes"]
first = len(nodes)
for i, p in enumerate(joint):
    tr = p if i == 0 else p - joint[int(parents[i])]
    nodes.append(
        {"name": "SMPLX_joint_%02d" % i, "translation": tr.tolist(), "children": []}
    )
for i in range(1, 55):
    nodes[first + int(parents[i])]["children"].append(first + i)
for node in nodes:
    if not node.get("children", True):
        node.pop("children", None)
root = doc["scenes"][doc.get("scene", 0)]["nodes"][0]
nodes[root].setdefault("children", []).append(first)
ib = np.tile(np.eye(4), (55, 1, 1))
ib[:, :3, 3] = -joint
iba = accessor(ib.transpose(0, 2, 1).reshape(55, 16).astype("<f4"), 5126, "MAT4")
doc["skins"] = [
    {
        "name": "Transferred SMPL-X weights; animation unvalidated",
        "joints": list(range(first, first + 55)),
        "skeleton": first,
        "inverseBindMatrices": iba,
    }
]
for node in nodes:
    if "mesh" in node:
        node["skin"] = 0
for m in doc["meshes"]:
    for p in m["primitives"]:
        p["attributes"].update({"JOINTS_0": ja, "WEIGHTS_0": wa})
doc["buffers"][0]["byteLength"] = len(binary)
js = json.dumps(doc, separators=(",", ":")).encode()
js += b" " * ((-len(js)) % 4)
binary += b"\0" * ((-len(binary)) % 4)
glb = (
    struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(js) + 8 + len(binary))
    + struct.pack("<II", len(js), 0x4E4F534A)
    + js
    + struct.pack("<II", len(binary), 0x004E4942)
    + binary
)
(OUT / "character_multiview.glb").write_bytes(glb)
with (OUT / "character_multiview.obj").open("w", encoding="ascii") as out:
    out.write("mtllib character_multiview.mtl\nusemtl reference_colour\ns 1\n")
    for p in v:
        out.write("v %.7f %.7f %.7f\n" % tuple(p))
    for p in uv.reshape(-1, 2):
        out.write("vt %.7f %.7f\n" % tuple(p))
    starts = {p["start"]: p["name"].replace(" ", "_") for p in ranges}
    for i, tri in enumerate(f):
        if i in starts:
            out.write("g " + starts[i] + "\n")
        out.write(
            "f "
            + " ".join(
                "%d/%d" % (int(vi) + 1, i * 3 + k + 1) for k, vi in enumerate(tri)
            )
            + "\n"
        )
(OUT / "character_multiview.mtl").write_text(
    "newmtl reference_colour\nKa 1 1 1\nKd 1 1 1\nmap_Kd character_albedo.png\n",
    encoding="ascii",
)

# Union an analytic oval base into the reconstructed field, then retain one
# connected solid. Units in STL are explicitly millimetres.
fd = np.load(ROOT / "surface/surface_field.npz")
field = fd["field"].copy()
lo = fd["lo"]
sp = fd["spacing"]
shape = np.array(field.shape)
axes = [lo[k] + np.arange(shape[k]) * sp[k] for k in range(3)]
bottom = float(d["vertices"][:, 1].min())
low = d["vertices"][d["vertices"][:, 1] < bottom + 0.11]
cx, cz = np.median(low[:, [0, 2]], axis=0)
rx = min(0.33, cx - axes[0][0] - 0.012, axes[0][-1] - cx - 0.012)
rz = min(0.23, cz - axes[2][0] - 0.012, axes[2][-1] - cz - 0.012)
radial = (
    np.sqrt(((axes[0][:, None] - cx) / rx) ** 2 + ((axes[2][None, :] - cz) / rz) ** 2)
    - 1
) * min(rx, rz)
base = np.maximum(
    radial[:, None, :], np.abs(axes[1][None, :, None] - (bottom - 0.005)) - 0.020
)
field = np.minimum(field, base)
# Reinforce only the print version with smooth ankle-to-foot capsules.
ankle_supports = []
for ankle, knee, foot in [(7, 4, 10), (8, 5, 11)]:
    start = d["joints"][ankle] + (d["joints"][knee] - d["joints"][ankle]) * 0.16
    end = d["joints"][foot].copy()
    end[1] = min(end[1], bottom + 0.022)
    delta = end - start
    den = float(np.dot(delta, delta))
    radius = 0.028
    X = axes[0][:, None, None] - start[0]
    Y = axes[1][None, :, None] - start[1]
    Z = axes[2][None, None, :] - start[2]
    t = np.clip((X * delta[0] + Y * delta[1] + Z * delta[2]) / den, 0, 1)
    capsule = (
        np.sqrt(
            (X - t * delta[0]) ** 2 + (Y - t * delta[1]) ** 2 + (Z - t * delta[2]) ** 2
        )
        - radius
    )
    field = np.minimum(field, capsule)
    ankle_supports.append(
        {"start": start.tolist(), "end": end.tolist(), "radius_body_m": radius}
    )
field[[0, -1]] = 1
field[:, [0, -1]] = 1
field[:, :, [0, -1]] = 1
pv, pf, _, _ = marching_cubes(field, 0, spacing=tuple(sp), allow_degenerate=False)
pv += lo
printmesh = trimesh.Trimesh(pv, pf, process=True)
printmesh.fix_normals()
parts = printmesh.split(only_watertight=False)
printmesh = max(parts, key=lambda p: len(p.faces))
pv = np.asarray(printmesh.vertices).copy()
print_origin = pv.min(0)
print_scale = 200 / np.ptp(pv[:, 1])
pv = (pv - print_origin) * print_scale
# STL follows Z-up convention for convenient slicer import.
pv = pv[:, [0, 2, 1]]
pv[:, 1] *= -1
printmesh = trimesh.Trimesh(pv, printmesh.faces[:, ::-1], process=True)
printmesh.fix_normals()
printmesh.export(OUT / "character_print_200mm.stl")
(OUT / "print_construction.json").write_text(
    json.dumps(
        {
            "origin_body_m": print_origin.tolist(),
            "scale_mm_per_body_m": float(print_scale),
            "ankle_capsules": ankle_supports,
            "nominal_ankle_support_diameter_mm": float(0.056 * print_scale),
            "base": "solid oval union",
            "hair": "closed back loft embedded in head and upper back; front strips removed",
            "method_limit": "z-envelope solidification can merge front/back cavities; not preservation of all garment interiors",
        },
        indent=2,
    ),
    encoding="utf8",
)

# Render the exported triangles and atlas, including a clay inspection view.
vn = normals(v, f)
center = (v.min(0) + v.max(0)) / 2
for name, angle in [
    ("front", 0),
    ("threequarter", -0.38),
    ("left", math.pi / 2),
    ("back", math.pi),
    ("clay", -0.38),
]:
    pp = v - center
    scale = 900 / np.ptp(v[:, 1])
    xy, z = project(pp, angle, scale, 350, 515)
    ids, bc, zb = rasterize(xy, z, f, 1030, 700)
    valid = ids >= 0
    fi = ids[valid]
    tu = np.einsum("ni,nij->nj", bc[valid], uv[fi])
    x = np.clip((tu[:, 0] * 4095).astype(int), 0, 4095)
    y = np.clip(((1 - tu[:, 1]) * 4095).astype(int), 0, 4095)
    col = atlas[y, x].astype(float)
    if name == "clay":
        col[:] = [184, 188, 194]
    nn = np.einsum("ni,nij->nj", bc[valid], vn[f[fi]])
    light = np.array([-0.25, 0.5, 0.85])
    light /= np.linalg.norm(light)
    # Rotate the light with the view to retain readable geometry.
    ry = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    shading = 0.78 + 0.22 * np.maximum((nn @ ry.T) @ light, 0)
    if name == "clay":
        shading = 0.35 + 0.65 * np.maximum((nn @ ry.T) @ light, 0)
    image = np.full((1030, 700, 3), [43, 36, 29], "uint8")
    image[valid] = (col * shading[:, None]).clip(0, 255).astype("uint8")
    cv2.imwrite(str(OUT / ("preview_" + name + ".png")), image)
report = {
    "surface_vertices": len(v),
    "surface_triangles": len(f),
    "uv_split_vertices": len(splitv),
    "atlas": [4096, 4096],
    "bones": 55,
    "animation_validated": False,
    "print_triangles": len(printmesh.faces),
    "procedural_hair_texture": True,
    "skin_tone_fill_from_reference": True,
    "semantic_filtering": True,
    "face_source": "one original photo aligned to SMPL-X 51 landmarks",
    "opaque_sleeves": True,
    "front_hair_strips_removed": True,
    "ankles_reinforced_print_only": True,
    "print_watertight": bool(printmesh.is_watertight),
    "print_components": len(printmesh.split(only_watertight=False)),
    "print_size_mm": printmesh.extents.tolist(),
    "normal_maps_used": False,
    "native_blend_generated": False,
    "export_method": "Local Python, projective atlas, GLB skin attributes and solid-field base union",
}
(OUT / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf8")
print("EXPORT_COMPLETE", json.dumps(report), flush=True)
