"""Actual exported-mesh close-ups and print preview, not generated imagery."""

from human3d.settings import (
    work_dir,
    load_settings,
)
import math, cv2, numpy as np, trimesh
from PIL import Image, ImageOps, ImageDraw
from human3d.core import project, normals, rasterize

R = work_dir()
O = R / "delivery"
scene = trimesh.load(O / "character_multiview.glb", force="scene", process=False)
m = next(iter(scene.geometry.values()))
v = np.asarray(m.vertices)
f = np.asarray(m.faces)
uv = np.asarray(m.visual.uv)
atlas = cv2.imread(str(O / "character_albedo.png"))
vn = normals(v, f)


def render(name, v, f, center, scale, angle=0, uv=None, vn=None):
    if vn is None:
        vn = normals(v, f)
    xy, z = project(v - center, angle, scale, 400, 400)
    ids, bc, zb = rasterize(xy, z, f, 800, 800)
    ok = ids >= 0
    fi = ids[ok]
    col = np.full((ok.sum(), 3), [184, 188, 194], float)
    if uv is not None:
        u = np.einsum("ni,nij->nj", bc[ok], uv[f[fi]])
        x = np.clip((u[:, 0] * 4095).astype(int), 0, 4095)
        y = np.clip(((1 - u[:, 1]) * 4095).astype(int), 0, 4095)
        col = atlas[y, x].astype(float)
    n = np.einsum("ni,nij->nj", bc[ok], vn[f[fi]])
    ry = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    light = np.array([-0.25, 0.5, 0.85])
    light /= np.linalg.norm(light)
    amb = 0.78 if uv is not None else 0.35
    sh = amb + (1 - amb) * np.maximum((n @ ry.T) @ light, 0)
    image = np.full((800, 800, 3), [43, 36, 29], "uint8")
    image[ok] = (col * sh[:, None]).clip(0, 255).astype("uint8")
    cv2.imwrite(str(O / name), image)


center = np.array([0, v[:, 1].max() - 0.15, 0])
render("detail_face.png", v, f, center, 2350, uv=uv, vn=vn)
render(
    "detail_hair_back.png",
    v,
    f,
    center - [0, 0.09, 0],
    1500,
    angle=math.pi,
    uv=uv,
    vn=vn,
)
render(
    "detail_hair_clay.png",
    v,
    f,
    center - [0, 0.09, 0],
    1500,
    angle=math.pi - 0.5,
    vn=vn,
)
p = trimesh.load(O / "character_print_200mm.stl", force="mesh", process=False)
pv = np.asarray(p.vertices)[:, [0, 2, 1]].copy()
pv[:, 2] *= -1
render(
    "preview_print.png",
    pv,
    np.asarray(p.faces),
    (pv.min(0) + pv.max(0)) / 2,
    3.6,
    angle=-0.45,
)
names = ["front", "threequarter", "left", "back", "clay"]
board = Image.new("RGB", (1500, 500), "#ebeef1")
draw = ImageDraw.Draw(board)
for i, name in enumerate(names):
    im = ImageOps.contain(
        Image.open(O / ("preview_" + name + ".png")).convert("RGB"), (300, 460)
    )
    board.paste(im, (i * 300 + (300 - im.width) // 2, 30))
    draw.text((i * 300 + 12, 10), name, fill="#223344")
board.save(O / "preview_board.png")
board.resize((1200, 400)).save(O / "preview_check.jpg")
old = load_settings().baseline
comparison = Image.new("RGB", (1200, 800), "#ebeef1")
draw = ImageDraw.Draw(comparison)
for row, (folder, label) in enumerate([(old, "V1"), (O, "V2")]):
    if folder is None or not folder.exists():
        continue
    for col, name in enumerate(["front", "back", "clay"]):
        im = ImageOps.contain(
            Image.open(folder / ("preview_" + name + ".png")).convert("RGB"), (400, 365)
        )
        comparison.paste(im, (col * 400 + (400 - im.width) // 2, row * 400 + 30))
        draw.text((col * 400 + 12, row * 400 + 10), label + " " + name, fill="#223344")
if old is not None and old.exists():
    comparison.save(O / "comparison_v1_v2.png")
print("REVIEW_IMAGES_READY")
