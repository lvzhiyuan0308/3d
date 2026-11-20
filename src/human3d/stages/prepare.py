"""Prepare the actual reference package without inventing additional views."""

from human3d.settings import (
    work_dir,
    input_file,
    load_settings,
)
import json, cv2, numpy as np

ROOT = work_dir()
OUT = ROOT / "prepared"
OUT.mkdir(exist_ok=True)
image = cv2.imread(str(input_file("turnaround")))
# Manually reviewed panel bounds, excluding titles and dividing lines.
specs = load_settings().views
views = []
import mediapipe as mp

pose = mp.solutions.pose.Pose(
    static_image_mode=True, model_complexity=1, min_detection_confidence=0.35
)
mapping = {
    "nose": 0,
    "l_eye": 2,
    "r_eye": 5,
    "l_shoulder": 11,
    "r_shoulder": 12,
    "l_elbow": 13,
    "r_elbow": 14,
    "l_wrist": 15,
    "r_wrist": 16,
    "l_hip": 23,
    "r_hip": 24,
    "l_knee": 25,
    "r_knee": 26,
    "l_ankle": 27,
    "r_ankle": 28,
}
for name, box, yaw in specs:
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
        raise ValueError("Crop is outside turnaround image: " + name)
    im = image[y0:y1, x0:x1].copy()
    h, w = im.shape[:2]
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(float)
    bg = np.median(np.concatenate([lab[:, :8], lab[:, -8:]], axis=1), axis=1)[
        :, None, :
    ]
    diff = np.linalg.norm(lab - bg, axis=2)
    mask = np.where(diff > 15, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype("uint8")
    mask[diff > 34] = cv2.GC_FGD
    mask[:, :8] = mask[:, -8:] = cv2.GC_BGD
    mask[:2] = mask[-3:] = cv2.GC_BGD
    cv2.grabCut(
        im, mask, None, np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_MASK
    )
    mask = np.isin(mask, [1, 3]).astype("uint8")
    count, cc, stat, cen = cv2.connectedComponentsWithStats(mask)
    keep = [i for i in range(1, count) if stat[i, 4] > 25]
    mask = np.isin(cc, keep).astype("uint8")
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), "uint8"))
    result = pose.process(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    kp = {}
    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark
        kp = {
            n: [lm[i].x * w, lm[i].y * h, float(lm[i].visibility)]
            for n, i in mapping.items()
        }
    if not kp:
        raise RuntimeError("No pose keypoints detected in view: " + name)
    labels = np.zeros((h, w), "uint8")
    # Region priors are explicit heuristics, not claimed Sapiens predictions.
    yy, xx = np.indices((h, w))
    labels[mask > 0] = 1
    white = (
        (lab[:, :, 0] > 160)
        & (np.abs(lab[:, :, 1] - 128) < 16)
        & (np.abs(lab[:, :, 2] - 128) < 22)
    )
    blue = (
        (im[:, :, 0].astype(float) - im[:, :, 2] > 18)
        & (yy > h * 0.26)
        & (yy < h * 0.63)
    )
    hair = (yy < h * 0.33) & (lab[:, :, 0] < 125) & (mask > 0)
    labels[white & (mask > 0) & (yy < h * 0.40) & (yy > h * 0.14)] = 2
    labels[blue & (mask > 0)] = 3
    labels[hair] = 4
    labels[(yy > h * 0.89) & (mask > 0)] = 5
    cv2.imwrite(str(OUT / (name + ".png")), im)
    cv2.imwrite(str(OUT / (name + "_mask.png")), mask * 255)
    cv2.imwrite(str(OUT / (name + "_labels.png")), labels)
    over = im.copy()
    over[mask == 0] = (over[mask == 0] * 0.25).astype("uint8")
    for n, (x, y, c) in kp.items():
        cv2.circle(over, (round(x), round(y)), 3, (0, 255, 0), -1)
        cv2.putText(
            over,
            n,
            (round(x) + 3, round(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.23,
            (0, 0, 255),
            1,
        )
    cv2.imwrite(str(OUT / (name + "_check.png")), over)
    views.append(
        {
            "name": name,
            "yaw_deg": yaw,
            "crop": box,
            "size": [w, h],
            "keypoints": kp,
            "source": "generated_reference",
            "geometry_weight": 0.6,
            "texture_weight": 0.7,
        }
    )
pose.close()
(OUT / "views.json").write_text(json.dumps(views, indent=2), encoding="utf8")
panels = [
    cv2.resize(cv2.imread(str(OUT / (v["name"] + "_check.png"))), (230, 600))
    for v in views
]
cv2.imwrite(str(OUT / "input_review.jpg"), np.concatenate(panels, axis=1))
print(
    json.dumps([{"name": v["name"], "keypoints": len(v["keypoints"])} for v in views]),
    flush=True,
)
