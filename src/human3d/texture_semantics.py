"""Conservative local label masks; no learned semantic model is claimed."""

import cv2, numpy as np
from scipy.ndimage import distance_transform_edt

FACE_MP51 = [
    70,
    63,
    105,
    66,
    107,
    336,
    296,
    334,
    293,
    300,
    168,
    6,
    197,
    1,
    98,
    97,
    2,
    326,
    327,
    33,
    160,
    158,
    133,
    153,
    144,
    362,
    385,
    387,
    263,
    373,
    380,
    61,
    40,
    37,
    0,
    267,
    270,
    291,
    321,
    314,
    17,
    84,
    91,
    78,
    81,
    13,
    311,
    308,
    402,
    14,
    178,
]
OVAL = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]


def masks_and_clean(image, mask, oldlabels):
    """Only eroded class-consistent pixels survive as measured evidence.

    Invalid pixels are filled before projection, so a triangle crossing an
    occlusion boundary cannot pull hair, skin, or sleeves into skirt colour.
    """
    h, w = image.shape[:2]
    yy, xx = np.indices((h, w))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(float)
    b, g, r = image.astype(float).transpose(2, 0, 1)
    fg = cv2.erode(mask.astype("uint8"), np.ones((5, 5), "uint8")) > 0
    white = (
        (oldlabels == 2)
        & (np.abs(lab[:, :, 1] - 128) < 15)
        & (np.abs(lab[:, :, 2] - 128) < 19)
        & (lab[:, :, 0] > 155)
    )
    blue = (
        (oldlabels == 3)
        & (b - r > 17)
        & (g - r > 3)
        & (yy > h * 0.28)
        & (yy < h * 0.64)
    )
    valid = {}
    clean = {}
    for label, selected in [(2, white), (3, blue)]:
        selected = (
            cv2.erode((selected & fg).astype("uint8"), np.ones((5, 5), "uint8")) > 0
        )
        if selected.sum() < 10:
            raise RuntimeError(
                "Too few conservative source pixels for label " + str(label)
            )
        valid[label] = selected
        dist, ij = distance_transform_edt(~selected, return_indices=True)
        fill = image[ij[0], ij[1]].astype(float)
        tone = np.median(image[selected], axis=0)
        # Outside measured support, use a smooth class-colour fill. No raw
        # source image fallback is allowed, even if all cameras are occluded.
        blend = np.clip((dist - 6) / 16, 0, 1)[:, :, None]
        fill = fill * (1 - blend) + tone * blend
        if label == 2:
            # Neutral opacity: conservative white removes skin tint in lace.
            grey = fill.mean(2, keepdims=True)
            fill = 0.78 * fill + 0.22 * grey
        clean[label] = fill.clip(0, 255).astype("uint8")
    return valid, clean


def skin_tone(image, labels):
    yy = np.indices(labels.shape)[0]
    h = labels.shape[0]
    good = (labels == 1) & (yy > h * 0.59) & (yy < h * 0.85)
    good = cv2.erode(good.astype("uint8"), np.ones((5, 5), "uint8")) > 0
    pixels = image[good]
    if len(pixels) < 10:
        raise RuntimeError("No reliable visible leg skin for colour matching")
    return np.median(pixels, axis=0)


def neutral_face(image, landmarks, tone, surface_support=None):
    """Single original-photo face, cheek-matched, with a feathered outline."""
    poly = cv2.convexHull(landmarks[OVAL].astype("float32"))
    mask = np.zeros(image.shape[:2], "uint8")
    cv2.fillConvexPoly(mask, np.rint(poly).astype(int), 1)
    if surface_support is not None:
        mask &= surface_support
    cheeks = np.zeros_like(mask)
    for indices in (
        [50, 101, 118, 119, 120, 100, 205, 206],
        [280, 330, 347, 348, 349, 329, 425, 426],
    ):
        cv2.fillConvexPoly(
            cheeks, cv2.convexHull(np.rint(landmarks[indices]).astype(int)), 1
        )
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(float)
    target = cv2.cvtColor(np.uint8([[tone]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    observed = np.median(lab[cheeks > 0], axis=0)
    delta = np.clip(target - observed, [-16, -7, -7], [16, 7, 7])
    lab += delta
    corrected = cv2.cvtColor(
        lab.clip(0, 255).astype("uint8"), cv2.COLOR_LAB2BGR
    ).astype(float)
    alpha = np.clip(distance_transform_edt(mask) / 15, 0, 1)[:, :, None]
    output = corrected * alpha + np.asarray(tone)[None, None, :] * (1 - alpha)
    return output.clip(0, 255).astype("uint8")
