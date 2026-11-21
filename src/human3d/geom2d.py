"""Small 2-D geometry helpers."""

import numpy as np


def points_in_convex_polygon(points, polygon, tol=0.0):
    """Vectorised inside-or-on-boundary test for a convex polygon (any orientation).

    Replaces a Python loop over cv2.pointPolygonTest(..., False) >= 0. points: (N, 2);
    polygon: (M, 2) or cv2.convexHull output (M, 1, 2). Returns a bool array (N,).
    """
    poly = np.asarray(polygon, dtype=float).reshape(-1, 2)
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(poly) < 3:
        return np.zeros(len(pts), dtype=bool)
    edge = np.roll(poly, -1, axis=0) - poly
    rel = pts[:, None, :] - poly[None]
    cross = edge[None, :, 0] * rel[:, :, 1] - edge[None, :, 1] * rel[:, :, 0]
    return (cross >= -tol).all(axis=1) | (cross <= tol).all(axis=1)
