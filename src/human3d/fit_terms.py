"""Pure, unit-testable pieces of the fitting loss (no model, no I/O)."""

import torch

# Geman-McClure scale in units of image heights (0.05 image-heights).
SIGMA2 = 0.0025


def keypoint_term(pred, gt, conf, heights, sigma2=SIGMA2):
    """Robust keypoint loss, resolution independent.

    pred, gt: (V, K, 2) pixel coordinates; conf: (V, K); heights: (V,) image heights in pixels.
    The error is normalised by each view's OWN height. (It used to be divided by a constant 534, the
    panel height of the original reference board, which silently changed the loss for any other
    layout.) Scaling coordinates and heights by the same factor leaves the value unchanged.
    """
    err = ((pred - gt) / heights[:, None, None]) ** 2
    return (err / (err + sigma2) * conf[:, :, None]).mean()


def view_weight_vector(views, key="geometry_weight", enabled=False, device=None):
    """Per-view weights from views.json. All ones unless enabled (keeps old numerics by default)."""
    weights = [float(view.get(key, 1.0)) if enabled else 1.0 for view in views]
    for name, value in zip((v.get("name", "?") for v in views), weights):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"{key} of view {name} must be in (0, 1], got {value}")
    return torch.tensor(weights, dtype=torch.float32, device=device)
