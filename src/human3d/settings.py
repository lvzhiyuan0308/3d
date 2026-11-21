"""Resolve all model and data paths relative to one TOML configuration."""

from dataclasses import dataclass
from pathlib import Path
import os
import tomllib

DEFAULT_VIEWS = (
    ("front", (12, 36, 223, 570), 0),
    ("left", (227, 36, 398, 570), 90),
    ("right", (402, 36, 579, 570), -90),
    ("back", (583, 36, 783, 570), 180),
)


@dataclass(frozen=True)
class Settings:
    config: Path
    work: Path
    model: Path
    photo: Path
    turnaround: Path
    baseline: Path | None
    device: str
    views: tuple
    face_photo_crop: tuple
    face_reference_crop: tuple
    # Subject and fitting options. Defaults reproduce the original behaviour.
    height_m: float = 1.695
    height_is_measured: bool = False
    fit_use_view_weights: bool = False
    fit_height_weight: float = 0.4


def load_settings(config=None):
    path = Path(
        config or os.environ.get("HUMAN3D_CONFIG", "config.local.toml")
    ).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration missing: {path}. Copy config.example.toml first."
        )
    with path.open("rb") as stream:
        data = tomllib.load(stream)

    def resolve(value):
        expanded = Path(os.path.expandvars(str(value))).expanduser()
        return (path.parent / expanded).resolve()

    paths = data["paths"]
    device = data.get("runtime", {}).get("device", "auto")
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("runtime.device must be auto, cpu or cuda")
    views = []
    for item in data.get("views", []):
        crop = tuple(item["crop"])
        if len(crop) != 4 or crop[2] <= crop[0] or crop[3] <= crop[1]:
            raise ValueError(
                "Each crop must be [left, top, right, bottom] with positive area"
            )
        views.append((item["name"], crop, item["yaw_deg"]))
    if views and [v[0] for v in views] != ["front", "left", "right", "back"]:
        raise ValueError("Views must be ordered front, left, right, back")
    baseline = paths.get("baseline_delivery")
    face = data.get("face", {})
    subject = data.get("subject", {})
    fit = data.get("fit", {})
    height = float(subject.get("height_m", 1.695))
    if not 1.0 <= height <= 2.3:
        raise ValueError("subject.height_m must be between 1.0 and 2.3 metres")
    height_weight = float(fit.get("height_weight", 0.4))
    if height_weight < 0:
        raise ValueError("fit.height_weight must be non-negative")
    return Settings(
        path,
        resolve(paths.get("work_dir", "work")),
        resolve(paths["smplx_model"]),
        resolve(paths["original_photo"]),
        resolve(paths["turnaround"]),
        resolve(baseline) if baseline else None,
        device,
        tuple(views) or DEFAULT_VIEWS,
        tuple(face.get("photo_crop", [290, 40, 480, 235])),
        tuple(face.get("reference_crop", [60, 0, 165, 115])),
        height,
        bool(subject.get("height_measured", False)),
        bool(fit.get("use_view_weights", False)),
        height_weight,
    )


def work_dir():
    return load_settings().work


def model_path():
    path = load_settings().model
    if not path.is_file():
        raise FileNotFoundError(
            f"SMPL-X weights missing: {path}. Set paths.smplx_model."
        )
    return path


def input_file(name):
    settings = load_settings()
    path = {"photo": settings.photo, "turnaround": settings.turnaround}[name]
    if not path.is_file():
        raise FileNotFoundError(f"Input missing: {path}")
    return path


def compute_device():
    import torch

    requested = load_settings().device
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "auto"
        else requested
    )
