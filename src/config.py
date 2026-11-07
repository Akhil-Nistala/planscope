"""Typed configuration loaded from YAML (configs/inference.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "inference.yaml"


@dataclass
class RouterConfig:
    """Thresholds for the vector/raster page classifier."""

    min_words: int = 15
    min_valid_char_ratio: float = 0.85
    min_text_coverage: float = 0.0


@dataclass
class RenderConfig:
    """Rasterisation settings."""

    dpi: int = 300
    vector_dpi: int = 150
    max_pixels: int = 120_000_000


@dataclass
class ImageConfig:
    """Settings for plain image inputs."""

    default_dpi: int | None = None


@dataclass
class TilingConfig:
    """Sliding-window inference (see src/tiling.py)."""

    enabled: bool = True
    tile: int = 640               # tile side in pixels at detect_dpi (= training tile size)
    overlap: float = 0.25
    full_pass: bool = True        # also run once on the whole (downscaled) sheet, for regions larger than a tile
    full_imgsz: int = 1024
    merge_iou: float = 0.5        # same class, IoU >= this -> duplicate
    merge_ios: float = 0.6        # same class, intersection-over-smaller >= this -> pieces of one object


@dataclass
class DetectorConfig:
    """Layout detector selection and inference settings."""

    kind: str = "null"            # "null" | "yolo_obb"
    weights: str = "models/planscope_obb.pt"  # relative to the repo root unless absolute
    imgsz: int = 1024             # inference size when tiling is disabled
    conf: float = 0.25
    iou: float = 0.5              # NMS IoU inside the model
    detect_dpi: int | None = 150  # pages are resampled to this DPI before detection (training scale)
    device: str = "cpu"
    max_det: int = 1000
    tiling: TilingConfig = field(default_factory=TilingConfig)


@dataclass
class TablesConfig:
    """Table structure recognition on detected schedule tables (src/table.py)."""

    enabled: bool = True


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    seed: int = 42
    router: RouterConfig = field(default_factory=RouterConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    images: ImageConfig = field(default_factory=ImageConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tables: TablesConfig = field(default_factory=TablesConfig)


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Recursively build a dataclass from a dict, rejecting unknown keys."""
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    for key, value in (data or {}).items():
        if key not in known:
            raise KeyError(f"Unknown config key '{key}' for {cls.__name__}")
        default = known[key].default_factory() if callable(known[key].default_factory) else None  # type: ignore[misc]
        if is_dataclass(default) and isinstance(value, dict):
            kwargs[key] = _build(type(default), value)
        else:
            kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """Load a :class:`PipelineConfig` from YAML; falls back to defaults if ``path`` is None and no default file exists."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        if path:
            raise FileNotFoundError(cfg_path)
        return PipelineConfig()
    with cfg_path.open("r", encoding="utf-8") as fh:
        return _build(PipelineConfig, yaml.safe_load(fh) or {})
