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
class DetectorConfig:
    """Layout detector selection (filled in from Milestone 3)."""

    kind: str = "null"


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    seed: int = 42
    router: RouterConfig = field(default_factory=RouterConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    images: ImageConfig = field(default_factory=ImageConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)


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
