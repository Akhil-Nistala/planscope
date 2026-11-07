"""Typed configuration for the synthetic generator (configs/synth.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from src.config import REPO_ROOT, _build

from .layout import LAYOUTS, SHEET_SIZES, TITLE_BLOCK_STYLES

DEFAULT_SYNTH_CONFIG = REPO_ROOT / "configs" / "synth.yaml"
VOCAB_PATH = REPO_ROOT / "configs" / "vocab.yaml"
MODES = ("vector", "scan", "outlined")
SPLITS = ("train", "val", "test")


@dataclass
class ScanConfig:
    """Print-and-scan degradation."""

    skew_deg: float = 1.0
    jpeg_quality: list[int] = field(default_factory=lambda: [45, 85])
    noise_sigma: list[float] = field(default_factory=lambda: [3.0, 12.0])
    blur_prob: float = 0.7


@dataclass
class SynthConfig:
    """Generator settings. Mappings of name -> weight are normalised when sampled."""

    seed: int = 1234
    n_sheets: int = 3000
    workers: int = 0
    splits: dict[str, float] = field(default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1})
    modes: dict[str, float] = field(default_factory=lambda: {"vector": 0.5, "scan": 0.3, "outlined": 0.2})
    dpi: dict[str, int] = field(default_factory=lambda: {"vector": 150, "scan": 200, "outlined": 200})
    langs: dict[str, float] = field(default_factory=lambda: {"ja": 0.7, "en": 0.3})
    sheet_sizes: dict[str, float] = field(default_factory=lambda: {"A3": 0.55, "A2": 0.25, "A1": 0.20})
    layouts: dict[str, float] = field(default_factory=lambda: {"plan_side": 0.5, "schedule_sheet": 0.25,
                                                               "plan_only": 0.25})
    title_block_styles: dict[str, float] = field(default_factory=lambda: {"box": 0.7, "strip": 0.3})
    text_size_pt: list[float] = field(default_factory=lambda: [5.5, 9.0])
    callout_angles: dict[str, float] = field(default_factory=lambda: {"horizontal": 0.7, "vertical": 0.15,
                                                                      "oblique": 0.15})
    oblique_range_deg: list[float] = field(default_factory=lambda: [10.0, 60.0])
    scan: ScanConfig = field(default_factory=ScanConfig)
    save_pdf: bool | list[str] = True  # True/False, or the splits to keep PDFs for (e.g. ["test"])

    def keeps_pdf(self, split: str) -> bool:
        """Whether the source PDF of a vector/outlined sheet in ``split`` is written."""
        return split in self.save_pdf if isinstance(self.save_pdf, list) else bool(self.save_pdf)

    def validate(self) -> None:
        """Reject unknown categories and unusable weights early (before spawning workers)."""
        allowed = {
            "splits": SPLITS, "modes": MODES, "langs": ("ja", "en"), "sheet_sizes": tuple(SHEET_SIZES),
            "layouts": LAYOUTS, "title_block_styles": TITLE_BLOCK_STYLES,
            "callout_angles": ("horizontal", "vertical", "oblique"),
        }
        for name, ok in allowed.items():
            weights: dict[str, float] = getattr(self, name)
            bad = set(weights) - set(ok)
            if bad:
                raise ValueError(f"{name}: unknown keys {sorted(bad)} (allowed: {list(ok)})")
            if any(w < 0 for w in weights.values()) or sum(weights.values()) <= 0:
                raise ValueError(f"{name}: weights must be >= 0 with a positive sum")
        missing = {m for m, w in self.modes.items() if w > 0} - set(self.dpi)
        if missing:
            raise ValueError(f"dpi: no DPI for modes {sorted(missing)}")
        if self.n_sheets < 1:
            raise ValueError("n_sheets must be >= 1")
        if isinstance(self.save_pdf, list) and set(self.save_pdf) - set(SPLITS):
            raise ValueError(f"save_pdf: unknown splits {sorted(set(self.save_pdf) - set(SPLITS))}")


def load_synth_config(path: str | Path | None = None) -> SynthConfig:
    """Load and validate a :class:`SynthConfig` (defaults if ``path`` is None and no default file)."""
    p = Path(path) if path else DEFAULT_SYNTH_CONFIG
    if not p.exists():
        if path:
            raise FileNotFoundError(p)
        cfg = SynthConfig()
    else:
        cfg = _build(SynthConfig, yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    cfg.validate()
    return cfg


def load_vocab(path: str | Path | None = None) -> dict[str, Any]:
    """Load the shared vocabulary."""
    return yaml.safe_load(Path(path or VOCAB_PATH).read_text(encoding="utf-8"))
