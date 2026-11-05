"""Layout detection interface.

Milestone 1 ships only the interface and a :class:`NullDetector` so the pipeline,
schema and demo work end to end. The YOLO11-OBB detector + tiled (SAHI-style)
inference for small callouts are added in Milestones 3-4 behind the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

CLASSES: tuple[str, ...] = ("title_block", "schedule_table", "legend", "notes_block", "text_callout")


@dataclass
class Detection:
    """One oriented layout region in page-image pixel coordinates."""

    cls: str
    conf: float
    polygon: list[tuple[float, float]]  # 4 corners, clockwise


class Detector(Protocol):
    """Anything that maps a page image to oriented layout regions."""

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect layout regions on an RGB uint8 image."""
        ...


class NullDetector:
    """Placeholder detector: returns no regions (used until a model is trained)."""

    def detect(self, image: np.ndarray) -> list[Detection]:  # noqa: ARG002
        """Return an empty list."""
        return []


def build_detector(kind: str) -> Detector:
    """Factory keyed by ``configs/inference.yaml::detector.kind``."""
    if kind == "null":
        return NullDetector()
    raise ValueError(f"Unknown detector kind '{kind}' (YOLO-OBB arrives in Milestone 3)")
