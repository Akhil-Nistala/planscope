"""Layout detection.

:class:`YoloObbDetector` wraps an Ultralytics YOLO11-OBB model trained on the synthetic sheets
(``training/``). Pages are resampled to ``detect_dpi`` (the training scale), then detected either in
one pass or on overlapping tiles plus an optional full-sheet pass, with pieces merged back together
(``src/tiling.py``). :class:`NullDetector` returns nothing and keeps the pipeline usable without a model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import cv2
import numpy as np

from .geometry import canonical_corners
from .tiling import OBB, merge_obbs, tile_grid

if TYPE_CHECKING:
    from .config import DetectorConfig

log = logging.getLogger(__name__)

CLASSES: tuple[str, ...] = ("title_block", "schedule_table", "legend", "notes_block", "text_callout")
MIN_SHEET_INCHES = 11.0  # long side of A4 is 11.7 in; drawings are never smaller


@dataclass
class Detection:
    """One oriented layout region in page-image pixel coordinates."""

    cls: str
    conf: float
    polygon: list[tuple[float, float]]  # 4 corners, clockwise


class Detector(Protocol):
    """Anything that maps a page image to oriented layout regions."""

    def detect(self, image: np.ndarray, dpi: int | None = None) -> list[Detection]:
        """Detect layout regions on an RGB uint8 image rendered at ``dpi`` (None if unknown)."""
        ...


class NullDetector:
    """Placeholder detector: returns no regions (used when no model is configured)."""

    def detect(self, image: np.ndarray, dpi: int | None = None) -> list[Detection]:  # noqa: ARG002
        """Return an empty list."""
        return []


class YoloObbDetector:
    """Ultralytics OBB model with DPI normalisation and optional tiled inference."""

    def __init__(self, cfg: "DetectorConfig", weights: str | Path | None = None):
        from ultralytics import YOLO  # heavy import: only when a model is actually used

        from .config import REPO_ROOT

        w = Path(weights or cfg.weights)
        w = w if w.is_absolute() else REPO_ROOT / w
        if not w.exists():
            raise FileNotFoundError(f"detector weights not found: {w} (train with training/train.py)")
        self.cfg = cfg
        self.model = YOLO(str(w), task="obb")
        names = self.model.names
        unknown = set(names.values()) - set(CLASSES)
        if unknown:
            raise ValueError(f"model has classes outside the schema: {sorted(unknown)}")
        self.names = {int(k): v for k, v in names.items()}

    def _predict(self, img: np.ndarray, imgsz: int) -> list[OBB]:
        res = self.model.predict(np.ascontiguousarray(img[..., ::-1]), imgsz=imgsz, conf=self.cfg.conf,
                                 iou=self.cfg.iou, device=self.cfg.device, max_det=self.cfg.max_det,
                                 verbose=False)[0]
        if res.obb is None or len(res.obb) == 0:
            return []
        polys = res.obb.xyxyxyxy.cpu().numpy().astype(np.float64)
        return [OBB(int(c), float(s), p) for c, s, p in
                zip(res.obb.cls.cpu().numpy(), res.obb.conf.cpu().numpy(), polys)]

    def detect_obbs(self, image: np.ndarray) -> list[OBB]:
        """Detect on an image already at ``detect_dpi`` (pixel coordinates of that image)."""
        t = self.cfg.tiling
        h, w = image.shape[:2]
        if not t.enabled:
            return merge_obbs(self._predict(image, self.cfg.imgsz), t.merge_iou, t.merge_ios)
        dets: list[OBB] = []
        for x0, y0, x1, y1 in tile_grid(w, h, t.tile, t.overlap):
            for d in self._predict(image[y0:y1, x0:x1], t.tile):
                d.poly = d.poly + [x0, y0]
                dets.append(d)
        if t.full_pass and max(h, w) > t.tile:
            dets += self._predict(image, t.full_imgsz)
        return merge_obbs(dets, t.merge_iou, t.merge_ios)

    def detect(self, image: np.ndarray, dpi: int | None = None) -> list[Detection]:
        """Detect layout regions; polygons are returned in ``image`` pixel coordinates."""
        scale = 1.0
        # Distrust DPI metadata implying a sheet smaller than A4 (e.g. a microfilm frame tagged 600 dpi):
        # rescaling by it would shrink a full drawing to a thumbnail.
        if dpi and max(image.shape[:2]) / dpi < MIN_SHEET_INCHES:
            log.warning("ignoring dpi=%s: page would be %.1f in long", dpi, max(image.shape[:2]) / dpi)
            dpi = None
        if dpi and self.cfg.detect_dpi:
            scale = float(np.clip(self.cfg.detect_dpi / dpi, 0.25, 4.0))
        img = image if abs(scale - 1.0) < 1e-3 else cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        out = []
        for d in self.detect_obbs(img):
            poly = canonical_corners(d.poly / scale)
            out.append(Detection(self.names[d.cls], d.conf, [(float(x), float(y)) for x, y in poly]))
        return out


def build_detector(cfg: "DetectorConfig | str") -> Detector:
    """Factory keyed by ``configs/inference.yaml::detector.kind`` (a bare kind string is accepted too)."""
    from .config import DetectorConfig

    if isinstance(cfg, str):
        cfg = DetectorConfig(kind=cfg)
    if cfg.kind == "null":
        return NullDetector()
    if cfg.kind == "yolo_obb":
        return YoloObbDetector(cfg)
    raise ValueError(f"Unknown detector kind '{cfg.kind}' (expected 'null' or 'yolo_obb')")
