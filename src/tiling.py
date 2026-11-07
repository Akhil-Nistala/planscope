"""Sliding-window tiling and merging of oriented detections (SAHI-style, for OBBs).

Drawing sheets are large (an A1 sheet at 150 DPI is ~5000 x 3500 px) while callout text is ~10 px
tall, so detection runs on overlapping tiles at native resolution, optionally plus one downscaled
full-sheet pass for regions larger than a tile. Pieces of one object cut by tile borders are merged
back together: same class + high intersection-over-smaller -> union box.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Poly, area, as_poly, canonical_corners, ios, iou, min_area_box


def tile_origins(length: int, tile: int, overlap: float) -> list[int]:
    """Start offsets of tiles covering ``[0, length)``; the last tile is flush with the end."""
    if length <= tile:
        return [0]
    stride = max(1, int(round(tile * (1.0 - overlap))))
    starts = list(range(0, length - tile, stride))
    starts.append(length - tile)
    return sorted(set(starts))


def tile_grid(width: int, height: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Tile rectangles (x0, y0, x1, y1) covering a ``width`` x ``height`` image."""
    return [(x, y, min(x + tile, width), min(y + tile, height))
            for y in tile_origins(height, tile, overlap) for x in tile_origins(width, tile, overlap)]


@dataclass
class OBB:
    """Class-agnostic oriented detection used during merging."""

    cls: int
    conf: float
    poly: Poly


def merge_obbs(dets: list[OBB], iou_thr: float = 0.5, ios_thr: float = 0.6, max_rounds: int = 5) -> list[OBB]:
    """Class-wise greedy merge.

    For detections of the same class, sorted by confidence: if a detection overlaps an already kept
    one with IoU >= ``iou_thr`` it is a duplicate (dropped); else if its intersection-over-smaller is
    >= ``ios_thr`` it is a piece of the same object (kept box grows to the union). Repeats until stable,
    because a grown box can newly overlap another piece.
    """
    cur = sorted((d for d in dets if area(as_poly(d.poly)) >= 1.0), key=lambda d: -d.conf)  # drop slivers
    for _ in range(max_rounds):
        kept: list[OBB] = []
        grew = False
        for d in cur:
            duplicate, target = False, None
            for k in kept:
                if k.cls != d.cls:
                    continue
                if iou(k.poly, d.poly) >= iou_thr:
                    duplicate = True
                    break
                if ios(k.poly, d.poly) >= ios_thr:
                    target = k
                    break
            if duplicate:
                continue
            if target is not None:
                target.poly = min_area_box(np.vstack([target.poly, d.poly]))
                target.conf = max(target.conf, d.conf)
                grew = True
            else:
                kept.append(OBB(d.cls, d.conf, canonical_corners(as_poly(d.poly))))
        cur = sorted(kept, key=lambda d: -d.conf)
        if not grew:
            break
    return cur
