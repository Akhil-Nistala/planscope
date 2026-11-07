"""Oriented-box geometry shared by training (tiling), inference (tile merging) and evaluation.

Polygons are (4, 2) float arrays in image pixels, y down. "Clockwise" means clockwise on screen,
i.e. a positive shoelace sum with y pointing down.
"""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

Poly = np.ndarray  # (N, 2) float


def as_poly(p: Sequence[Sequence[float]]) -> Poly:
    """Coerce to a (N, 2) float64 array."""
    return np.asarray(p, dtype=np.float64).reshape(-1, 2)


def signed_area(p: Poly) -> float:
    """Shoelace sum / 2 (positive = clockwise on screen with y down)."""
    x, y = p[:, 0], p[:, 1]
    return float((np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def area(p: Poly) -> float:
    """Polygon area."""
    return abs(signed_area(p))


def _convex(p: Poly) -> np.ndarray:
    """Convex hull as float32 for OpenCV (also fixes winding)."""
    return cv2.convexHull(p.astype(np.float32))


def intersection_area(a: Poly, b: Poly) -> float:
    """Intersection area of two convex polygons (oriented boxes)."""
    if area(a) <= 0 or area(b) <= 0:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(_convex(a), _convex(b))
    return max(0.0, float(inter))


def iou(a: Poly, b: Poly) -> float:
    """Intersection over union of two convex polygons."""
    i = intersection_area(a, b)
    u = area(a) + area(b) - i
    return i / u if u > 0 else 0.0


def ios(a: Poly, b: Poly) -> float:
    """Intersection over the smaller polygon's area."""
    i = intersection_area(a, b)
    m = min(area(a), area(b))
    return i / m if m > 0 else 0.0


def clip_to_rect(p: Poly, x0: float, y0: float, x1: float, y1: float) -> Poly:
    """Sutherland-Hodgman clip of a polygon to an axis-aligned rectangle (may return < 3 points)."""
    def clip(pts: list, inside, cross) -> list:
        out = []
        for i, cur in enumerate(pts):
            prev = pts[i - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(cross(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cross(prev, cur))
        return out

    def x_cross(xc):
        return lambda a, b: (xc, a[1] + (b[1] - a[1]) * (xc - a[0]) / (b[0] - a[0]))

    def y_cross(yc):
        return lambda a, b: (a[0] + (b[0] - a[0]) * (yc - a[1]) / (b[1] - a[1]), yc)

    pts = [tuple(v) for v in as_poly(p)]
    for inside, cross in ((lambda q: q[0] >= x0, x_cross(x0)), (lambda q: q[0] <= x1, x_cross(x1)),
                          (lambda q: q[1] >= y0, y_cross(y0)), (lambda q: q[1] <= y1, y_cross(y1))):
        if not pts:
            break
        pts = clip(pts, inside, cross)
    return np.array(pts, dtype=np.float64).reshape(-1, 2)


def min_area_box(points: Poly) -> Poly:
    """Minimum-area oriented rectangle around ``points``, as 4 canonical corners."""
    rect = cv2.minAreaRect(as_poly(points).astype(np.float32))
    return canonical_corners(cv2.boxPoints(rect).astype(np.float64))


def canonical_corners(p: Poly) -> Poly:
    """Reorder an oriented box's corners: clockwise on screen, starting at the box's top-left.

    The box's "own orientation" is taken as the rotation of its edge closest to horizontal
    (|angle| <= 45 deg). That is what a detector can recover: OBB angles are ambiguous modulo 90 deg,
    so a vertical text box is reported in its landscape-ish frame.
    """
    p = as_poly(p)
    if signed_area(p) < 0:
        p = p[::-1]
    best = None
    for i in range(4):  # pick the edge whose direction is closest to +x
        e = p[(i + 1) % 4] - p[i]
        ang = math.atan2(e[1], e[0])
        if best is None or abs(ang) < abs(best[1]):
            best = (i, ang)
    return np.roll(p, -best[0], axis=0)


def rotation_deg(p: Poly) -> float:
    """Angle of the first (top) edge, degrees counter-clockwise on screen."""
    e = p[1] - p[0]
    return -math.degrees(math.atan2(e[1], e[0]))
