"""Table structure recognition for detected ``schedule_table`` regions (Milestone 5).

Rule-based, because finish schedules are ruled grids:

1. Warp the oriented region to an upright crop (handles scan skew and rotated tables).
2. Binarise; extract horizontal / vertical ruling lines with long morphological openings.
3. Row separators = rows of the crop where horizontal rule pixels span most of the width; column
   separators likewise. If a table has (almost) no vertical rules, columns come from whitespace gaps
   in the text ink instead.
4. Merged cells: a rule segment missing between two grid cells joins them (union-find), e.g. a level
   column whose cell spans several rows.
5. Cell text: vector pages take the text-layer words whose centre falls in the cell (reading order);
   raster pages leave text ``null`` for OCR. The first row is the header row; headers are mapped to
   canonical fields (room, floor, baseboard, wall, ceiling, ...) via ``configs/vocab.yaml``.

Cell boxes are returned as axis-aligned page-pixel boxes, as the output contract requires.
"""

from __future__ import annotations

import logging
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import yaml

from .geometry import as_poly, canonical_corners

log = logging.getLogger(__name__)

VOCAB_PATH = Path(__file__).resolve().parents[1] / "configs" / "vocab.yaml"
TABLE_DPI = 200  # crops are resampled to this resolution before grid detection
MIN_CELL = 8  # px at TABLE_DPI: separators closer than this are one blurred rule (rows are ~30+ px)


# -- header canonicalisation ---------------------------------------------------
def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    return "".join(ch for ch in s if not ch.isspace())


@lru_cache(maxsize=4)
def header_lexicon(path: str = str(VOCAB_PATH)) -> tuple[tuple[str, str], ...]:
    """(normalised spelling, canonical field) pairs, longest spelling first."""
    v = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    pairs: dict[str, str] = {}
    for lang in v.get("headers", {}).values():
        for canon, spelling in lang.items():
            pairs[_norm(spelling)] = canon
    for lang in v.get("header_variants", {}).values():
        for canon, spellings in lang.items():
            for s in spellings:
                pairs[_norm(s)] = canon
    return tuple(sorted(pairs.items(), key=lambda kv: -len(kv[0])))


def canonical_header(text: str | None, lexicon: Sequence[tuple[str, str]] | None = None) -> str | None:
    """Map a header cell to a canonical field: exact spelling first, then the longest known spelling
    contained in it (e.g. "天井仕上げ（既存）" -> ceiling). None if nothing matches."""
    if not text:
        return None
    lex = lexicon or header_lexicon()
    t = _norm(text)
    for spelling, canon in lex:
        if t == spelling:
            return canon
    for spelling, canon in lex:
        if len(spelling) >= 2 and spelling in t:
            return canon
    return None


# -- grid detection ----------------------------------------------------------
def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index runs where a 1-D boolean mask is True."""
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return list(zip(idx[::2], idx[1::2]))


def _separators(profile: np.ndarray, frac: float, length: int, max_thick: int = 10) -> list[float]:
    """Separators where a rule-pixel profile exceeds ``frac * length``.

    A thin run is one rule (its centre). A thick run is a filled band (e.g. a grey header row merged
    with its rules): both of its edges are separators.
    """
    seps: list[float] = []
    for a, b in _runs(profile >= frac * length):
        if b - a <= max_thick:
            seps.append((a + b - 1) / 2)
        else:
            seps += [float(a), float(b - 1)]
    return seps


def rule_mask(gray: np.ndarray) -> np.ndarray:
    """Pixels clearly darker than the paper: catches faint thin rules on faded, noisy scans."""
    bg = float(np.median(gray))
    dark = float(np.percentile(gray, 1))
    thr = bg - max(18.0, 0.2 * (bg - dark))
    return (gray < thr).astype(np.uint8) * 255


def _gap_separators(ink_profile: np.ndarray, min_gap: int) -> list[float]:
    """Column separators from whitespace: centres of empty runs at least ``min_gap`` wide (interior only)."""
    empty = ink_profile == 0
    seps = []
    for a, b in _runs(empty):
        if a > 0 and b < len(ink_profile) and b - a >= min_gap:
            seps.append((a + b - 1) / 2)
    return seps


def _ensure_borders(seps: list[float], length: int, tol: float) -> list[float]:
    s = sorted(seps)
    if not s or s[0] > tol:
        s = [0.0] + s
    if s[-1] < length - 1 - tol:
        s = s + [float(length - 1)]
    return s


def _dedupe(seps: list[float], min_dist: float) -> list[float]:
    out: list[float] = []
    for s in sorted(seps):
        if not out or s - out[-1] >= min_dist:
            out.append(s)
    return out


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        self.p[self.find(a)] = self.find(b)


def detect_grid(gray: np.ndarray) -> tuple[list[float], list[float], np.ndarray, np.ndarray, bool]:
    """Row/column separators of an upright table crop.

    Returns (ys, xs, horizontal-rule mask, vertical-rule mask, columns_from_rules).
    """
    h, w = gray.shape
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    rules = rule_mask(gray)
    # bridge small breaks (noise, JPEG) along each direction, then keep only long straight runs
    hr = cv2.morphologyEx(rules, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
    vr = cv2.morphologyEx(rules, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))
    hmask = cv2.morphologyEx(hr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 6), 1))) > 0
    vmask = cv2.morphologyEx(vr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 6)))) > 0
    # Row rules may skip a merged first column, so 0.5 of the width is enough.
    ys = _dedupe(_separators(hmask.sum(1), 0.5, w), MIN_CELL)
    xs = _dedupe(_separators(vmask.sum(0), 0.5, h), MIN_CELL)
    from_rules = len(xs) >= 3
    if not from_rules:
        # borderless columns: whitespace gaps in the text ink below the header row
        text = ink.copy()
        text[hmask | vmask] = 0
        body_top = int(ys[1]) + 2 if len(ys) >= 2 else 0
        prof = (text[body_top:] > 0).sum(0)
        typical_h = (ys[-1] - ys[0]) / max(1, len(ys) - 1) if len(ys) >= 2 else 20
        xs = _dedupe(xs + _gap_separators(prof, max(4, int(typical_h * 0.6))), MIN_CELL)
    tol = 2 * MIN_CELL  # a gap smaller than a cell between crop edge and outer rule is margin, not a row
    return _ensure_borders(ys, h, tol), _ensure_borders(xs, w, tol), hmask, vmask, from_rules


def build_cells(ys: list[float], xs: list[float], hmask: np.ndarray, vmask: np.ndarray, merge_cols: bool,
                present: float = 0.5) -> list[tuple[int, int, int, int]]:
    """Grid cells as (row, col, row_span, col_span), merging across missing rule segments."""
    nr, nc = len(ys) - 1, len(xs) - 1
    uf = _UF(nr * nc)

    def seg_h(y: float, x0: float, x1: float) -> float:  # share of a horizontal segment that is ruled
        yi, a, b = int(round(y)), int(x0) + 2, int(x1) - 2
        if b <= a:
            return 1.0
        band = hmask[max(0, yi - 2): yi + 3, a:b]
        return float(band.any(0).mean()) if band.size else 0.0

    def seg_v(x: float, y0: float, y1: float) -> float:
        xi, a, b = int(round(x)), int(y0) + 2, int(y1) - 2
        if b <= a:
            return 1.0
        band = vmask[a:b, max(0, xi - 2): xi + 3]
        return float(band.any(1).mean()) if band.size else 0.0

    for r in range(nr - 1):
        for c in range(nc):
            if seg_h(ys[r + 1], xs[c], xs[c + 1]) < present:
                uf.union(r * nc + c, (r + 1) * nc + c)
    if merge_cols:
        for r in range(nr):
            for c in range(nc - 1):
                if seg_v(xs[c + 1], ys[r], ys[r + 1]) < present:
                    uf.union(r * nc + c, r * nc + c + 1)
    groups: dict[int, list[tuple[int, int]]] = {}
    for r in range(nr):
        for c in range(nc):
            groups.setdefault(uf.find(r * nc + c), []).append((r, c))
    cells = []
    for members in groups.values():
        rs, cs = [m[0] for m in members], [m[1] for m in members]
        r0, c0, rsp, csp = min(rs), min(cs), max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
        if rsp * csp == len(members):
            cells.append((r0, c0, rsp, csp))
        else:  # non-rectangular merge (noise): fall back to single cells
            cells += [(r, c, 1, 1) for r, c in members]
    return sorted(cells)


# -- public entry point --------------------------------------------------------
def upright_crop(gray: np.ndarray, polygon: Sequence[Sequence[float]], margin: int = 4, scale: float = 1.0
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Warp an oriented box to an upright crop, resampled by ``scale``. Returns (crop, 2x3 affine page->crop)."""
    p = canonical_corners(as_poly(polygon))
    w = int(round(np.linalg.norm(p[1] - p[0]) * scale))
    h = int(round(np.linalg.norm(p[3] - p[0]) * scale))
    src = np.float32([p[0], p[1], p[3]])
    dst = np.float32([[margin, margin], [margin + w, margin], [margin, margin + h]])
    A = cv2.getAffineTransform(src, dst)
    crop = cv2.warpAffine(gray, A, (w + 2 * margin, h + 2 * margin), flags=cv2.INTER_LINEAR, borderValue=255)
    return crop, A


def recognize_table(image: np.ndarray, polygon: Sequence[Sequence[float]], words: list | None,
                    lexicon: Sequence[tuple[str, str]] | None = None, dpi: int | None = None) -> dict[str, Any] | None:
    """Recover a schema ``table`` from a page image region.

    ``words`` are the page's text-layer words (objects with ``.text`` and ``.bbox`` in page pixels) for
    vector pages, or None for raster pages (cell text is then null / needs_ocr). The crop is resampled to
    ``TABLE_DPI`` when the page ``dpi`` is known, so pixel thresholds mean the same physical size.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    scale = float(np.clip(TABLE_DPI / dpi, 0.5, 3.0)) if dpi else 1.0
    crop, A = upright_crop(gray, polygon, scale=scale)
    if min(crop.shape) < 12:
        return None
    ys, xs, hmask, vmask, from_rules = detect_grid(crop)
    if len(ys) < 3 or len(xs) < 2:  # need a header row and at least one body row
        return None
    cells = build_cells(ys, xs, hmask, vmask, merge_cols=from_rules)
    Ainv = cv2.invertAffineTransform(A)

    # words -> crop coordinates (centres)
    placed: list[tuple[float, float, str]] = []
    if words is not None:
        for wd in words:
            cx, cy = (wd.bbox[0] + wd.bbox[2]) / 2, (wd.bbox[1] + wd.bbox[3]) / 2
            u, v = A @ np.array([cx, cy, 1.0])
            if ys[0] <= v <= ys[-1] and xs[0] <= u <= xs[-1]:
                placed.append((u, v, wd.text))

    vector = words is not None
    src = "vector" if vector else "needs_ocr"
    rows: dict[int, list[dict[str, Any]]] = {}
    for r0, c0, rsp, csp in cells:
        x0, x1, y0, y1 = xs[c0], xs[c0 + csp], ys[r0], ys[r0 + rsp]
        text = None
        if vector:
            inside = sorted(((v, u, t) for u, v, t in placed if x0 <= u < x1 and y0 <= v < y1),
                            key=lambda q: (round(q[0] / 4), q[1]))
            text = " ".join(t for _, _, t in inside)
        corners = np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]], dtype=np.float64) @ Ainv.T
        bbox = [round(float(corners[:, 0].min()), 2), round(float(corners[:, 1].min()), 2),
                round(float(corners[:, 0].max()), 2), round(float(corners[:, 1].max()), 2)]
        rows.setdefault(r0, []).append({"text": text, "bbox": bbox, "text_source": src, "row_span": rsp,
                                        "col_span": csp, "is_header": r0 == 0})
    header_cells = sorted(rows.get(0, []), key=lambda c: c["bbox"][0])
    headers = [c["text"] for c in header_cells]
    canonical = [canonical_header(h, lexicon) if vector else None for h in headers]
    return {"headers": headers, "canonical_headers": canonical,
            "rows": [{"cells": sorted(rows[r], key=lambda c: c["bbox"][0])} for r in sorted(rows)]}
