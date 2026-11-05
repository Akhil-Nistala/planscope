"""Drawing canvas: a PyMuPDF page plus font measurement, oriented text boxes and an outline mode.

Coordinates are PDF points on the unrotated page, origin top-left, y down (PyMuPDF's
convention). Angles are degrees, counter-clockwise as seen on the rendered page.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf as fitz

INK = (0.0, 0.0, 0.0)
GREY = (0.45, 0.45, 0.45)
LIGHT = (0.75, 0.75, 0.75)

Point = tuple[float, float]
Poly = list[Point]

# Japanese fonts, all used if present (one is drawn per sheet). Extra files can be added with
# PLANSCOPE_JP_FONTS (os.pathsep-separated). PyMuPDF's built-in "japan" font is the last resort:
# it sets Latin characters full-width, which splits words on extraction.
JP_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",       # Debian/Ubuntu/Colab (fonts-noto-cjk)
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",            # Arch/Fedora
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",                  # macOS
    "C:/Windows/Fonts/YuGothR.ttc",
    "C:/Windows/Fonts/YuGothM.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
    "C:/Windows/Fonts/msmincho.ttc",
    "C:/Windows/Fonts/meiryo.ttc",
    "C:/Windows/Fonts/BIZ-UDGothicR.ttc",
]
LATIN_FONTS = {"helv": 0.7, "cour": 0.15, "tiro": 0.15}  # PDF base-14: Helvetica, Courier, Times


@lru_cache(maxsize=1)
def jp_font_files() -> tuple[str, ...]:
    """Existing Japanese font files, environment overrides first."""
    env = [p for p in os.environ.get("PLANSCOPE_JP_FONTS", "").split(os.pathsep) if p]
    return tuple(dict.fromkeys(p for p in env + JP_FONT_CANDIDATES if Path(p).is_file()))


@dataclass(frozen=True)
class FontChoice:
    """A font to draw a sheet with: PDF resource name plus either a file or a built-in name."""

    name: str
    file: str | None = None

    @property
    def label(self) -> str:
        """Short human-readable name (for metadata)."""
        return Path(self.file).name if self.file else self.name


def choose_font(lang: str, rng: np.random.Generator) -> FontChoice:
    """Pick the font for one sheet."""
    if lang == "ja":
        files = jp_font_files()
        return FontChoice("jpf", files[int(rng.integers(len(files)))]) if files else FontChoice("japan")
    names = list(LATIN_FONTS)
    p = np.array([LATIN_FONTS[n] for n in names])
    return FontChoice(str(rng.choice(names, p=p / p.sum())))


@lru_cache(maxsize=16)
def _load_font(choice: FontChoice) -> fitz.Font:
    return fitz.Font(fontfile=choice.file) if choice.file else fitz.Font(choice.name)


def rotate_point(p: Point, pivot: Point, angle: float) -> Point:
    """Rotate ``p`` about ``pivot`` by ``angle`` degrees, counter-clockwise on screen (y down)."""
    a = math.radians(angle)
    dx, dy = p[0] - pivot[0], p[1] - pivot[1]
    return (pivot[0] + dx * math.cos(a) + dy * math.sin(a), pivot[1] - dx * math.sin(a) + dy * math.cos(a))


def rect_poly(x0: float, y0: float, x1: float, y1: float, pad: float = 0.0) -> Poly:
    """Axis-aligned rectangle as a clockwise polygon starting top-left."""
    return [(x0 - pad, y0 - pad), (x1 + pad, y0 - pad), (x1 + pad, y1 + pad), (x0 - pad, y1 + pad)]


def aabb(poly: Poly) -> tuple[float, float, float, float]:
    """Axis-aligned bounds of a polygon."""
    xs, ys = [p[0] for p in poly], [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def overlaps(a: tuple[float, ...], b: tuple[float, ...], gap: float = 0.0) -> bool:
    """True if two boxes overlap (or come closer than ``gap``)."""
    return not (a[2] + gap <= b[0] or b[2] + gap <= a[0] or a[3] + gap <= b[1] or b[3] + gap <= a[1])


@dataclass
class Canvas:
    """One PDF page with drawing helpers.

    In ``outline`` mode, text drawn with ``keep_real=True`` is held back and replayed on the
    final page after every other glyph has been converted to paths (see ``sheet.outline_text``),
    mimicking CAD exports where only a few attributes survive as real text.

    Everything is buffered in one :class:`pymupdf.Shape` and written by :meth:`commit` (one
    content-stream write instead of one per element, which is quadratic on busy sheets). Text is
    always painted above line work.
    """

    page: fitz.Page
    font: FontChoice
    outline: bool = False
    deferred: list[tuple[Point, str, dict[str, Any]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._font = _load_font(self.font)
        if self.font.file:
            self.page.insert_font(fontname=self.font.name, fontfile=self.font.file)
        self.shape = self.page.new_shape()
        # Clamp font metrics so boxes stay reasonably tight across very different fonts.
        self._asc = min(max(self._font.ascender, 0.75), 1.0)
        self._desc = min(max(-self._font.descender, 0.1), 0.3)

    @property
    def W(self) -> float:
        """Page width (pt)."""
        return self.page.rect.width

    @property
    def H(self) -> float:
        """Page height (pt)."""
        return self.page.rect.height

    # -- measurement --------------------------------------------------------
    def width(self, s: str, size: float) -> float:
        """Advance width of ``s`` (pt)."""
        return self._font.text_length(s, fontsize=size)

    def asc(self, size: float) -> float:
        """Height above the baseline (pt)."""
        return self._asc * size

    def desc(self, size: float) -> float:
        """Depth below the baseline (pt, positive)."""
        return self._desc * size

    def baseline_for(self, top: float, bottom: float, size: float) -> float:
        """Baseline that vertically centres a line of ``size`` between ``top`` and ``bottom``."""
        return (top + bottom) / 2 + (self.asc(size) - self.desc(size)) / 2

    def text_poly(self, x: float, y: float, s: str, size: float, angle: float = 0.0, pad: float = 0.0) -> Poly:
        """Oriented box of text whose baseline starts at (x, y)."""
        corners = rect_poly(x, y - self.asc(size), x + self.width(s, size), y + self.desc(size), pad)
        return [rotate_point(c, (x, y), angle) for c in corners] if angle else corners

    # -- drawing ------------------------------------------------------------
    def text(self, x: float, y: float, s: str, size: float, color=INK, angle: float = 0.0,
             keep_real: bool = False) -> Poly:
        """Draw ``s`` with its baseline starting at (x, y); returns its oriented box."""
        if s:
            kw: dict[str, Any] = {"fontname": self.font.name, "fontsize": size, "color": color}
            if angle:
                kw["morph"] = (fitz.Point(x, y), fitz.Matrix(angle))
            if self.outline and keep_real:
                self.deferred.append(((x, y), s, kw))
            else:
                self.shape.insert_text(fitz.Point(x, y), s, **kw)
        return self.text_poly(x, y, s, size, angle)

    def commit(self) -> None:
        """Write everything drawn so far to the page."""
        self.shape.commit()
        self.shape = self.page.new_shape()

    def replay_deferred(self, page: fitz.Page) -> None:
        """Draw held-back real text onto ``page`` (outline mode)."""
        if self.deferred and self.font.file:
            page.insert_font(fontname=self.font.name, fontfile=self.font.file)
        for (x, y), s, kw in self.deferred:
            page.insert_text(fitz.Point(x, y), s, **kw)

    def line(self, p0: Point, p1: Point, w: float = 0.5, color=INK, dashes: str | None = None) -> None:
        """Straight line."""
        self.shape.draw_line(fitz.Point(*p0), fitz.Point(*p1))
        self.shape.finish(color=color, width=w, dashes=dashes, closePath=False)

    def rect(self, r: tuple[float, float, float, float], w: float = 0.6, color=INK, fill=None,
             dashes: str | None = None) -> None:
        """Rectangle outline and/or fill (``color=None`` for fill only)."""
        self.shape.draw_rect(fitz.Rect(*r))
        self.shape.finish(color=color, width=w, fill=fill, dashes=dashes)

    def circle(self, c: Point, radius: float, w: float = 0.5, color=INK, fill=None) -> None:
        """Circle."""
        self.shape.draw_circle(fitz.Point(*c), radius)
        self.shape.finish(color=color, width=w, fill=fill)

    def polyline(self, pts: list[Point], w: float = 0.5, color=INK, fill=None, closed: bool = False) -> None:
        """Polyline / polygon."""
        self.shape.draw_polyline([fitz.Point(*p) for p in pts])
        self.shape.finish(color=color, width=w, fill=fill, closePath=closed)

    def arc(self, p0: Point, ctrl: Point, p1: Point, w: float = 0.4, color=GREY) -> None:
        """Quadratic-looking arc (door swing)."""
        self.shape.draw_curve(fitz.Point(*p0), fitz.Point(*ctrl), fitz.Point(*p1))
        self.shape.finish(color=color, width=w, closePath=False)
