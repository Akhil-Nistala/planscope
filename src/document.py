"""Document loading and page rendering (PDF via PyMuPDF, or plain images).

Coordinate convention used everywhere downstream: **pixels of the rendered page
image**, origin top-left, x right, y down. ``PageData.scale`` converts PDF points
to pixels (``px = pt * dpi / 72``).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import pymupdf as fitz
import numpy as np
from PIL import Image

from .config import PipelineConfig
from .router import Route, RouteDecision, route_page

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
PDF_SUFFIXES = {".pdf"}


@dataclass
class Word:
    """A word from the PDF text layer, in rendered-image pixel coordinates."""

    text: str
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 (px)
    block: int
    line: int


@dataclass
class PageData:
    """A routed + rendered page ready for detection."""

    page: int  # 1-based
    route: Route
    decision: RouteDecision | None
    dpi: int | None
    image: np.ndarray  # H x W x 3, uint8, RGB
    words: list[Word] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return int(self.image.shape[1])


def effective_dpi(width_pt: float, height_pt: float, dpi: int, max_pixels: int) -> int:
    """Lower ``dpi`` so that the rendered page stays below ``max_pixels``."""
    px = (width_pt * dpi / 72.0) * (height_pt * dpi / 72.0)
    if px <= max_pixels:
        return dpi
    return max(36, int(math.floor(dpi * math.sqrt(max_pixels / px))))


def render_page(page: fitz.Page, dpi: int) -> np.ndarray:
    """Render a PDF page to an RGB uint8 array (respects page rotation)."""
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3).copy()


def words_to_pixels(page: fitz.Page, raw_words: list, dpi: int) -> list[Word]:
    """Convert PyMuPDF word tuples (PDF points) to :class:`Word` in rendered-image pixels.

    Text extraction returns unrotated-page coordinates; the pixmap respects /Rotate.
    """
    m = page.rotation_matrix * fitz.Matrix(dpi / 72.0, dpi / 72.0)
    words: list[Word] = []
    for w in raw_words:
        r = (fitz.Rect(w[:4]) * m).normalize()
        words.append(Word(w[4], (r.x0, r.y0, r.x1, r.y1), int(w[5]), int(w[6])))
    return words


def _iter_pdf(path: Path, cfg: PipelineConfig) -> Iterator[PageData]:
    with fitz.open(path) as doc:
        for page in doc:
            t0 = time.perf_counter()
            decision, raw_words = route_page(page, cfg.router)
            t_route = time.perf_counter() - t0

            base_dpi = cfg.render.vector_dpi if decision.route is Route.VECTOR else cfg.render.dpi
            dpi = effective_dpi(page.rect.width, page.rect.height, base_dpi, cfg.render.max_pixels)
            if dpi != base_dpi:
                log.warning("page %d: DPI lowered %d -> %d (max_pixels)", page.number + 1, base_dpi, dpi)
            t1 = time.perf_counter()
            image = render_page(page, dpi)
            t_render = time.perf_counter() - t1

            words = words_to_pixels(page, raw_words, dpi) if decision.route is Route.VECTOR else []
            yield PageData(
                page=page.number + 1,
                route=decision.route,
                decision=decision,
                dpi=dpi,
                image=image,
                words=words,
                timings={"route_s": t_route, "render_s": t_render},
            )


def _iter_image(path: Path, cfg: PipelineConfig) -> Iterator[PageData]:
    t0 = time.perf_counter()
    with Image.open(path) as im:
        dpi_meta = im.info.get("dpi")
        image = np.asarray(im.convert("RGB")).copy()
    dpi = int(round(dpi_meta[0])) if dpi_meta and dpi_meta[0] else cfg.images.default_dpi
    log.info("%s: image input -> raster (no text layer)", path.name)
    yield PageData(
        page=1,
        route=Route.RASTER,
        decision=None,
        dpi=dpi,
        image=image,
        timings={"route_s": 0.0, "render_s": time.perf_counter() - t0},
    )


def load_pages(path: str | Path, cfg: PipelineConfig) -> Iterator[PageData]:
    """Yield routed, rendered pages for a PDF or image file."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in PDF_SUFFIXES:
        yield from _iter_pdf(p, cfg)
    elif suffix in IMAGE_SUFFIXES:
        yield from _iter_image(p, cfg)
    else:
        raise ValueError(f"Unsupported input type: {p.suffix} ({p})")
