"""Vector/raster input router.

For every PDF page we decide whether its text layer is usable:

* ``VECTOR`` -- the page has a real text layer (digital PDF exported from CAD/Office
  with fonts embedded). Words + coordinates are taken directly from the PDF; the
  page is still rendered (at a lower DPI) so the layout detector can run on it.
* ``RASTER`` -- a scan, or a CAD export where glyphs were drawn as vector paths /
  have a broken ToUnicode map. Text must come from OCR downstream (``needs_ocr``).

The heuristic is deliberately simple and fully configurable
(:class:`src.config.RouterConfig`): a page is VECTOR only if it has at least
``min_words`` words, at least ``min_valid_char_ratio`` of its characters are
"valid" (see :func:`is_valid_char`) and, optionally, word boxes cover at least
``min_text_coverage`` of the page.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Sequence

from .config import RouterConfig

log = logging.getLogger(__name__)

# PyMuPDF word tuple: (x0, y0, x1, y1, text, block_no, line_no, word_no)
WordTuple = tuple[float, float, float, float, str, int, int, int]

_INVALID_CATEGORIES = {"Co", "Cn", "Cc", "Cs"}  # private use, unassigned, control, surrogate
_REPLACEMENT = "�"


class Route(str, Enum):
    """Routing decision for a page."""

    VECTOR = "vector"
    RASTER = "raster"


@dataclass
class RouteDecision:
    """Routing decision plus the statistics it was based on (logged and stored in the JSON)."""

    route: Route
    n_words: int
    n_chars: int
    valid_char_ratio: float
    text_coverage: float
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-serialisable statistics (the route itself is stored at page level)."""
        d = asdict(self)
        d.pop("route")
        d["valid_char_ratio"] = round(self.valid_char_ratio, 4)
        d["text_coverage"] = round(self.text_coverage, 4)
        return d


def is_valid_char(ch: str) -> bool:
    """Return True if ``ch`` looks like real extracted text.

    Rejects the Unicode replacement character and characters in the private-use,
    unassigned, control and surrogate categories -- the typical output of fonts
    with missing/broken ToUnicode maps in CAD exports.
    """
    if ch == _REPLACEMENT:
        return False
    return unicodedata.category(ch) not in _INVALID_CATEGORIES


def text_stats(words: Sequence[WordTuple], page_area: float) -> tuple[int, float, float]:
    """Compute ``(n_chars, valid_char_ratio, text_coverage)`` for a list of PyMuPDF words."""
    chars = [c for w in words for c in w[4] if not c.isspace()]
    n_chars = len(chars)
    valid_ratio = (sum(is_valid_char(c) for c in chars) / n_chars) if n_chars else 0.0
    covered = sum(max(0.0, w[2] - w[0]) * max(0.0, w[3] - w[1]) for w in words)
    coverage = min(1.0, covered / page_area) if page_area > 0 else 0.0
    return n_chars, valid_ratio, coverage


def classify_words(words: Sequence[WordTuple], page_area: float, cfg: RouterConfig) -> RouteDecision:
    """Classify a page from its extracted words (pure function -- easy to unit test)."""
    n_chars, valid_ratio, coverage = text_stats(words, page_area)
    reasons: list[str] = []
    if len(words) < cfg.min_words:
        reasons.append(f"n_words={len(words)} < min_words={cfg.min_words}")
    if valid_ratio < cfg.min_valid_char_ratio:
        reasons.append(f"valid_char_ratio={valid_ratio:.3f} < {cfg.min_valid_char_ratio}")
    if cfg.min_text_coverage > 0 and coverage < cfg.min_text_coverage:
        reasons.append(f"text_coverage={coverage:.4f} < {cfg.min_text_coverage}")
    route = Route.RASTER if reasons else Route.VECTOR
    if not reasons:
        reasons.append("text layer passes all checks")
    return RouteDecision(route, len(words), n_chars, valid_ratio, coverage, reasons)


def route_page(page: "pymupdf.Page", cfg: RouterConfig) -> tuple[RouteDecision, list[WordTuple]]:  # noqa: F821
    """Extract words from a PyMuPDF page and classify it.

    Returns the decision and the raw word tuples (in PDF points, rotated-page space).
    """
    words: list[WordTuple] = page.get_text("words", sort=True)
    decision = classify_words(words, page.rect.width * page.rect.height, cfg)
    log.info(
        "page %d -> %s (%s)", page.number + 1, decision.route.value, "; ".join(decision.reasons)
    )
    return decision, words
