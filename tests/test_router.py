"""Router: heuristic unit tests + routing of real PDFs + coordinate correctness."""

from __future__ import annotations

import pymupdf as fitz
import pytest

from src.config import RouterConfig
from src.document import effective_dpi, load_pages
from src.router import Route, classify_words, is_valid_char


def _words(texts: list[str]) -> list[tuple]:
    return [(10.0 * i, 0.0, 10.0 * i + 8, 10.0, t, 0, 0, i) for i, t in enumerate(texts)]


@pytest.mark.parametrize("ch,ok", [("a", True), ("床", True), ("ビ", True), ("㎡", True), ("1", True),
                                   ("�", False), ("", False), ("\x07", False)])
def test_is_valid_char(ch: str, ok: bool) -> None:
    assert is_valid_char(ch) is ok


def test_enough_clean_words_is_vector() -> None:
    d = classify_words(_words(["gypsum"] * 20), 1e4, RouterConfig(min_words=15))
    assert d.route is Route.VECTOR and d.valid_char_ratio == 1.0


def test_few_words_is_raster() -> None:
    d = classify_words(_words(["A-101"] * 3), 1e4, RouterConfig(min_words=15))
    assert d.route is Route.RASTER
    assert any("min_words" in r for r in d.reasons)


def test_garbled_text_layer_is_raster() -> None:
    # CAD export with broken ToUnicode: glyphs extracted as private-use / replacement chars
    garbage = ["", "��", "ab"] * 10
    d = classify_words(_words(garbage), 1e4, RouterConfig(min_words=15, min_valid_char_ratio=0.85))
    assert d.route is Route.RASTER
    assert any("valid_char_ratio" in r for r in d.reasons)


def test_thresholds_are_configurable() -> None:
    words = _words(["x"] * 5)
    assert classify_words(words, 1e4, RouterConfig(min_words=15)).route is Route.RASTER
    assert classify_words(words, 1e4, RouterConfig(min_words=5)).route is Route.VECTOR


def test_coverage_check_optional() -> None:
    words = _words(["word"] * 20)  # covers 20*80 = 1600 of 1e6
    assert classify_words(words, 1e6, RouterConfig(min_text_coverage=0.0)).route is Route.VECTOR
    assert classify_words(words, 1e6, RouterConfig(min_text_coverage=0.01)).route is Route.RASTER


def test_empty_page() -> None:
    d = classify_words([], 1e4, RouterConfig())
    assert d.route is Route.RASTER and d.n_words == 0 and d.valid_char_ratio == 0.0


def test_vector_pdf_routes_vector(vector_pdf, cfg) -> None:
    (page,) = list(load_pages(vector_pdf, cfg))
    assert page.route is Route.VECTOR
    assert page.dpi == cfg.render.vector_dpi
    texts = {w.text for w in page.words}
    assert {"gypsum", "Ceiling", "ビニル床タイル"} <= texts


def test_raster_pdf_routes_raster(raster_pdf, cfg) -> None:
    (page,) = list(load_pages(raster_pdf, cfg))
    assert page.route is Route.RASTER and page.words == []
    assert page.dpi == cfg.render.dpi
    h, w, c = page.image.shape
    assert c == 3 and abs(h - 842 * 100 / 72) <= 1 and abs(w - 595 * 100 / 72) <= 1


def test_image_input_is_raster_with_dpi(png_image, cfg) -> None:
    (page,) = list(load_pages(png_image, cfg))
    assert page.route is Route.RASTER and page.dpi == 200 and page.image.shape == (300, 200, 3)


@pytest.mark.parametrize("fixture", ["vector_pdf", "rotated_vector_pdf"])
def test_word_boxes_land_on_ink(fixture, cfg, request) -> None:
    """Word boxes (scaled to pixels) must cover dark pixels in the rendered image, incl. /Rotate pages."""
    (page,) = list(load_pages(request.getfixturevalue(fixture), cfg))
    assert page.words
    gray = page.image.mean(axis=2)
    for w in page.words:
        x0, y0, x1, y1 = (int(round(v)) for v in w.bbox)
        assert 0 <= x0 < x1 <= page.width and 0 <= y0 < y1 <= page.height, w
        assert gray[y0:y1, x0:x1].min() < 128, f"no ink under word {w.text!r} at {w.bbox}"


def test_rotated_page_image_is_landscape(rotated_vector_pdf, cfg) -> None:
    (page,) = list(load_pages(rotated_vector_pdf, cfg))
    assert page.width > page.height


def test_effective_dpi_caps_pixels() -> None:
    a0 = (3370.0, 2384.0)
    dpi = effective_dpi(*a0, dpi=300, max_pixels=50_000_000)
    assert dpi < 300
    assert (a0[0] * dpi / 72) * (a0[1] * dpi / 72) <= 50_000_000
    assert effective_dpi(595, 842, 300, 120_000_000) == 300


def test_mixed_document_routes_per_page(tmp_path, vector_pdf, raster_pdf, cfg) -> None:
    doc = fitz.open()
    for p in (vector_pdf, raster_pdf, vector_pdf):
        with fitz.open(p) as d:
            doc.insert_pdf(d)
    out = tmp_path / "mixed.pdf"
    doc.save(out)
    routes = [p.route for p in load_pages(out, cfg)]
    assert routes == [Route.VECTOR, Route.RASTER, Route.VECTOR]
