"""Shared fixtures: tiny in-memory PDFs covering vector / raster / rotated cases."""

from __future__ import annotations

from pathlib import Path

import pymupdf as fitz
import numpy as np
import pytest

from src.config import PipelineConfig

LINES = [
    "FINISH SCHEDULE Room Floor Baseboard Wall Ceiling",
    "Office vinyl floor tile vinyl base H60 gypsum board painted",
    "Corridor carpet tile timber skirting calcium silicate board",
    "Storage trowelled mortar mortar base exposed concrete",
]


@pytest.fixture()
def cfg() -> PipelineConfig:
    """Default config with fast render DPIs for tests."""
    c = PipelineConfig()
    c.render.dpi = 100
    c.render.vector_dpi = 72
    return c


def _text_pdf(path: Path, rotate: int = 0) -> Path:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    for i, line in enumerate(LINES):
        page.insert_text((50, 100 + 20 * i), line, fontsize=11)
    page.insert_text((50, 220), "仕上表 室名 床 幅木 壁 天井 ビニル床タイル 石膏ボード", fontname="japan", fontsize=11)
    page.set_rotation(rotate)
    doc.save(path)
    return path


@pytest.fixture()
def vector_pdf(tmp_path: Path) -> Path:
    """Digital PDF with a real text layer."""
    return _text_pdf(tmp_path / "vector.pdf")


@pytest.fixture()
def rotated_vector_pdf(tmp_path: Path) -> Path:
    """Digital PDF with /Rotate 90."""
    return _text_pdf(tmp_path / "rotated.pdf", rotate=90)


@pytest.fixture()
def raster_pdf(tmp_path: Path, vector_pdf: Path) -> Path:
    """Image-only PDF (scan-like): render the vector page and embed it as a picture."""
    src = fitz.open(vector_pdf)
    pix = src[0].get_pixmap(dpi=100)
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, pixmap=pix)
    out = tmp_path / "raster.pdf"
    doc.save(out)
    return out


@pytest.fixture()
def png_image(tmp_path: Path) -> Path:
    """Plain PNG with DPI metadata."""
    from PIL import Image

    out = tmp_path / "scan.png"
    Image.fromarray(np.full((300, 200, 3), 255, np.uint8)).save(out, dpi=(200, 200))
    return out
