"""Generate one synthetic sheet end to end.

``sample_spec`` -> draw the PDF (``layout.compose``) -> render for the sheet's mode -> labels.

Every sheet uses three independent random streams derived from ``(seed, index)``: one for the
spec (split, mode, language, size, ...), one for the drawing content and one for the scan
degradation. Changing e.g. the mode mix therefore does not change what a sheet looks like.

Outputs per sheet (see :class:`SheetResult`):

* image   -- greyscale PNG (vector / outlined) or JPEG (scan) with DPI metadata
* labels  -- YOLO-OBB text: ``class x1 y1 x2 y2 x3 y3 x4 y4`` normalised to [0, 1]
* gt      -- the document a perfect pipeline would output for this sheet, validated against
             ``src/schema.json`` (built with the pipeline's own ``page_record``)
* meta    -- generation parameters, router verdict and the full true text of every region
             (including raster sheets, where the schema-valid GT must carry ``text: null``)
"""

from __future__ import annotations

import io
import logging
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
import pymupdf as fitz
from PIL import Image

from src.config import RouterConfig
from src.detect import CLASSES, Detection
from src.document import PageData, render_page, words_to_pixels
from src.pipeline import page_record
from src.router import Route, route_page
from src.schema import SCHEMA_VERSION, validate_output

from .canvas import Canvas, FontChoice, choose_font
from .components import Ctx, Region, Table
from .config import ScanConfig, SynthConfig
from .layout import SHEET_SIZES, compose

log = logging.getLogger(__name__)


def weighted_choice(rng: np.random.Generator, weights: dict[str, float]) -> str:
    """Sample a key with probability proportional to its (non-negative) weight."""
    keys = [k for k, w in weights.items() if w > 0]
    p = np.array([weights[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


@dataclass
class SheetSpec:
    """Everything that defines a sheet apart from its drawn content."""

    id: str
    index: int
    split: str
    mode: str
    lang: str
    sheet_size: str
    layout: str
    title_block_style: str
    dpi: int
    text_size: float


def sample_spec(index: int, cfg: SynthConfig) -> SheetSpec:
    """Deterministic spec for sheet ``index``."""
    rng = np.random.default_rng([cfg.seed, index, 0])
    split = weighted_choice(rng, cfg.splits)
    mode = weighted_choice(rng, cfg.modes)
    return SheetSpec(
        id=f"{index:05d}", index=index, split=split, mode=mode,
        lang=weighted_choice(rng, cfg.langs),
        sheet_size=weighted_choice(rng, cfg.sheet_sizes),
        layout=weighted_choice(rng, cfg.layouts),
        title_block_style=weighted_choice(rng, cfg.title_block_styles),
        dpi=int(cfg.dpi[mode]),
        text_size=round(float(rng.uniform(*cfg.text_size_pt)), 3),
    )


def build_pdf(spec: SheetSpec, cfg: SynthConfig, vocab: dict) -> tuple[fitz.Document, Canvas, list[Region], FontChoice]:
    """Draw the sheet as a one-page PDF (text still real; see :func:`outline_text`)."""
    rng = np.random.default_rng([cfg.seed, spec.index, 1])
    font = choose_font(spec.lang, rng)
    doc = fitz.open()
    W, H = SHEET_SIZES[spec.sheet_size]
    cv = Canvas(doc.new_page(width=W, height=H), font, outline=(spec.mode == "outlined"))
    ctx = Ctx(cv, spec.lang, vocab, rng, spec.text_size, dict(cfg.callout_angles), tuple(cfg.oblique_range_deg))
    regions = compose(ctx, spec.layout, spec.title_block_style)
    cv.commit()
    return doc, cv, regions, font


def outline_text(doc: fitz.Document, cv: Canvas) -> fitz.Document:
    """Convert every glyph on page 0 to vector paths (no text layer), then replay held-back real text.

    Uses MuPDF's SVG writer with ``text_as_path`` and converts the SVG back to PDF, which gives real
    glyph outlines -- what CAD "explode text" / "text as geometry" exports produce.
    """
    page = doc[0]
    svg = page.get_svg_image(text_as_path=True)
    with fitz.open(stream=svg.encode("utf-8"), filetype="svg") as sdoc, \
            fitz.open("pdf", sdoc.convert_to_pdf()) as src:
        out = fitz.open()
        new = out.new_page(width=page.rect.width, height=page.rect.height)
        new.show_pdf_page(new.rect, src, 0)
    cv.replay_deferred(new)
    return out


def scan_effect(gray: np.ndarray, rng: np.random.Generator, sc: ScanConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    """Print-and-scan look: skew + shift, blur, faded ink, uneven illumination, noise.

    Returns the image, the 2x3 affine applied to pixel coordinates, and the parameters used.
    JPEG compression happens when the image is encoded (quality is in the parameters).
    """
    h, w = gray.shape
    angle = float(rng.uniform(-sc.skew_deg, sc.skew_deg))
    shift = rng.uniform(-0.004, 0.004, 2) * np.array([w, h])
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    M[:, 2] += shift
    g = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255).astype(np.float32)
    blur = float(rng.uniform(0.4, 1.1)) if rng.random() < sc.blur_prob else 0.0
    if blur:
        g = cv2.GaussianBlur(g, (0, 0), blur)
    fade = float(rng.uniform(0.7, 0.95))
    g = g * fade + 255.0 * (1.0 - fade)
    tint = float(rng.uniform(0, 18))
    ramp = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    g -= tint * (ramp if rng.random() < 0.5 else ramp[:, ::-1])
    sigma = float(rng.uniform(*sc.noise_sigma))
    g += rng.standard_normal(g.shape, dtype=np.float32) * sigma
    params = {"skew_deg": round(angle, 4), "shift_px": [round(float(v), 2) for v in shift], "blur_sigma": round(blur, 3),
              "fade": round(fade, 3), "noise_sigma": round(sigma, 2),
              "jpeg_quality": int(rng.integers(sc.jpeg_quality[0], sc.jpeg_quality[1] + 1))}
    return np.clip(g, 0, 255).astype(np.uint8), M, params


def encode_image(gray: np.ndarray, fmt: str, dpi: int, quality: int = 90) -> bytes:
    """PNG or JPEG bytes with DPI metadata (the pipeline reads it for image inputs)."""
    buf = io.BytesIO()
    img = Image.fromarray(gray)
    if fmt == "png":
        img.save(buf, "PNG", dpi=(dpi, dpi), compress_level=3)
    else:
        img.save(buf, "JPEG", dpi=(dpi, dpi), quality=quality)
    return buf.getvalue()


# -- geometry ----------------------------------------------------------------
def apply(T: np.ndarray, pts) -> np.ndarray:
    """Apply a 3x3 affine to an (N, 2) array of points."""
    p = np.asarray(pts, dtype=np.float64)
    return p @ T[:2, :2].T + T[:2, 2]


def poly_area(p: np.ndarray) -> float:
    """Shoelace area."""
    x, y = p[:, 0], p[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def to_px_polygon(poly_pt, T: np.ndarray, W: int, H: int) -> np.ndarray:
    """Region polygon in image pixels, clipped to the image."""
    p = apply(T, poly_pt)
    p[:, 0] = np.clip(p[:, 0], 0, W)
    p[:, 1] = np.clip(p[:, 1], 0, H)
    return p


def to_px_bbox(bbox_pt, T: np.ndarray, W: int, H: int) -> list[float]:
    """Axis-aligned pixel bbox of a (possibly skewed) PDF-point box."""
    x0, y0, x1, y1 = bbox_pt
    p = to_px_polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], T, W, H)
    return [round(float(v), 2) for v in (p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())]


def yolo_obb_line(cls: str, poly_px: np.ndarray, W: int, H: int) -> str:
    """One Ultralytics OBB label line: class index + 4 corners normalised to [0, 1]."""
    norm = poly_px / np.array([W, H], dtype=np.float64)
    return f"{CLASSES.index(cls)} " + " ".join(f"{v:.6f}" for v in np.clip(norm, 0, 1).ravel())


def _table_record(t: Table, T: np.ndarray, W: int, H: int, vector: bool) -> dict[str, Any]:
    src = "vector" if vector else "needs_ocr"
    return {
        "headers": [h if vector else None for h in t.headers],
        "canonical_headers": [c if vector else None for c in t.canonical_headers],
        "rows": [{"cells": [{
            "text": c.text if vector else None, "bbox": to_px_bbox(c.bbox, T, W, H), "text_source": src,
            "row_span": c.row_span, "col_span": c.col_span, "is_header": c.is_header,
        } for c in row]} for row in t.rows],
    }


def _truth_table(t: Table) -> dict[str, Any]:
    return {"headers": t.headers, "canonical_headers": t.canonical_headers,
            "rows": [[{"text": c.text, "row": c.row, "col": c.col, "row_span": c.row_span,
                       "col_span": c.col_span, "is_header": c.is_header} for c in row] for row in t.rows]}


# -- one sheet ---------------------------------------------------------------
@dataclass
class SheetResult:
    """Everything written for one sheet."""

    spec: SheetSpec
    image: bytes
    image_ext: str
    pdf: bytes | None
    labels: str
    gt: dict[str, Any]
    meta: dict[str, Any]


def generate_sheet(index: int, cfg: SynthConfig, vocab: dict, router_cfg: RouterConfig | None = None) -> SheetResult:
    """Generate sheet ``index`` (pure function of ``cfg.seed`` and ``index``)."""
    router_cfg = router_cfg or RouterConfig()
    spec = sample_spec(index, cfg)
    doc, cv, regions, font = build_pdf(spec, cfg, vocab)
    s = spec.dpi / 72.0
    T = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float64)
    true_route = Route.VECTOR if spec.mode == "vector" else Route.RASTER
    decision, words, pdf_bytes, scan_params, router_info = None, [], None, None, None

    if spec.mode in ("vector", "outlined"):
        if spec.mode == "outlined":
            outlined = outline_text(doc, cv)
            doc.close()
            doc = outlined
        doc.subset_fonts()
        pdf_bytes = doc.tobytes(garbage=3, deflate=True)
        # route + render from the serialised bytes: exactly what the pipeline would see
        with fitz.open("pdf", pdf_bytes) as d:
            decision, raw = route_page(d[0], router_cfg)
            gray = cv2.cvtColor(render_page(d[0], spec.dpi), cv2.COLOR_RGB2GRAY)
            if true_route is Route.VECTOR:
                words = words_to_pixels(d[0], raw, spec.dpi)
        agrees = decision.route is true_route
        if not agrees:
            log.warning("sheet %s (%s): router says %s", spec.id, spec.mode, decision.route.value)
        router_info = {"route": decision.route.value, "agrees": agrees, "reasons": decision.reasons}
        image, ext, document = encode_image(gray, "png", spec.dpi), "png", f"{spec.id}.pdf"
    else:
        gray = cv2.cvtColor(render_page(doc[0], spec.dpi), cv2.COLOR_RGB2GRAY)
        gray, M, scan_params = scan_effect(gray, np.random.default_rng([cfg.seed, index, 2]), cfg.scan)
        T = np.vstack([M, [0, 0, 1]]) @ T
        image, ext, document = encode_image(gray, "jpg", spec.dpi, scan_params["jpeg_quality"]), "jpg", f"{spec.id}.jpg"
    doc.close()
    H, W = gray.shape

    kept: list[tuple[Region, np.ndarray]] = []
    for reg in regions:
        p = to_px_polygon(reg.polygon, T, W, H)
        if poly_area(p) < 4.0:
            log.warning("sheet %s: dropping degenerate %s region", spec.id, reg.cls)
            continue
        kept.append((reg, p))

    vector = true_route is Route.VECTOR
    page = PageData(page=1, route=true_route, decision=decision, dpi=spec.dpi,
                    image=np.broadcast_to(np.zeros((1, 1, 3), np.uint8), (H, W, 3)), words=words)
    dets = [Detection(reg.cls, 1.0, [(float(x), float(y)) for x, y in p]) for reg, p in kept]
    prec = page_record(page, dets)
    prec.pop("timings", None)
    for (reg, _), rrec in zip(kept, prec["regions"]):
        if vector:
            rrec["text"] = reg.text
        if reg.table is not None:
            rrec["table"] = _table_record(reg.table, T, W, H, vector)
    gt = {"schema_version": SCHEMA_VERSION, "document": document, "pages": [prec]}
    validate_output(gt)

    labels = "\n".join(yolo_obb_line(reg.cls, p, W, H) for reg, p in kept) + ("\n" if kept else "")
    counts = {c: sum(reg.cls == c for reg, _ in kept) for c in CLASSES}
    meta = {
        **asdict(spec), "font": font.label, "width": W, "height": H,
        "image": f"images/{spec.split}/{spec.id}.{ext}", "label": f"labels/{spec.split}/{spec.id}.txt",
        "gt": f"gt/{spec.id}.json", "pdf": f"pdf/{spec.id}.pdf" if (pdf_bytes and cfg.keeps_pdf(spec.split)) else None,
        "router": router_info, "scan": scan_params, "class_counts": counts,
        "regions": [{"id": f"r{i + 1}", "class": reg.cls, "text": reg.text,
                     "table": _truth_table(reg.table) if reg.table else None} for i, (reg, _) in enumerate(kept)],
    }
    return SheetResult(spec, image, ext, pdf_bytes if cfg.keeps_pdf(spec.split) else None, labels, gt, meta)
