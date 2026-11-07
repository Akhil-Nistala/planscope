"""End-to-end pipeline: PDF/image -> routed pages -> layout regions -> schema-valid JSON.

Usage::

    python -m src.pipeline --input data/samples --out outputs/
    python -m src.pipeline --input drawing.pdf --out outputs/ --config configs/inference.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .config import PipelineConfig, load_config
from .detect import Detection, Detector, build_detector
from .document import IMAGE_SUFFIXES, PDF_SUFFIXES, PageData, Word, load_pages
from .router import Route
from .schema import SCHEMA_VERSION, validate_output
from .table import recognize_table

log = logging.getLogger(__name__)


def point_in_polygon(x: float, y: float, poly: Sequence[Sequence[float]]) -> bool:
    """Ray-casting point-in-polygon test (works for any simple polygon, incl. rotated boxes)."""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


def words_in_polygon(words: Iterable[Word], poly: Sequence[Sequence[float]]) -> list[Word]:
    """Words whose box centre lies inside ``poly``, in reading order (block, line, x)."""
    hits = [
        w for w in words
        if point_in_polygon((w.bbox[0] + w.bbox[2]) / 2, (w.bbox[1] + w.bbox[3]) / 2, poly)
    ]
    return sorted(hits, key=lambda w: (w.block, w.line, w.bbox[0]))


def _r(v: float) -> float:
    return round(float(v), 2)


def region_record(idx: int, det: Detection, page: PageData, tables: bool = False) -> dict[str, Any]:
    """Build a schema ``region`` from a detection; ``tables`` runs structure recognition on schedules."""
    polygon = [[_r(x), _r(y)] for x, y in det.polygon]
    vector = page.route is Route.VECTOR
    if vector:
        text: str | None = " ".join(w.text for w in words_in_polygon(page.words, det.polygon))
        source = "vector"
    else:
        text, source = None, "needs_ocr"
    table = None
    if tables and det.cls == "schedule_table":
        try:
            table = recognize_table(page.image, det.polygon, page.words if vector else None, dpi=page.dpi)
        except Exception:  # a failed table must not lose the page; the region is still reported
            log.exception("page %d region r%d: table recognition failed", page.page, idx)
    return {
        "id": f"r{idx}",
        "class": det.cls,
        "conf": round(float(det.conf), 4),
        "polygon": polygon,
        "text": text,
        "text_source": source,
        "table": table,
    }


def page_record(page: PageData, detections: list[Detection], tables: bool = False) -> dict[str, Any]:
    """Build a schema ``page`` object (``tables``: recognise schedule-table structure)."""
    return {
        "page": page.page,
        "route": page.route.value,
        "dpi": page.dpi,
        "width": page.width,
        "height": page.height,
        "needs_ocr": page.route is Route.RASTER,
        "routing": page.decision.to_dict() if page.decision else None,
        "words": [
            {"text": w.text, "bbox": [_r(v) for v in w.bbox], "block": w.block, "line": w.line}
            for w in page.words
        ],
        "regions": [region_record(i + 1, d, page, tables) for i, d in enumerate(detections)],
        "timings": {k: round(v, 4) for k, v in page.timings.items()},
    }


def process_document(
    path: str | Path,
    cfg: PipelineConfig,
    detector: Detector | None = None,
    keep_images: bool = False,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    """Run the pipeline on one file.

    Returns the validated output dict and (if ``keep_images``) the rendered page images.
    """
    detector = detector or build_detector(cfg.detector)
    pages, images = [], []
    for page in load_pages(path, cfg):
        t0 = time.perf_counter()
        dets = detector.detect(page.image, page.dpi)
        page.timings["detect_s"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        rec = page_record(page, dets, tables=cfg.tables.enabled)
        rec["timings"]["regions_s"] = round(time.perf_counter() - t1, 4)
        pages.append(rec)
        if keep_images:
            images.append(page.image)
    doc = {"schema_version": SCHEMA_VERSION, "document": Path(path).name, "pages": pages}
    validate_output(doc)
    return doc, images


def _collect_inputs(inp: Path) -> list[Path]:
    if inp.is_file():
        return [inp]
    exts = PDF_SUFFIXES | IMAGE_SUFFIXES
    return sorted(p for p in inp.iterdir() if p.suffix.lower() in exts)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="PDF/image file or directory")
    ap.add_argument("--out", required=True, type=Path, help="output directory for JSON files")
    ap.add_argument("--config", type=Path, default=None, help="YAML config (default configs/inference.yaml)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    detector = build_detector(cfg.detector)
    args.out.mkdir(parents=True, exist_ok=True)

    for f in _collect_inputs(args.input):
        t0 = time.perf_counter()
        doc, _ = process_document(f, cfg, detector)
        dt = time.perf_counter() - t0
        (args.out / f"{f.stem}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        routes = ",".join(p["route"] for p in doc["pages"])
        print(f"{f.name:40s} pages={len(doc['pages']):3d} routes=[{routes}] "
              f"words={sum(len(p['words']) for p in doc['pages']):5d} {len(doc['pages']) / dt:6.2f} pages/s")


if __name__ == "__main__":
    main()
