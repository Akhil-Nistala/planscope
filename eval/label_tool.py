"""Helper for annotating the real held-out set (data/real).

Pages are rendered exactly as the pipeline renders them (``src.document.load_pages`` with
configs/inference.yaml), so label coordinates live in the pipeline's pixel frame.

Usage::

    python -m eval.label_tool init   data/real raw/<file>.pdf --page 1     # writes labels/<stem>_p1.json
    python -m eval.label_tool grid   data/real labels/<stem>_p1.json        # preview with a coordinate grid
    python -m eval.label_tool check  data/real labels/<stem>_p1.json        # preview with the labelled boxes
    python -m eval.label_tool crop   data/real labels/<stem>_p1.json --box x0 y0 x1 y1   # zoom (full-res px)

Label file format (one page per file)::

    {"input": "raw/<file>", "page": 1, "attrs": {"lang": "en", "kind": "scanned_pdf"},
     "width": W, "height": H, "dpi": D, "route": "raster",
     "regions": [{"class": "title_block", "box": [x0, y0, x1, y1]},
                 {"class": "text_callout", "polygon": [[x, y], [x, y], [x, y], [x, y]]}]}

``box`` (axis-aligned) or ``polygon`` (oriented, 4 corners) in full-resolution pipeline pixels.
Previews go to ``<root>/previews/`` (gitignored).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from src.config import load_config
from src.detect import CLASSES
from src.document import load_pages

COLORS = {"title_block": (230, 120, 0), "schedule_table": (0, 150, 60), "legend": (150, 60, 200),
          "notes_block": (0, 110, 220), "text_callout": (220, 30, 60)}


def render(root: Path, rel: str, page: int):
    """The pipeline's PageData for page ``page`` (1-based) of ``root/rel``."""
    for p in load_pages(root / rel, load_config()):
        if p.page == page:
            return p
    raise ValueError(f"{rel} has no page {page}")


def label_polygons(lab: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    """(class, (4, 2) polygon) for every labelled region (``box`` or ``polygon``)."""
    out = []
    for r in lab["regions"]:
        if r["class"] not in CLASSES:
            raise ValueError(f"unknown class {r['class']!r}")
        if "polygon" in r:
            out.append((r["class"], np.array(r["polygon"], float).reshape(4, 2)))
        else:
            x0, y0, x1, y1 = r["box"]
            out.append((r["class"], np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)))
    return out


def to_gt_page(lab: dict[str, Any]) -> dict[str, Any]:
    """Minimal schema-like GT page for eval.metrics (class + polygon per region; no text/tables)."""
    return {"page": lab["page"], "route": lab["route"], "dpi": lab["dpi"], "width": lab["width"],
            "height": lab["height"], "words": [],
            "regions": [{"id": f"r{i + 1}", "class": c, "conf": 1.0, "polygon": p.tolist(), "text": None,
                         "text_source": "needs_ocr", "table": None} for i, (c, p) in enumerate(label_polygons(lab))]}


def _preview(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    k = min(1.0, max_side / max(img.shape[:2]))
    return cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img.copy(), k


def draw_grid(img: np.ndarray, k: float, step: int) -> np.ndarray:
    """Grid lines every ``step`` full-res pixels, labelled in full-res units."""
    out = img.copy()
    H, W = out.shape[:2]
    for v in range(0, int(W / k) + 1, step):
        x = int(v * k)
        cv2.line(out, (x, 0), (x, H), (255, 0, 255) if v % (step * 5) == 0 else (255, 180, 255), 1)
        cv2.putText(out, str(v), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 0, 200), 1)
    for v in range(0, int(H / k) + 1, step):
        y = int(v * k)
        cv2.line(out, (0, y), (W, y), (255, 0, 255) if v % (step * 5) == 0 else (255, 180, 255), 1)
        cv2.putText(out, str(v), (2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 0, 200), 1)
    return out


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["init", "grid", "check", "crop"])
    ap.add_argument("root", type=Path)
    ap.add_argument("target", help="raw/<file> for init, labels/<file>.json otherwise")
    ap.add_argument("--page", type=int, default=1)
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("--step", type=int, default=0, help="grid step in full-res px (default: auto)")
    ap.add_argument("--box", type=float, nargs=4, default=None)
    args = ap.parse_args(argv)
    root = args.root
    (root / "previews").mkdir(exist_ok=True)

    if args.cmd == "init":
        p = render(root, args.target, args.page)
        lab = {"input": args.target, "page": args.page, "attrs": {}, "width": p.width, "height": p.height,
               "dpi": p.dpi, "route": p.route.value, "regions": []}
        out = root / "labels" / f"{Path(args.target).stem}_p{args.page}.json"
        out.parent.mkdir(exist_ok=True)
        if out.exists():
            raise SystemExit(f"{out} exists; edit it instead")
        out.write_text(json.dumps(lab, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {out}  ({p.width}x{p.height} px, dpi {p.dpi}, route {p.route.value})")
        return

    lab_path = root / args.target
    lab = json.loads(lab_path.read_text(encoding="utf-8"))
    p = render(root, lab["input"], lab["page"])
    assert (p.width, p.height) == (lab["width"], lab["height"]), "render size differs from label file"
    img = p.image
    stem = lab_path.stem
    if args.cmd == "crop":
        x0, y0, x1, y1 = (int(v) for v in args.box)
        crop = img[max(0, y0):y1, max(0, x0):x1]
        prev, k = _preview(crop, args.max_side)
        step = args.step or max(10, int(round(max(crop.shape[:2]) / 20, -1)))
        prev = draw_grid(prev, k, step)
        # grid labels are relative to the crop; print the offset to add
        out = root / "previews" / f"{stem}_crop_{x0}_{y0}.png"
        cv2.imwrite(str(out), cv2.cvtColor(prev, cv2.COLOR_RGB2BGR))
        print(f"wrote {out}; grid values are crop-relative: add ({x0}, {y0}) for full-res coordinates")
        return
    prev, k = _preview(img, args.max_side)
    if args.cmd == "grid":
        step = args.step or max(50, int(round(max(img.shape[:2]) / 40, -2)))
        prev = draw_grid(prev, k, step)
    else:
        for cls, poly in label_polygons(lab):
            pts = (poly * k).astype(np.int32)
            cv2.polylines(prev, [pts], True, COLORS[cls], 2)
            cv2.putText(prev, cls, (int(pts[0][0]) + 2, int(pts[0][1]) + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        COLORS[cls], 1)
    out = root / "previews" / f"{stem}_{args.cmd}.png"
    cv2.imwrite(str(out), cv2.cvtColor(prev, cv2.COLOR_RGB2BGR))
    print(f"wrote {out}  (full-res {img.shape[1]}x{img.shape[0]}, preview scale {k:.4f})")


if __name__ == "__main__":
    main()
