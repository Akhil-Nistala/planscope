"""Draw labels over generated images to check them by eye.

Usage::

    python -m data_gen.preview --root data/synth --n 8            # first 8 sheets
    python -m data_gen.preview --root data/synth --ids 00003 00017 --cells

Writes ``<root>/previews/<id>.png`` (downscaled): OBB polygons from the YOLO label file, coloured
by class, plus (``--cells``) the table cell boxes from the GT JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

from src.detect import CLASSES

COLORS = {  # RGB (the preview image is RGB, so cv2 draws these as-is)
    "title_block": (230, 120, 0), "schedule_table": (0, 150, 60), "legend": (150, 60, 200),
    "notes_block": (0, 110, 220), "text_callout": (220, 30, 60),
}


def render_preview(root: Path, sheet_id: str, max_side: int = 2400, cells: bool = False) -> np.ndarray:
    """RGB preview image for one sheet."""
    meta = json.loads((root / "meta" / f"{sheet_id}.json").read_text(encoding="utf-8"))
    img = np.asarray(Image.open(root / meta["image"]).convert("RGB")).copy()
    H, W = img.shape[:2]
    k = min(1.0, max_side / max(H, W))
    img = cv2.resize(img, (int(W * k), int(H * k)), interpolation=cv2.INTER_AREA)
    if cells:
        gt = json.loads((root / meta["gt"]).read_text(encoding="utf-8"))
        for reg in gt["pages"][0]["regions"]:
            for row in (reg["table"] or {"rows": []})["rows"]:
                for c in row["cells"]:
                    x0, y0, x1, y1 = (np.array(c["bbox"]) * k).astype(int)
                    cv2.rectangle(img, (x0, y0), (x1, y1), (255, 170, 0), 1)
    for line in (root / meta["label"]).read_text(encoding="utf-8").splitlines():
        v = line.split()
        cls = CLASSES[int(v[0])]
        pts = (np.array(v[1:], float).reshape(4, 2) * [W * k, H * k]).astype(np.int32)
        cv2.polylines(img, [pts], True, COLORS[cls], 2)
        cv2.circle(img, tuple(int(c) for c in pts[0]), 4, COLORS[cls], -1)  # first corner
    return img


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-side", type=int, default=2400)
    ap.add_argument("--cells", action="store_true", help="also draw table cell boxes from the GT JSON")
    args = ap.parse_args(argv)
    ids = args.ids or sorted(p.stem for p in (args.root / "meta").glob("*.json"))[: args.n]
    out = args.root / "previews"
    out.mkdir(exist_ok=True)
    for sid in ids:
        Image.fromarray(render_preview(args.root, sid, args.max_side, args.cells)).save(out / f"{sid}.png")
    print(f"wrote {len(ids)} previews to {out}")


if __name__ == "__main__":
    main()
