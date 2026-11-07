"""Cut generated sheets into overlapping fixed-DPI tiles with clipped OBB labels (YOLO-OBB format).

Usage::

    python -m training.tile_dataset --src data/synth --out data/synth_tiles --tile 640 --max-tiles 6

Sheets are first resampled to ``--dpi`` (default 150, the detector's ``detect_dpi``) so every tile
shows drawings at the scale used at inference. Each label is clipped to the tile; it is kept if at
least ``--min-visible`` of it is inside, or if the visible part covers a large share of the tile
(a big table seen through a small window). Kept pieces are re-boxed with a minimum-area rectangle.
``--max-tiles`` samples tiles per sheet (all tiles of an A1 sheet would dominate a CPU-sized budget);
a fraction of label-free tiles is kept as background.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import yaml
from PIL import Image

from src.detect import CLASSES
from src.geometry import area, clip_to_rect, min_area_box
from src.tiling import tile_grid


def read_labels(path: Path, W: int, H: int) -> list[tuple[int, np.ndarray]]:
    """YOLO-OBB label file -> [(class, (4, 2) pixel polygon)]."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        v = line.split()
        if len(v) == 9:
            out.append((int(v[0]), np.array(v[1:], float).reshape(4, 2) * [W, H]))
    return out


def tile_labels(labels: list[tuple[int, np.ndarray]], rect: tuple[int, int, int, int], min_visible: float,
                min_tile_cover: float = 0.25, min_px: float = 16.0) -> list[tuple[int, np.ndarray]]:
    """Clip sheet labels to one tile; returns polygons in tile pixel coordinates."""
    x0, y0, x1, y1 = rect
    tile_area = (x1 - x0) * (y1 - y0)
    out = []
    for cls, poly in labels:
        full = area(poly)
        clipped = clip_to_rect(poly, x0, y0, x1, y1)
        if len(clipped) < 3 or full <= 0:
            continue
        vis = area(clipped)
        if vis < min_px or (vis < min_visible * full and vis < min_tile_cover * tile_area):
            continue
        box = min_area_box(clipped) - [x0, y0]
        box[:, 0] = np.clip(box[:, 0], 0, x1 - x0)
        box[:, 1] = np.clip(box[:, 1], 0, y1 - y0)
        out.append((cls, box))
    return out


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("data/synth"))
    ap.add_argument("--out", type=Path, default=Path("data/synth_tiles"))
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--max-tiles", type=int, default=6, help="tiles sampled per sheet (0 = all)")
    ap.add_argument("--empty-frac", type=float, default=0.1, help="share of kept tiles that may be label-free")
    ap.add_argument("--min-visible", type=float, default=0.5)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    if args.out.exists():
        shutil.rmtree(args.out)
    manifest = [json.loads(ln) for ln in (args.src / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    counts: dict[str, Counter] = {s: Counter() for s in args.splits}
    n_tiles = Counter()
    for split in args.splits:
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for rec in manifest:
        split = rec["split"]
        if split not in args.splits:
            continue
        img = np.asarray(Image.open(args.src / rec["image"]).convert("L"))
        H0, W0 = img.shape
        labels = read_labels(args.src / rec["label"], W0, H0)
        s = args.dpi / rec["dpi"]
        if abs(s - 1) > 1e-3:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
            labels = [(c, p * s) for c, p in labels]
        H, W = img.shape
        tiles = [(r, tile_labels(labels, r, args.min_visible)) for r in tile_grid(W, H, args.tile, args.overlap)]
        full = [t for t in tiles if t[1]]
        empty = [t for t in tiles if not t[1]]
        k = len(full) if args.max_tiles <= 0 else min(args.max_tiles, len(full))
        chosen = [full[i] for i in rng.permutation(len(full))[:k]]
        n_empty = int(round(args.empty_frac * len(chosen)))
        chosen += [empty[i] for i in rng.permutation(len(empty))[:n_empty]]
        ext = Path(rec["image"]).suffix
        for (x0, y0, x1, y1), labs in chosen:
            name = f"{rec['id']}_{x0}_{y0}"
            crop = Image.fromarray(img[y0:y1, x0:x1])
            if ext == ".jpg":
                crop.save(args.out / "images" / split / f"{name}.jpg", quality=95)
            else:
                crop.save(args.out / "images" / split / f"{name}.png", compress_level=3)
            tw, th = x1 - x0, y1 - y0
            lines = [f"{c} " + " ".join(f"{v:.6f}" for v in (p / [tw, th]).ravel()) for c, p in labs]
            (args.out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                                                     encoding="utf-8")
            counts[split].update(CLASSES[c] for c, _ in labs)
            n_tiles[split] += 1

    data = {"path": args.out.resolve().as_posix(), "train": "images/train", "val": "images/val",
            "names": dict(enumerate(CLASSES))}
    (args.out / "dataset.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    summary = {"args": {k: str(v) for k, v in vars(args).items()}, "tiles": dict(n_tiles),
               "instances": {s: dict(c) for s, c in counts.items()}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["tiles"]), json.dumps(summary["instances"]))


if __name__ == "__main__":
    main()
