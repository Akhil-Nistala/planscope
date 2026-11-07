"""Error analysis for an evaluation run: worst pages drawn as GT-vs-prediction overlays + error tallies.

Usage::

    python -m eval.errors --run eval/results/tiles_640 --data data/synth --n 6
    python -m eval.errors --run eval/results/real_tiles_640 --real data/real --n 6

Writes ``<run>/errors/<key>.png`` (GT thin green, matched predictions blue, false positives red,
missed GT orange) and ``<run>/errors/summary.json`` with per-class counts of false positives, misses,
class confusions and size statistics of missed objects.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image

from src.geometry import area, as_poly, iou

from .evaluate import real_items, synth_items


def classify_errors(gt: dict, pred: dict, conf: float, thr: float = 0.5) -> dict:
    """Per-page TP / FP / FN / confusion lists (class-aware matching first, then class-agnostic)."""
    s = gt["width"] / pred["width"]
    G = [(as_poly(r["polygon"]), r["class"]) for r in gt["regions"]]
    P = sorted([(as_poly(r["polygon"]) * s, r["class"], r["conf"]) for r in pred["regions"] if r["conf"] >= conf],
               key=lambda t: -t[2])
    used_g, tp, fp, conf_pairs = set(), [], [], []
    for poly, cls, c in P:
        cands = [(iou(poly, g), j) for j, (g, gc) in enumerate(G) if j not in used_g and gc == cls]
        best = max(cands, default=(0, -1))
        if best[0] >= thr:
            used_g.add(best[1])
            tp.append((poly, cls))
            continue
        other = max([(iou(poly, g), j) for j, (g, _) in enumerate(G) if j not in used_g], default=(0, -1))
        if other[0] >= thr:
            used_g.add(other[1])
            conf_pairs.append((poly, G[other[1]][1], cls))
        else:
            fp.append((poly, cls))
    fn = [(g, c) for j, (g, c) in enumerate(G) if j not in used_g]
    return {"tp": tp, "fp": fp, "fn": fn, "confused": conf_pairs, "gt": G}


def draw(img: np.ndarray, e: dict, k: float) -> np.ndarray:
    """Overlay one page's errors on a downscaled copy."""
    out = img.copy()

    def poly(p, color, t):
        cv2.polylines(out, [(p * k).astype(np.int32)], True, color, t)

    for g, _ in e["gt"]:
        poly(g, (0, 170, 0), 1)
    for p, _ in e["tp"]:
        poly(p, (40, 90, 255), 2)
    for p, cls in e["fp"]:
        poly(p, (230, 0, 0), 2)
        cv2.putText(out, f"FP {cls}", tuple((p[0] * k).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (230, 0, 0), 1)
    for g, cls in e["fn"]:
        poly(g, (255, 140, 0), 3)
        cv2.putText(out, f"MISS {cls}", tuple((g[0] * k).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 140, 0), 1)
    for p, gcls, pcls in e["confused"]:
        poly(p, (200, 0, 200), 2)
        cv2.putText(out, f"{gcls}->{pcls}", tuple((p[0] * k).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (200, 0, 200), 1)
    return out


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=Path)
    src.add_argument("--real", type=Path)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=6, help="worst pages to draw")
    ap.add_argument("--max-side", type=int, default=1800)
    args = ap.parse_args(argv)

    metrics = json.loads((args.run / "metrics.json").read_text(encoding="utf-8"))
    conf = metrics["settings"]["conf"]
    items = list(real_items(args.real, None) if args.real else synth_items(args.data, args.split, None))
    out = args.run / "errors"
    out.mkdir(exist_ok=True)
    tallies: dict[str, Counter] = defaultdict(Counter)
    missed_area: dict[str, list[float]] = defaultdict(list)
    found_area: dict[str, list[float]] = defaultdict(list)
    per_page = []
    for key, src_path, page_idx, gt, attrs in items:
        pf = args.run / "pred" / f"{key}.json"
        if not pf.exists():
            continue
        pred = json.loads(pf.read_text(encoding="utf-8"))["pages"][page_idx]
        e = classify_errors(gt, pred, conf)
        page_area = gt["width"] * gt["height"]
        for _, c in e["fp"]:
            tallies[c]["false_positive"] += 1
        for g, c in e["fn"]:
            tallies[c]["missed"] += 1
            missed_area[c].append(area(g) / page_area)
        for p, c in e["tp"]:
            tallies[c]["correct"] += 1
            found_area[c].append(area(p) / page_area)
        for _, gc, pc in e["confused"]:
            tallies[gc][f"labelled_as_{pc}"] += 1
        per_page.append((len(e["fp"]) + len(e["fn"]) + len(e["confused"]), key, src_path, page_idx, gt, pred, e))

    per_page.sort(key=lambda t: -t[0])
    from src.config import load_config
    from src.document import load_pages
    cfg = load_config()
    for n_err, key, src_path, page_idx, gt, pred, e in per_page[: args.n]:
        page = next(p for i, p in enumerate(load_pages(src_path, cfg)) if i == page_idx)
        img = page.image
        if img.shape[1] != gt["width"]:  # draw in the GT frame
            img = cv2.resize(img, (gt["width"], gt["height"]), interpolation=cv2.INTER_AREA)
        k = min(1.0, args.max_side / max(img.shape[:2]))
        small = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        Image.fromarray(draw(small, e, k)).save(out / f"{key}.png")

    def med(v: list[float]) -> float | None:
        return round(float(np.median(v)) * 100, 4) if v else None

    summary = {"conf": conf, "pages": len(per_page),
               "worst_pages": [{"key": k, "errors": n} for n, k, *_ in per_page[: args.n]],
               "per_class": {c: dict(t) for c, t in sorted(tallies.items())},
               "median_area_pct_of_page": {c: {"missed": med(missed_area[c]), "found": med(found_area[c])}
                                           for c in sorted(set(missed_area) | set(found_area))}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
