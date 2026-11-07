"""End-to-end evaluation: run the real pipeline on held-out sheets and score its JSON against GT.

Usage::

    # synthetic test split (vector/outlined sheets run from their PDFs, scans from their JPEGs)
    python -m eval.evaluate --data data/synth --split test --weights models/tiles_640.pt --name tiles_640
    python -m eval.evaluate ... --no-tiling            # ablation: one full-sheet pass only
    python -m eval.evaluate ... --no-full-pass         # ablation: tiles only
    python -m eval.evaluate --data data/synth --split test --oracle --name oracle   # GT boxes -> tables only
    # real held-out set (labels in data/real/labels/*.json, see data/real/README.md)
    python -m eval.evaluate --real data/real --weights models/tiles_640.pt --name real_tiles_640

Writes ``<out>/<name>/metrics.json``, ``report.md`` and ``pred/<key>.json`` (pipeline outputs).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from src.config import PipelineConfig, load_config
from src.detect import CLASSES, Detection, build_detector
from src.pipeline import process_document

from .metrics import PagePair, confusion, detection_metrics, table_metrics

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
log = logging.getLogger(__name__)


class OracleDetector:
    """Returns the GT regions (scaled to the rendered page): isolates the post-detection stages."""

    def __init__(self, gt_page: dict[str, Any]):
        self.gt = gt_page

    def detect(self, image: np.ndarray, dpi: int | None = None) -> list[Detection]:  # noqa: ARG002
        s = image.shape[1] / self.gt["width"]
        return [Detection(r["class"], 1.0, [(x * s, y * s) for x, y in r["polygon"]]) for r in self.gt["regions"]]


def synth_items(root: Path, split: str, limit: int | None) -> Iterator[tuple[str, Path, int, dict, dict]]:
    """(key, input file, page index, GT page, attrs) for a generated split."""
    recs = [json.loads(ln) for ln in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    recs = [r for r in recs if r["split"] == split][: limit or None]
    for r in recs:
        gt = json.loads((root / r["gt"]).read_text(encoding="utf-8"))["pages"][0]
        src = root / (r["pdf"] or r["image"])
        attrs = {"mode": r["mode"], "lang": r["lang"], "sheet_size": r["sheet_size"], "layout": r["layout"],
                 "input": "pdf" if r["pdf"] else "image"}
        yield r["id"], src, 0, gt, attrs


def real_items(root: Path, limit: int | None) -> Iterator[tuple[str, Path, int, dict, dict]]:
    """Items of the real held-out set: one label file per page (see data/real/README.md)."""
    from .label_tool import to_gt_page

    for f in sorted((root / "labels").glob("*.json"))[: limit or None]:
        lab = json.loads(f.read_text(encoding="utf-8"))
        attrs = {"lang": lab.get("attrs", {}).get("lang", "?"), "kind": lab.get("attrs", {}).get("kind", "?")}
        yield f.stem, root / lab["input"], int(lab.get("page", 1)) - 1, to_gt_page(lab), attrs


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return "–" if math.isnan(v) else f"{v:.3f}"
    return str(v)


def breakdown(pairs: list[PagePair], attr: str, conf: float) -> dict[str, Any]:
    """Detection metrics per value of a sheet attribute."""
    out = {}
    for val in sorted({p.attrs.get(attr, "?") for p in pairs}):
        sub = [p for p in pairs if p.attrs.get(attr, "?") == val]
        m = detection_metrics(sub, conf)
        out[val] = {"n_pages": len(sub), "map50": m["map50"], "map50_95": m["map50_95"],
                    "callout_recall": m["per_class"]["text_callout"]["recall"]}
    return out


def report_md(name: str, m: dict[str, Any]) -> str:
    """Human-readable markdown summary of metrics.json."""
    d = m["detection"]
    lines = [f"# Evaluation: {name}", "", f"- pages: {d['n_pages']}  ·  settings: `{json.dumps(m['settings'])}`",
             f"- **mAP50 {_fmt(d['map50'])}**  ·  mAP50:95 {_fmt(d['map50_95'])}  "
             f"·  median page time {_fmt(m['timing']['median_page_s'])} s", "",
             "| class | GT | P | R | F1 | AP50 | AP50:95 |", "|---|---|---|---|---|---|---|"]
    for c in CLASSES:
        v = d["per_class"][c]
        lines.append(f"| {c} | {v['n_gt']} | {_fmt(v['precision'])} | {_fmt(v['recall'])} | {_fmt(v['f1'])} | "
                     f"{_fmt(v['ap50'])} | {_fmt(v['ap50_95'])} |")
    for attr, rows in m["breakdown"].items():
        lines += ["", f"**By {attr}**", "", "| value | pages | mAP50 | mAP50:95 | callout recall |", "|---|---|---|---|---|"]
        lines += [f"| {k} | {v['n_pages']} | {_fmt(v['map50'])} | {_fmt(v['map50_95'])} | {_fmt(v['callout_recall'])} |"
                  for k, v in rows.items()]
    t = m["tables"]
    lines += ["", "**Tables** (matched GT schedules, IoU ≥ 0.5)", "",
              "| matched / GT | cell P | cell R | cell F1 | span acc | text exact | header acc | exact grid |",
              "|---|---|---|---|---|---|---|---|",
              f"| {t['matched_tables']} / {t['gt_tables']} | {_fmt(t['cell_precision'])} | {_fmt(t['cell_recall'])} | "
              f"{_fmt(t['cell_f1'])} | {_fmt(t['span_accuracy'])} | {_fmt(t['text_exact'])} | "
              f"{_fmt(t['header_accuracy'])} | {_fmt(t['exact_grid_rate'])} |"]
    cm = m["confusion"]
    cols = list(CLASSES) + ["missed"]
    lines += ["", "**Confusion** (rows GT, columns predicted, IoU ≥ 0.5, class-agnostic matching)", "",
              "| GT \\ pred | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    lines += [f"| {r} | " + " | ".join(str(cm[r][c]) for c in cols) + " |" for r in cm]
    return "\n".join(lines) + "\n"


def run_eval(items: list, cfg: PipelineConfig, out: Path, oracle: bool) -> dict[str, Any]:
    """Run the pipeline on every item and compute all metrics."""
    detector = None if oracle else build_detector(cfg.detector)
    (out / "pred").mkdir(parents=True, exist_ok=True)
    pairs, times, failures = [], [], []
    for i, (key, src, page_idx, gt, attrs) in enumerate(items):
        t0 = time.perf_counter()
        try:
            doc, _ = process_document(src, cfg, OracleDetector(gt) if oracle else detector)
        except Exception as e:
            log.exception("%s failed", key)
            failures.append({"key": key, "error": f"{type(e).__name__}: {e}"})
            continue
        times.append(time.perf_counter() - t0)
        pred = doc["pages"][page_idx]
        (out / "pred" / f"{key}.json").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        pairs.append(PagePair(key, gt, pred, attrs))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(items)} pages  {np.mean(times):.1f} s/page", flush=True)
    conf = cfg.detector.conf
    return {
        "detection": detection_metrics(pairs, conf),
        "breakdown": {a: breakdown(pairs, a, conf) for a in ("mode", "lang", "sheet_size", "layout", "kind")
                      if any(a in p.attrs for p in pairs)},
        "confusion": confusion(pairs, 0.5, conf),
        "tables": table_metrics(pairs),
        "timing": {"median_page_s": float(np.median(times)) if times else float("nan"),
                   "mean_page_s": float(np.mean(times)) if times else float("nan")},
        "failures": failures,
    }


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=Path, help="generated dataset root (data/synth)")
    src.add_argument("--real", type=Path, help="real held-out set root (data/real)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--name", required=True)
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--oracle", action="store_true", help="use GT boxes instead of a detector")
    ap.add_argument("--no-tiling", action="store_true")
    ap.add_argument("--no-full-pass", action="store_true")
    ap.add_argument("--no-tables", action="store_true")
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None, help="inference size for untiled / full-sheet passes")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("eval/results"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    if not args.oracle:
        cfg.detector.kind = "yolo_obb"
        if args.weights:
            cfg.detector.weights = args.weights
    cfg.detector.tiling.enabled = not args.no_tiling
    cfg.detector.tiling.full_pass = not args.no_full_pass
    cfg.tables.enabled = not args.no_tables
    if args.tile:
        cfg.detector.tiling.tile = args.tile
    if args.imgsz:
        cfg.detector.imgsz = cfg.detector.tiling.full_imgsz = args.imgsz
    if args.conf is not None:
        cfg.detector.conf = args.conf

    items = list(real_items(args.real, args.limit) if args.real else synth_items(args.data, args.split, args.limit))
    out = args.out / args.name
    print(f"evaluating {len(items)} pages -> {out}")
    m = run_eval(items, cfg, out, args.oracle)
    m["settings"] = {"weights": None if args.oracle else cfg.detector.weights, "oracle": args.oracle,
                     "tiling": cfg.detector.tiling.enabled, "full_pass": cfg.detector.tiling.full_pass,
                     "tile": cfg.detector.tiling.tile, "imgsz": cfg.detector.imgsz,
                     "full_imgsz": cfg.detector.tiling.full_imgsz,
                     "conf": cfg.detector.conf, "detect_dpi": cfg.detector.detect_dpi,
                     "tables": cfg.tables.enabled, "data": str(args.real or args.data),
                     "split": None if args.real else args.split}
    (out / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "report.md").write_text(report_md(args.name, m), encoding="utf-8")
    print(report_md(args.name, m))


if __name__ == "__main__":
    main()
