"""Detection and table-structure metrics on PlanScope JSON documents (schema in src/schema.json).

Detection: oriented-box IoU (exact polygon intersection), per-class average precision (all-point
interpolated, VOC/COCO style) at IoU 0.5 and averaged over 0.5:0.95, plus precision / recall / F1 at
the configured confidence threshold.

Tables: for each GT ``schedule_table`` matched to a predicted one (IoU >= 0.5), cells are matched by
box IoU >= 0.5; we report cell precision / recall / F1, span accuracy of matched cells, text exact-match
rate of matched cells (vector pages), and canonical-header accuracy.
"""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from src.detect import CLASSES
from src.geometry import as_poly, iou

IOU_THRESHOLDS = tuple(np.round(np.arange(0.5, 0.96, 0.05), 2))


@dataclass
class PagePair:
    """GT page and predicted page of one sheet, predictions rescaled to the GT pixel frame."""

    key: str
    gt: dict[str, Any]
    pred: dict[str, Any]
    attrs: dict[str, str] = field(default_factory=dict)

    @property
    def scale(self) -> float:
        """Factor taking predicted pixel coordinates to GT pixel coordinates."""
        return self.gt["width"] / self.pred["width"]


def _polys(regions: Iterable[dict], cls: str, scale: float = 1.0) -> list[tuple[np.ndarray, float, dict]]:
    return [(as_poly(r["polygon"]) * scale, float(r.get("conf", 1.0)), r) for r in regions if r["class"] == cls]


def match_page(gt: list, pred: list, thr: float) -> tuple[list[tuple[float, bool]], int, list[tuple[int, int]]]:
    """Greedy (by confidence) one-to-one matching. Returns [(conf, is_tp)], n_gt, [(pred_i, gt_i)]."""
    order = sorted(range(len(pred)), key=lambda i: -pred[i][1])
    used: set[int] = set()
    scored, pairs = [], []
    for i in order:
        best, best_j = thr, -1
        for j, (g, _, _) in enumerate(gt):
            if j in used:
                continue
            v = iou(pred[i][0], g)
            if v >= best:
                best, best_j = v, j
        if best_j >= 0:
            used.add(best_j)
            pairs.append((i, best_j))
        scored.append((pred[i][1], best_j >= 0))
    return scored, len(gt), pairs


def average_precision(scored: list[tuple[float, bool]], n_gt: int) -> float:
    """All-point interpolated AP from (confidence, is_tp) over all images."""
    if n_gt == 0:
        return float("nan")
    if not scored:
        return 0.0
    s = sorted(scored, key=lambda t: -t[0])
    tp = np.cumsum([t[1] for t in s])
    fp = np.cumsum([not t[1] for t in s])
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-9)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[1.0], prec, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def detection_metrics(pairs: list[PagePair], conf_thr: float = 0.25) -> dict[str, Any]:
    """Per-class and mean AP50 / AP50:95 and P/R/F1 at ``conf_thr``."""
    per_class: dict[str, dict[str, float]] = {}
    for cls in CLASSES:
        aps = []
        for thr in IOU_THRESHOLDS:
            scored, n_gt = [], 0
            for p in pairs:
                s, n, _ = match_page(_polys(p.gt["regions"], cls), _polys(p.pred["regions"], cls, p.scale), thr)
                scored += s
                n_gt += n
            aps.append(average_precision(scored, n_gt))
            if thr == 0.5:
                tp = sum(1 for c, t in scored if t and c >= conf_thr)
                fp = sum(1 for c, t in scored if not t and c >= conf_thr)
                n_gt50 = n_gt
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / n_gt50 if n_gt50 else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) and n_gt50 else 0.0
        per_class[cls] = {"n_gt": n_gt50, "tp": tp, "fp": fp, "fn": n_gt50 - tp, "precision": prec, "recall": rec,
                          "f1": f1, "ap50": aps[0], "ap50_95": float(np.nanmean(aps)) if n_gt50 else float("nan")}
    present = [c for c in CLASSES if per_class[c]["n_gt"]]
    return {"per_class": per_class,
            "map50": float(np.mean([per_class[c]["ap50"] for c in present])) if present else float("nan"),
            "map50_95": float(np.mean([per_class[c]["ap50_95"] for c in present])) if present else float("nan"),
            "n_pages": len(pairs)}


def confusion(pairs: list[PagePair], thr: float = 0.5, conf_thr: float = 0.25) -> dict[str, dict[str, int]]:
    """Class confusion at IoU >= ``thr``, ignoring class: rows = GT class (+ 'background' for FPs),
    columns = predicted class (+ 'missed' for FNs)."""
    cols = list(CLASSES) + ["missed"]
    m = {r: {c: 0 for c in cols} for r in list(CLASSES) + ["background"]}
    for p in pairs:
        gt = [(as_poly(r["polygon"]), r["class"]) for r in p.gt["regions"]]
        pred = sorted([(as_poly(r["polygon"]) * p.scale, r["class"], r["conf"]) for r in p.pred["regions"]
                       if r["conf"] >= conf_thr], key=lambda t: -t[2])
        used: set[int] = set()
        for poly, cls, _ in pred:
            best, bj = thr, -1
            for j, (g, _) in enumerate(gt):
                if j not in used:
                    v = iou(poly, g)
                    if v >= best:
                        best, bj = v, j
            if bj >= 0:
                used.add(bj)
                m[gt[bj][1]][cls] += 1
            else:
                m["background"][cls] += 1
        for j, (_, gcls) in enumerate(gt):
            if j not in used:
                m[gcls]["missed"] += 1
    return m


def _norm(s: str | None) -> str:
    """Text normalisation for exact-match: NFKC, unify dashes, drop whitespace."""
    s = unicodedata.normalize("NFKC", s or "")
    s = s.translate(str.maketrans({c: "-" for c in "‐‑‒–—−"}))
    return "".join(s.split())


def _bbox_poly(b: list[float]) -> np.ndarray:
    return as_poly([(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])])


def table_metrics(pairs: list[PagePair], thr: float = 0.5) -> dict[str, Any]:
    """Cell-level structure and text metrics over matched schedule tables."""
    agg = defaultdict(float)
    for p in pairs:
        gts = _polys(p.gt["regions"], "schedule_table")
        preds = _polys(p.pred["regions"], "schedule_table", p.scale)
        agg["gt_tables"] += len(gts)
        _, _, matches = match_page(gts, preds, thr)
        for pi, gi in matches:
            gt_t, pr_t = gts[gi][2]["table"], preds[pi][2]["table"]
            agg["matched_tables"] += 1
            if not gt_t or not pr_t:
                agg["gt_cells"] += sum(len(r["cells"]) for r in gt_t["rows"]) if gt_t else 0
                continue
            agg["tables_with_structure"] += 1
            gcells = [c for r in gt_t["rows"] for c in r["cells"]]
            pcells = [c for r in pr_t["rows"] for c in r["cells"]]
            agg["gt_cells"] += len(gcells)
            agg["pred_cells"] += len(pcells)
            used: set[int] = set()
            table_tp = 0
            for pc in pcells:
                pp = _bbox_poly(pc["bbox"]) * p.scale
                best, bj = thr, -1
                for j, gc in enumerate(gcells):
                    if j not in used:
                        v = iou(pp, _bbox_poly(gc["bbox"]))
                        if v >= best:
                            best, bj = v, j
                if bj < 0:
                    continue
                used.add(bj)
                gc = gcells[bj]
                agg["cell_tp"] += 1
                table_tp += 1
                agg["span_ok"] += (pc.get("row_span", 1), pc.get("col_span", 1)) == (gc.get("row_span", 1),
                                                                                      gc.get("col_span", 1))
                if gc["text"] is not None and pc["text"] is not None:
                    agg["text_pairs"] += 1
                    agg["text_ok"] += _norm(pc["text"]) == _norm(gc["text"])
            gh, ph = gt_t["canonical_headers"], pr_t["canonical_headers"]
            if any(h is not None for h in gh) and len(gh) == len(ph):
                agg["header_tables"] += 1
                agg["header_ok"] += gh == ph
            agg["exact_grid"] += table_tp == len(gcells) == len(pcells)  # every cell recovered, nothing extra

    def r(a: str, b: str) -> float:
        return agg[a] / agg[b] if agg[b] else float("nan")

    prec, rec = r("cell_tp", "pred_cells"), r("cell_tp", "gt_cells")
    return {
        "gt_tables": int(agg["gt_tables"]), "matched_tables": int(agg["matched_tables"]),
        "tables_with_structure": int(agg["tables_with_structure"]),
        "cell_precision": prec, "cell_recall": rec,
        "cell_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "span_accuracy": r("span_ok", "cell_tp"), "text_exact": r("text_ok", "text_pairs"),
        "header_accuracy": r("header_ok", "header_tables"), "exact_grid_rate": r("exact_grid", "matched_tables"),
    }
