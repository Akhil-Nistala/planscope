"""Milestones 3-5: geometry, tiled detection + merging (fake model), table structure, metrics, evaluation."""

from __future__ import annotations

import copy
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from data_gen.config import SynthConfig, load_vocab
from data_gen.generate import run as generate
from data_gen.sheet import generate_sheet
from eval.evaluate import OracleDetector, run_eval, synth_items
from eval.label_tool import to_gt_page
from eval.metrics import PagePair, average_precision, detection_metrics, table_metrics
from src.config import DetectorConfig, PipelineConfig
from src.detect import YoloObbDetector
from src.geometry import canonical_corners, clip_to_rect, iou, ios, min_area_box, signed_area
from src.pipeline import process_document
from src.schema import validate_output
from src.table import canonical_header, recognize_table
from src.tiling import OBB, merge_obbs, tile_grid, tile_origins

BOX = np.array([[10, 10], [110, 10], [110, 60], [10, 60]], float)


# -- geometry ------------------------------------------------------------------
def test_iou_ios_and_clipping() -> None:
    assert iou(BOX, BOX) == pytest.approx(1.0)
    shifted = BOX + [50, 0]
    assert iou(BOX, shifted) == pytest.approx(50 * 50 / (2 * 100 * 50 - 50 * 50))
    small = np.array([[20, 20], [40, 20], [40, 40], [20, 40]], float)
    assert ios(BOX, small) == pytest.approx(1.0)
    clipped = clip_to_rect(BOX, 0, 0, 60, 100)
    assert clipped[:, 0].max() == pytest.approx(60) and abs(signed_area(clipped)) == pytest.approx(50 * 50)


def test_canonical_corners_clockwise_from_top_left() -> None:
    for start in range(4):
        for order in (1, -1):
            p = np.roll(BOX[::order], start, axis=0)
            c = canonical_corners(p)
            assert signed_area(c) > 0 and tuple(c[0]) == (10, 10)
    rot = min_area_box(np.array([[0, 0], [100, 20], [96, 40], [-4, 20]], float))  # ~11 deg box
    assert signed_area(rot) > 0 and rot[0][0] < rot[1][0]


# -- tiling + merging -----------------------------------------------------------
def test_tile_grid_covers_image() -> None:
    assert tile_origins(500, 640, 0.25) == [0]
    xs = tile_origins(2000, 640, 0.25)
    assert xs[0] == 0 and xs[-1] == 2000 - 640 and all(b - a <= 480 for a, b in zip(xs, xs[1:]))
    tiles = tile_grid(2000, 1000, 640, 0.25)
    covered = np.zeros((1000, 2000), bool)
    for x0, y0, x1, y1 in tiles:
        covered[y0:y1, x0:x1] = True
    assert covered.all()


def test_merge_joins_pieces_and_drops_duplicates() -> None:
    left = OBB(1, 0.9, np.array([[0, 0], [60, 0], [60, 50], [0, 50]], float))
    right = OBB(1, 0.8, np.array([[40, 0], [100, 0], [100, 50], [40, 50]], float))  # overlaps in 40..60
    dup = OBB(1, 0.5, left.poly + 1)
    other = OBB(2, 0.7, left.poly.copy())  # different class: kept separately
    out = merge_obbs([left, right, dup, other], iou_thr=0.5, ios_thr=0.3)
    by_cls = {d.cls: d for d in out}
    assert len(out) == 2
    xs = by_cls[1].poly[:, 0]
    assert xs.min() == pytest.approx(0, abs=1) and xs.max() == pytest.approx(100, abs=1)
    assert by_cls[1].conf == pytest.approx(0.9)


def _fake_detector(tiling: bool, full_pass: bool, tile: int = 400) -> YoloObbDetector:
    """Detector whose 'model' sees one 300x100 object at page (500..800, 150..250) through any window."""
    cfg = DetectorConfig(kind="yolo_obb", detect_dpi=150)
    cfg.tiling.enabled, cfg.tiling.full_pass, cfg.tiling.tile, cfg.tiling.merge_ios = tiling, full_pass, tile, 0.3
    det = object.__new__(YoloObbDetector)
    det.cfg, det.names = cfg, {0: "schedule_table"}
    obj = np.array([[500, 150], [800, 150], [800, 250], [500, 250]], float)  # in detect_dpi pixels
    real_detect_obbs = YoloObbDetector.detect_obbs

    def detect_obbs(image):  # the real tiling code, remembering the page so windows can be located
        det._page = image
        return real_detect_obbs(det, image)

    def predict(img, imgsz):  # img is a numpy view into the page: recover its origin from memory offsets
        off = img.__array_interface__["data"][0] - det._page.__array_interface__["data"][0]
        y0, x0 = off // det._page.strides[0], (off % det._page.strides[0]) // det._page.strides[1]
        h, w = img.shape[:2]
        piece = clip_to_rect(obj, x0, y0, x0 + w, y0 + h)
        if len(piece) < 3:
            return []
        return [OBB(0, 0.9, min_area_box(piece) - [x0, y0])]

    det.detect_obbs, det._predict = detect_obbs, predict
    return det


@pytest.mark.parametrize("tiling,full_pass", [(False, False), (True, False), (True, True)])
def test_tiled_detection_reassembles_object(tiling, full_pass) -> None:
    det = _fake_detector(tiling, full_pass)
    page = np.full((600, 1200, 3), 255, np.uint8)
    dets = det.detect(page, dpi=150)
    assert len(dets) == 1
    want = np.array([[500, 150], [800, 150], [800, 250], [500, 250]], float)
    assert iou(np.array(dets[0].polygon), want) > 0.97


def test_detect_rescales_to_detect_dpi() -> None:
    det = _fake_detector(False, False)
    page = np.full((1200, 2400, 3), 255, np.uint8)  # "300 dpi" page: object appears at 2x page coordinates
    out = det.detect(page, dpi=300)
    assert iou(np.array(out[0].polygon), np.array([[1000, 300], [1600, 300], [1600, 500], [1000, 500]], float)) > 0.97


# -- tables --------------------------------------------------------------------
def test_canonical_header_variants() -> None:
    assert canonical_header("天井仕上げ") == "ceiling"
    assert canonical_header("床 仕上") == "floor"
    assert canonical_header("Room Name") == "room"
    assert canonical_header("CLG") == "ceiling"
    assert canonical_header("巾木（既存）") == "baseboard"
    assert canonical_header("Door No.") is None and canonical_header(None) is None


@pytest.fixture(scope="module")
def vocab() -> dict:
    return load_vocab()


@pytest.mark.parametrize("mode", ["vector", "scan"])
def test_table_structure_on_gt_boxes(vocab, mode) -> None:
    cfg = SynthConfig(modes={mode: 1}, layouts={"schedule_sheet": 1}, sheet_sizes={"A3": 1},
                      dpi={"vector": 150, "scan": 200, "outlined": 200})
    pairs = []
    for i in range(2):
        res = generate_sheet(i, cfg, vocab)
        gt = res.gt["pages"][0]
        img = np.asarray(Image.open(io.BytesIO(res.image)).convert("RGB"))
        words = [SimpleNamespace(text=w["text"], bbox=w["bbox"]) for w in gt["words"]] if mode == "vector" else None
        pred = copy.deepcopy(gt)
        for r in pred["regions"]:
            if r["class"] == "schedule_table":
                r["table"] = recognize_table(img, r["polygon"], words, dpi=gt["dpi"])
        pairs.append(PagePair(str(i), gt, pred))
    m = table_metrics(pairs)
    assert m["matched_tables"] == m["gt_tables"] > 0
    assert m["cell_f1"] > 0.9 and m["span_accuracy"] > 0.95
    if mode == "vector":
        assert m["text_exact"] > 0.95 and m["header_accuracy"] > 0.9


# -- metrics -------------------------------------------------------------------
def _page(regions, w=1000):
    return {"width": w, "height": w, "regions": regions}


def test_average_precision_and_detection_metrics() -> None:
    assert average_precision([(0.9, True), (0.8, True)], 2) == pytest.approx(1.0)
    assert average_precision([(0.9, False), (0.8, True)], 1) == pytest.approx(0.5)
    gt = _page([{"class": "legend", "polygon": BOX.tolist()}])
    perfect = _page([{"class": "legend", "conf": 0.9, "polygon": (BOX * 2).tolist()}], w=2000)  # 2x frame
    m = detection_metrics([PagePair("a", gt, perfect)])
    assert m["per_class"]["legend"]["ap50"] == pytest.approx(1.0) and m["map50"] == pytest.approx(1.0)
    wrong = _page([{"class": "notes_block", "conf": 0.9, "polygon": BOX.tolist()}])
    m = detection_metrics([PagePair("a", gt, wrong)])
    assert m["per_class"]["legend"]["recall"] == 0 and m["per_class"]["notes_block"]["fp"] == 1


def test_label_file_to_gt_page() -> None:
    lab = {"input": "raw/x.pdf", "page": 1, "width": 100, "height": 50, "dpi": 150, "route": "vector",
           "regions": [{"class": "title_block", "box": [1, 2, 30, 40]},
                       {"class": "text_callout", "polygon": [[0, 0], [5, 0], [5, 5], [0, 5]]}]}
    gt = to_gt_page(lab)
    assert [r["class"] for r in gt["regions"]] == ["title_block", "text_callout"]
    assert gt["regions"][0]["polygon"] == [[1, 2], [30, 2], [30, 40], [1, 40]]


# -- end to end: oracle detector through the real pipeline ----------------------
def test_oracle_evaluation_end_to_end(tmp_path) -> None:
    cfg = SynthConfig(n_sheets=4, workers=1, modes={"vector": 1, "scan": 1}, sheet_sizes={"A3": 1},
                      layouts={"schedule_sheet": 1, "plan_side": 1}, splits={"test": 1},
                      dpi={"vector": 100, "scan": 100, "outlined": 100}, save_pdf=["test"])
    generate(cfg, tmp_path / "s", workers=1, progress=False)
    items = list(synth_items(tmp_path / "s", "test", None))
    pc = PipelineConfig()
    pc.render.vector_dpi = 100
    m = run_eval(items, pc, tmp_path / "out", oracle=True)
    assert not m["failures"] and m["detection"]["n_pages"] == 4
    assert m["detection"]["map50"] == pytest.approx(1.0)  # GT boxes in, GT boxes out
    assert m["tables"]["matched_tables"] == m["tables"]["gt_tables"] > 0
    assert m["tables"]["cell_f1"] > 0.8
    for f in (tmp_path / "out" / "pred").glob("*.json"):
        validate_output(json.loads(f.read_text(encoding="utf-8")))


def test_pipeline_attaches_tables_on_vector_pdf(tmp_path, vocab) -> None:
    cfg = SynthConfig(modes={"vector": 1}, layouts={"schedule_sheet": 1}, sheet_sizes={"A3": 1},
                      dpi={"vector": 150, "scan": 200, "outlined": 200})
    res = generate_sheet(0, cfg, vocab)
    pdf = tmp_path / "s.pdf"
    pdf.write_bytes(res.pdf)
    doc, _ = process_document(pdf, PipelineConfig(), OracleDetector(res.gt["pages"][0]))
    tables = [r["table"] for r in doc["pages"][0]["regions"] if r["class"] == "schedule_table"]
    assert tables and all(t is not None and t["rows"] for t in tables)
    assert any("room" in t["canonical_headers"] for t in tables)
