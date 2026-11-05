"""Output contract: schema validation, semantic invariants, end-to-end pipeline output."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from src.detect import Detection
from src.pipeline import point_in_polygon, process_document
from src.schema import SCHEMA_PATH, OutputValidationError, validate_output


def _valid_doc() -> dict:
    cell = {"text": "ビニル床タイル", "bbox": [10, 10, 50, 20], "text_source": "vector"}
    return {
        "schema_version": "1.0",
        "document": "file.pdf",
        "pages": [{
            "page": 1, "route": "vector", "dpi": 150, "width": 100, "height": 100, "needs_ocr": False,
            "routing": {"n_words": 20, "n_chars": 100, "valid_char_ratio": 1.0, "text_coverage": 0.1,
                        "reasons": ["ok"]},
            "words": [{"text": "床", "bbox": [1, 1, 5, 5], "block": 0, "line": 0}],
            "regions": [{
                "id": "r1", "class": "schedule_table", "conf": 0.93,
                "polygon": [[0, 0], [90, 0], [90, 60], [0, 60]],
                "text": "室名 床", "text_source": "vector",
                "table": {"headers": ["室名", "床"], "canonical_headers": ["room", "floor"],
                          "rows": [{"cells": [cell, cell]}]},
            }],
        }],
    }


def test_schema_file_is_valid_json_schema() -> None:
    from jsonschema import Draft202012Validator
    Draft202012Validator.check_schema(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))


def test_valid_doc_passes() -> None:
    validate_output(_valid_doc())


@pytest.mark.parametrize("mutate,msg", [
    (lambda d: d["pages"][0]["regions"][0].update({"class": "asbestos"}), "class"),
    (lambda d: d["pages"][0]["regions"][0].update({"conf": 1.5}), "conf"),
    (lambda d: d["pages"][0]["regions"][0].update({"polygon": [[0, 0], [1, 1]]}), "polygon"),
    (lambda d: d["pages"][0].update({"route": "hybrid"}), "route"),
    (lambda d: d["pages"][0].pop("words"), "words"),
    (lambda d: d["pages"][0].update({"extra": 1}), "extra"),
])
def test_schema_rejects(mutate, msg) -> None:
    d = _valid_doc()
    mutate(d)
    with pytest.raises(OutputValidationError, match=msg):
        validate_output(d)


@pytest.mark.parametrize("mutate,msg", [
    (lambda d: d["pages"][0].update({"needs_ocr": True}), "needs_ocr"),
    (lambda d: d["pages"][0]["regions"][0].update({"text": None}), "null iff"),
    (lambda d: d["pages"][0]["regions"][0]["table"].update({"canonical_headers": ["room"]}), "length"),
    (lambda d: d["pages"][0]["words"][0].update({"bbox": [5, 5, 1, 1]}), "ordered"),
    (lambda d: d["pages"][0]["regions"][0].update({"class": "legend"}), "only schedule_table"),
])
def test_semantic_invariants(mutate, msg) -> None:
    d = _valid_doc()
    mutate(d)
    with pytest.raises(OutputValidationError, match=msg):
        validate_output(d)


def test_point_in_rotated_polygon() -> None:
    diamond = [(50, 0), (100, 50), (50, 100), (0, 50)]  # 45-degree rotated square
    assert point_in_polygon(50, 50, diamond)
    assert not point_in_polygon(5, 5, diamond)


class _FixedDetector:
    """Returns one box covering the top part of the page (where the test text is)."""

    def __init__(self, cls: str = "schedule_table") -> None:
        self.cls = cls

    def detect(self, image: np.ndarray) -> list[Detection]:
        h, w = image.shape[:2]
        return [Detection(self.cls, 0.9, [(0, 0), (w, 0), (w, h * 0.3), (0, h * 0.3)])]


def test_pipeline_vector_output_valid(vector_pdf, cfg) -> None:
    doc, _ = process_document(vector_pdf, cfg, _FixedDetector())
    p = doc["pages"][0]
    assert p["route"] == "vector" and not p["needs_ocr"] and p["words"]
    r = p["regions"][0]
    assert r["text_source"] == "vector" and "gypsum" in r["text"] and "仕上表" in r["text"]


def test_pipeline_raster_output_valid(raster_pdf, cfg) -> None:
    doc, _ = process_document(raster_pdf, cfg, _FixedDetector())
    p = doc["pages"][0]
    assert p["route"] == "raster" and p["needs_ocr"] and p["words"] == []
    assert p["regions"][0]["text"] is None and p["regions"][0]["text_source"] == "needs_ocr"


def test_pipeline_image_dpi_null_allowed(tmp_path, cfg) -> None:
    from PIL import Image
    f = tmp_path / "nodpi.png"
    Image.fromarray(np.full((50, 50, 3), 255, np.uint8)).save(f)
    doc, _ = process_document(f, cfg)
    assert doc["pages"][0]["dpi"] is None


def test_output_is_json_roundtrippable(vector_pdf, cfg) -> None:
    doc, _ = process_document(vector_pdf, cfg, _FixedDetector())
    again = json.loads(json.dumps(doc, ensure_ascii=False))
    validate_output(copy.deepcopy(again))
