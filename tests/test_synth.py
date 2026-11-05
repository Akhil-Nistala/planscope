"""Synthetic generator (Milestone 2): label geometry, GT contract, modes, determinism, CLI.

The geometry tests use the PDF's own text layer as an independent witness: in vector sheets every
labelled region must contain exactly the text the generator says it drew, and every table cell must
contain its cell text -- including rotated callouts.
"""

from __future__ import annotations

import copy
import io
import json
from collections import Counter

import cv2
import numpy as np
import pytest
from PIL import Image

from data_gen.config import SynthConfig, load_synth_config, load_vocab
from data_gen.generate import run
from data_gen.sheet import apply, generate_sheet, sample_spec, yolo_obb_line
from src.detect import CLASSES
from src.pipeline import point_in_polygon
from src.schema import validate_output

N = 3  # sheets per mode


def _cfg(mode: str, **kw) -> SynthConfig:
    cfg = SynthConfig(n_sheets=N, workers=1, modes={mode: 1.0}, sheet_sizes={"A3": 1.0},
                      dpi={"vector": 60, "scan": 60, "outlined": 60})
    for k, v in kw.items():
        setattr(cfg, k, v)
    cfg.validate()
    return cfg


@pytest.fixture(scope="module")
def vocab() -> dict:
    return load_vocab()


@pytest.fixture(scope="module")
def sheets(vocab) -> dict:
    """{mode: [SheetResult, ...]} -- the same N indices rendered in every mode."""
    return {m: [generate_sheet(i, _cfg(m), vocab) for i in range(N)] for m in ("vector", "scan", "outlined")}


_DASHES = str.maketrans({c: "-" for c in "‐‑‒–—−"})


def _squash(s: str) -> Counter:
    """Character multiset without whitespace (robust to word order and line breaks).

    Dashes are unified: Japanese fonts map "-" to U+2011 in the extracted text layer.
    """
    return Counter(c for c in s.translate(_DASHES) if not c.isspace())


def _words_in(words: list[dict], poly) -> list[dict]:
    return [w for w in words
            if point_in_polygon((w["bbox"][0] + w["bbox"][2]) / 2, (w["bbox"][1] + w["bbox"][3]) / 2, poly)]


def _labels(res) -> list[list[float]]:
    return [[float(v) for v in ln.split()] for ln in res.labels.splitlines()]


# -- contract ----------------------------------------------------------------
@pytest.mark.parametrize("mode,route", [("vector", "vector"), ("scan", "raster"), ("outlined", "raster")])
def test_outputs_valid(sheets, mode, route) -> None:
    for res in sheets[mode]:
        validate_output(copy.deepcopy(res.gt))
        page = res.gt["pages"][0]
        assert page["route"] == route and page["dpi"] == 60
        assert (len(page["words"]) > 0) == (route == "vector")
        img = Image.open(io.BytesIO(res.image))
        assert img.size == (page["width"], page["height"]) and round(img.info["dpi"][0]) == 60
        assert res.image_ext == ("jpg" if mode == "scan" else "png")
        assert (res.pdf is not None) == (mode != "scan")

        labels = _labels(res)
        assert len(labels) == len(page["regions"]) == len(res.meta["regions"]) > 0
        for lab, reg in zip(labels, page["regions"]):
            assert len(lab) == 9 and CLASSES[int(lab[0])] == reg["class"]
            assert all(0.0 <= v <= 1.0 for v in lab[1:])
        assert {r["class"] for r in page["regions"]} >= {"title_block"}


def test_every_sheet_has_one_title_block_and_text_on_raster_only_in_meta(sheets) -> None:
    for mode, results in sheets.items():
        for res in results:
            classes = [r["class"] for r in res.gt["pages"][0]["regions"]]
            assert classes.count("title_block") == 1
            for reg, truth in zip(res.gt["pages"][0]["regions"], res.meta["regions"]):
                assert truth["text"]  # true text is always recorded
                assert reg["text"] == (truth["text"] if mode == "vector" else None)


def test_polygons_are_clockwise_on_screen(sheets) -> None:
    for results in sheets.values():
        for res in results:
            for reg in res.gt["pages"][0]["regions"]:
                p = np.array(reg["polygon"])
                signed = np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
                assert signed > 0, reg  # y points down, so clockwise on screen is positive


# -- geometry against the text layer ----------------------------------------
def test_vector_regions_contain_exactly_their_text(sheets) -> None:
    for res in sheets["vector"]:
        page = res.gt["pages"][0]
        for reg in page["regions"]:
            got = " ".join(w["text"] for w in _words_in(page["words"], reg["polygon"]))
            assert _squash(got) == _squash(reg["text"]), (res.spec.id, reg["id"], reg["class"])


def test_vector_table_cells_contain_their_text(sheets) -> None:
    n_cells = 0
    for res in sheets["vector"]:
        page = res.gt["pages"][0]
        for reg in page["regions"]:
            if reg["table"] is None:
                continue
            assert reg["class"] == "schedule_table"
            for row in reg["table"]["rows"]:
                for cell in row["cells"]:
                    x0, y0, x1, y1 = cell["bbox"]
                    inside = _words_in(page["words"], [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
                    assert _squash(" ".join(w["text"] for w in inside)) == _squash(cell["text"]), cell
                    n_cells += 1
    assert n_cells > 0


def test_rotated_callouts_match_text_layer(vocab) -> None:
    cfg = _cfg("vector", callout_angles={"oblique": 0.5, "vertical": 0.5}, layouts={"plan_only": 1.0})
    n_rotated = 0
    for i in range(N):
        page = generate_sheet(i, cfg, vocab).gt["pages"][0]
        for reg in page["regions"]:
            p = np.array(reg["polygon"])
            if reg["class"] != "text_callout" or abs(p[1, 1] - p[0, 1]) < 1.0:
                continue  # axis-aligned (top edge horizontal)
            n_rotated += 1
            got = " ".join(w["text"] for w in _words_in(page["words"], reg["polygon"]))
            assert _squash(got) == _squash(reg["text"]), reg
    assert n_rotated >= 3


def test_scan_polygons_follow_the_skew(sheets) -> None:
    """Same index in scan vs vector mode: scan polygons = vector polygons moved by the scan affine."""
    for vec, scan in zip(sheets["vector"], sheets["scan"]):
        sp = scan.meta["scan"]
        h, w = scan.meta["height"], scan.meta["width"]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), sp["skew_deg"], 1.0)
        M[:, 2] += sp["shift_px"]
        T = np.vstack([M, [0, 0, 1]])
        for rv, rs in zip(vec.gt["pages"][0]["regions"], scan.gt["pages"][0]["regions"]):
            assert rv["class"] == rs["class"]
            want = np.clip(apply(T, rv["polygon"]), 0, [w, h])
            np.testing.assert_allclose(np.array(rs["polygon"]), want, atol=0.05)  # rounding + 4-dp skew


def test_outlined_sheets_have_no_usable_text_layer(sheets) -> None:
    import pymupdf as fitz
    for res in sheets["outlined"]:
        assert res.meta["router"]["route"] == "raster" and res.meta["router"]["agrees"]
        with fitz.open("pdf", res.pdf) as d:
            words = [w[4] for w in d[0].get_text("words")]
            assert len(d[0].get_drawings()) > 100  # glyphs became paths
        title = next(t for t in res.meta["regions"] if t["class"] == "title_block")
        # only the drawing number survives as real text, and it is in the title block
        assert 0 < len(words) <= 3 and all(w.translate(_DASHES) in title["text"] for w in words), words


def test_vector_router_agrees(sheets) -> None:
    for res in sheets["vector"]:
        assert res.meta["router"]["route"] == "vector" and res.meta["router"]["agrees"]


# -- determinism -------------------------------------------------------------
def test_deterministic_and_seed_sensitive(vocab, sheets) -> None:
    again = generate_sheet(1, _cfg("vector"), vocab)
    first = sheets["vector"][1]
    assert again.labels == first.labels and again.gt == first.gt and again.image == first.image
    other = generate_sheet(1, _cfg("vector", seed=999), vocab)
    assert other.labels != first.labels


def test_content_independent_of_mode_mix() -> None:
    a, b = SynthConfig(), SynthConfig(modes={"vector": 1.0})
    sa, sb = sample_spec(5, a), sample_spec(5, b)
    assert sa.lang == sb.lang and sa.layout == sb.layout and sa.sheet_size == sb.sheet_size


# -- small units -------------------------------------------------------------
def test_yolo_line_normalises_and_clips() -> None:
    poly = np.array([[0.0, 0.0], [200.0, 0.0], [200.0, 100.0], [0.0, 100.0]])
    v = yolo_obb_line("legend", poly, 400, 200).split()
    assert v[0] == str(CLASSES.index("legend"))
    assert [float(x) for x in v[1:]] == [0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5]
    big = yolo_obb_line("legend", poly * 3, 400, 200).split()
    assert max(float(x) for x in big[1:]) == 1.0


def test_config_rejects_bad_values(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        SynthConfig(modes={"vector": 1, "hologram": 1}).validate()
    with pytest.raises(ValueError, match="positive sum"):
        SynthConfig(langs={"ja": 0, "en": 0}).validate()
    bad = tmp_path / "s.yaml"
    bad.write_text("n_sheet: 5\n", encoding="utf-8")
    with pytest.raises(KeyError, match="n_sheet"):
        load_synth_config(bad)
    load_synth_config()  # the shipped config is valid


# -- CLI / dataset layout ----------------------------------------------------
def test_run_writes_dataset(tmp_path) -> None:
    import yaml
    cfg = _cfg("vector", n_sheets=5, modes={"vector": 1, "scan": 1, "outlined": 1}, splits={"train": 1, "val": 1})
    out = tmp_path / "synth"
    summary = run(cfg, out, workers=1, progress=False)
    assert summary["generated"] == 5 and not summary["failed"]

    ds = yaml.safe_load((out / "dataset.yaml").read_text(encoding="utf-8"))
    assert ds["names"] == dict(enumerate(CLASSES)) and ds["train"] == "images/train"
    manifest = [json.loads(ln) for ln in (out / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [m["index"] for m in manifest] == list(range(5))
    for m in manifest:
        assert (out / m["image"]).is_file() and (out / m["label"]).is_file() and (out / m["gt"]).is_file()
        assert (out / "meta" / f"{m['id']}.json").is_file()
        assert (m["pdf"] is None) == (m["mode"] == "scan")
        assert m["image"].split("/")[1] == m["split"] == m["label"].split("/")[1]
        validate_output(json.loads((out / m["gt"]).read_text(encoding="utf-8")))
    assert sum(summary["instances"].values()) == sum(len((out / m["label"]).read_text().splitlines()) for m in manifest)

    with pytest.raises(SystemExit, match="not empty"):
        run(cfg, out, workers=1, progress=False)
    run(cfg, out, workers=1, overwrite=True, progress=False)
