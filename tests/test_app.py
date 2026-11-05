"""Headless smoke test of the Streamlit demo on every sample (generates samples on first run)."""

from __future__ import annotations

from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.mark.parametrize("sample,routes", [
    ("jp_finish_schedule_vector.pdf", ["VECTOR"]),
    ("en_finish_schedule_vector.pdf", ["VECTOR"]),
    ("jp_scanned.pdf", ["RASTER"]),
    ("jp_scanned.png", ["RASTER"]),
    ("cad_outlined_text.pdf", ["RASTER"]),
    ("mixed_set.pdf", ["VECTOR", "RASTER", "RASTER"]),
])
def test_demo_renders_sample(sample: str, routes: list[str]) -> None:
    at = st_testing.AppTest.from_file(APP, default_timeout=180)
    at.run()
    at.sidebar.radio[0].set_value(sample).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    got = [("VECTOR" if "VECTOR" in s.value else "RASTER") for s in at.subheader]
    assert got == routes
