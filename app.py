"""Demo site: pick a sample drawing (or upload one) -> routing, overlays, schema-valid JSON.

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.config import load_config
from src.pipeline import process_document

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "data" / "samples"
CLASS_COLORS = {  # RGB
    "title_block": (220, 50, 47), "schedule_table": (38, 139, 210), "legend": (133, 153, 0),
    "notes_block": (211, 54, 130), "text_callout": (203, 75, 22),
}
SAMPLE_INFO = {
    "jp_finish_schedule_vector.pdf": "A3 digital sheet · Japanese 仕上表 + 特記仕様書 → expect VECTOR",
    "en_finish_schedule_vector.pdf": "A1 digital sheet · English finish schedule → expect VECTOR",
    "jp_scanned.pdf": "Same JP sheet, printed & scanned (image-only PDF) → expect RASTER",
    "jp_scanned.png": "The scan as a PNG image → RASTER (image input)",
    "cad_outlined_text.pdf": "CAD export, annotation text vectorised to strokes → expect RASTER",
    "mixed_set.pdf": "3 pages: vector + scan + outlined text → per-page routing",
}

st.set_page_config(page_title="Drawing Layout Analysis", layout="wide")


def ensure_samples() -> None:
    """Generate the sample drawings on first run (deterministic, seed 42)."""
    if not (SAMPLES / "mixed_set.pdf").exists():
        with st.spinner("Generating sample drawings (first run only)…"):
            subprocess.run([sys.executable, str(ROOT / "data" / "make_samples.py"), "--out", str(SAMPLES)],
                           check=True)


def overlay(img: np.ndarray, page: dict, show_words: bool, max_w: int = 1800) -> np.ndarray:
    """Draw word boxes (green) and region polygons (per-class colours) on a downscaled copy."""
    s = min(1.0, max_w / img.shape[1])
    out = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img.copy()
    if show_words:
        for w in page["words"]:
            x0, y0, x1, y1 = (int(v * s) for v in w["bbox"])
            cv2.rectangle(out, (x0, y0), (x1, y1), (0, 170, 0), 1)
    for r in page["regions"]:
        pts = (np.array(r["polygon"]) * s).astype(np.int32)
        cv2.polylines(out, [pts], True, CLASS_COLORS[r["class"]], 2)
        cv2.putText(out, f"{r['class']} {r['conf']:.2f}", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    CLASS_COLORS[r["class"]], 1)
    return out


@st.cache_data(show_spinner=False)
def run(path_str: str, min_words: int, min_ratio: float, _mtime: float) -> tuple[dict, list[np.ndarray]]:
    """Run the pipeline (cached per file + thresholds)."""
    cfg = load_config()
    cfg.router.min_words = min_words
    cfg.router.min_valid_char_ratio = min_ratio
    return process_document(path_str, cfg, keep_images=True)


ensure_samples()
st.title("Drawing Layout & Table Analysis — Project A")
st.caption("Stage 1 of an asbestos-material screening pipeline: find where material information lives on a "
           "drawing and emit structured JSON. **Milestone 1:** input router, rendering, output schema. "
           "Layout detection (YOLO11-OBB) and table structure arrive in later milestones — regions are empty for now.")

with st.sidebar:
    st.header("Input")
    samples = sorted(p.name for p in SAMPLES.iterdir() if p.suffix.lower() in {".pdf", ".png"})
    choice = st.radio("Sample drawing", samples, format_func=lambda n: n)
    st.caption(SAMPLE_INFO.get(choice, ""))
    up = st.file_uploader("…or upload a PDF / image", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"])
    st.header("Router thresholds")
    cfg0 = load_config()
    min_words = st.slider("min_words", 0, 200, cfg0.router.min_words)
    min_ratio = st.slider("min_valid_char_ratio", 0.0, 1.0, cfg0.router.min_valid_char_ratio, 0.01)
    show_words = st.checkbox("Show text-layer word boxes", True)

if up is not None:
    tmp = Path(tempfile.gettempdir()) / f"pa_upload_{up.name}"
    tmp.write_bytes(up.getvalue())
    path = tmp
else:
    path = SAMPLES / choice

prev = SAMPLES / "previews"
with st.expander("Sample gallery", expanded=False):
    cols = st.columns(3)
    for i, f in enumerate(sorted(prev.glob("*.png"))):
        cols[i % 3].image(str(f), caption=f.stem, width="stretch")

try:
    doc, images = run(str(path), min_words, min_ratio, path.stat().st_mtime)
except Exception as e:  # surface errors in the UI rather than a stack trace
    st.error(f"Failed to process {path.name}: {e}")
    st.stop()

st.success(f"**{doc['document']}** · {len(doc['pages'])} page(s) · output validated against `src/schema.json` ✔")
for page, img in zip(doc["pages"], images):
    route = page["route"].upper()
    badge = "🟢 VECTOR — text from PDF layer" if route == "VECTOR" else "🟠 RASTER — needs OCR"
    st.subheader(f"Page {page['page']}: {badge}")
    left, right = st.columns([3, 1])
    with left:
        st.image(overlay(img, page, show_words), width="stretch",
                 caption=f"{page['width']}×{page['height']} px @ {page['dpi']} DPI")
    with right:
        r = page["routing"]
        if r:
            st.metric("words", r["n_words"])
            st.metric("valid char ratio", f"{r['valid_char_ratio']:.3f}")
            st.write("**Reasons:**", "; ".join(r["reasons"]))
        else:
            st.write("Image input — no text layer, routed to OCR.")
        t = page.get("timings", {})
        st.write(f"route {t.get('route_s', 0) * 1000:.0f} ms · render {t.get('render_s', 0) * 1000:.0f} ms")
        if page["words"]:
            st.write("**First words:**")
            st.dataframe([{"text": w["text"], "bbox": [round(v) for v in w["bbox"]]} for w in page["words"][:40]],
                         height=300, width="stretch")

js = json.dumps(doc, ensure_ascii=False, indent=2)
st.download_button("Download JSON", js, file_name=f"{Path(doc['document']).stem}.json", mime="application/json")
with st.expander("Raw JSON output"):
    st.code(js[:60_000] + ("\n… (truncated)" if len(js) > 60_000 else ""), language="json")
