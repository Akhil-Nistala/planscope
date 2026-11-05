# PlanScope — Layout & Table Analysis for Building Drawings

First stage of an asbestos-containing-material (ACM) screening pipeline:

```
drawing PDF/image ──► PlanScope: router ─► render ─► layout detector ─► table structure ─► JSON ──► PlanRead (OCR) ──► ACM matching
                                │ vector: text from PDF layer ────────────────────────────────┘ (no OCR needed)
                                └ raster: text flagged needs_ocr ─────────────────────────────► PlanRead
```

PlanScope does **not** decide whether anything contains asbestos — it only locates where material
information lives (finish schedules 仕上表, notes 特記仕様書, title blocks, legends, callouts) and
turns it into structured, schema-validated JSON. The determination is made from text further downstream.

## Status

| Milestone | Status |
|---|---|
| 1. Router + PDF rendering + schema, with tests | ✅ done |
| 2. Synthetic generator (images + OBB labels + GT JSON) | ⏳ next |
| 3. Baseline YOLO11-OBB | — |
| 4. Real test set + tiled inference | — |
| 5. Table structure recognition + cell text assignment | — |
| 6. Ablation, error analysis, demo, final README | — |

## Quick start

Windows PowerShell (5.1 has no `&&`, so one command per line):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Linux / macOS / Git Bash: `python -m venv .venv && source .venv/bin/activate`

Then:

```bash
pip install -r requirements.txt
python data/make_samples.py                          # generate sample drawings
pytest                                               # unit tests
python -m src.pipeline --input data/samples --out outputs -v   # PDF -> JSON
streamlit run app.py                                 # demo site
```

## Router (vector fast path)

`src/router.py` classifies each PDF page from its PyMuPDF text layer (`page.get_text("words")`):

* **VECTOR** if `n_words >= min_words` **and** `valid_char_ratio >= min_valid_char_ratio`
  (and optionally word-box coverage `>= min_text_coverage`). Invalid chars = U+FFFD and Unicode
  categories Co/Cn/Cc/Cs — what CAD fonts with broken ToUnicode maps typically extract to.
  Words are kept with coordinates; the page is rendered at `vector_dpi` (150) for layout detection only.
* **RASTER** otherwise (scans; CAD exports with text drawn as paths). Rendered at `dpi` (300); text is `needs_ocr`.
* Image inputs (PNG/JPG/TIFF) are always RASTER; DPI comes from image metadata, else `images.default_dpi`.
* Very large sheets are capped by `render.max_pixels` (DPI lowered, logged).

All thresholds live in `configs/inference.yaml`. Each decision is logged and stored in the JSON
(`pages[].routing`) with the statistics and reasons behind it.

## Output contract

`src/schema.json` (JSON Schema 2020-12) + semantic checks in `src/schema.py`
(e.g. `needs_ocr == (route == raster)`, `text is null iff text_source == needs_ocr`,
`len(headers) == len(canonical_headers)`). Every pipeline output is validated before it is written.

Coordinates: **pixels of the rendered page image** at `pages[].dpi` (origin top-left).
Polygons are 4 corners clockwise. Beyond the example contract, each page also carries
`width`, `height`, `needs_ocr`, `routing`, full-page `words` (vector pages) and optional `timings`.

## Limitations (so far)

* The router heuristic is tuned on synthetic samples only; it has not yet been measured on real drawings
  (planned: labelled sample in Milestone 4). Hybrid pages (a scan with a partial OCR text layer, or a
  vector sheet with a pasted raster detail) are classified as a whole page, not per region.
* The sample drawings in `data/samples` are generated, not real; the "CAD outlined text" sample imitates
  vectorised text with random strokes.
