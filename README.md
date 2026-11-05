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
| 2. Synthetic generator (images + OBB labels + GT JSON) | ✅ done |
| 3. Baseline YOLO11-OBB | ⏳ next |
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
python -m data_gen.generate --out data/synth         # synthetic set: ~0.6 MB/sheet, 3,000 sheets ~1.8 GB (--n to shrink)
python -m data_gen.preview --root data/synth --n 8   # draw labels over a few sheets to check them
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

## Synthetic training data (`data_gen/`)

`python -m data_gen.generate --out data/synth` draws sheets as real PDFs with PyMuPDF and labels them from the
geometry it drew. All settings live in `configs/synth.yaml`, and all wording (headers and their spelling
variants, materials, notes, callouts, title-block fields) lives in `configs/vocab.yaml`.

* **Content:** floor plans (grid, rooms, doors, hatching, dimensions, north arrow), finish schedules (2–4 finish
  columns, optional merged level column and remarks, varied header spellings and ruling), notes, legends, and
  title blocks (corner box with optional revision rows, or a full-height strip). Japanese (70%) and English;
  A3/A2/A1. Text size is fixed in points, so callouts on an A1 sheet are as small as they are in practice.
* **Three modes**, matching the router's paths:
  * `vector` (50%): digital PDF with a text layer.
  * `scan` (30%): skewed, blurred, faded, noisy JPEG.
  * `outlined` (20%): every glyph converted to real vector outlines, with only the drawing number left as
    text, like CAD "text as geometry" exports.
* **Labels:** five classes as oriented boxes. Room tags and leader callouts can be vertical or oblique.
  Dimension figures, grid labels, zone marks and schedule titles are drawn but left unlabelled, so they act
  as negatives. The full rules are in the docstring of `data_gen/components.py`.
* **Outputs per sheet:**
  * a YOLO-OBB label file, plus `dataset.yaml` for Ultralytics;
  * a GT JSON that validates against `src/schema.json`: what a perfect pipeline would output, built with the
    pipeline's own `page_record`, including the real text-layer words for vector sheets and table cells
    with row spans;
  * a `meta/*.json` file with every generation parameter, the router's verdict, and the true text of every
    region and cell. Raster sheets need this because the contract requires `text: null` for them.
* **Reproducible:** each sheet is a pure function of `(seed, index)`. Content, spec and scan noise use separate
  random streams, so changing the mode mix doesn't change what the sheets look like.
* **Verified by tests** (`tests/test_synth.py`): in vector sheets, the PDF's own text layer must agree with
  every region polygon and every table cell box, rotated callouts included. Scan polygons must equal the
  vector polygons moved by the recorded skew. Outlined sheets must have no usable text layer, and the router
  must agree with the intended route.

A 100-sheet run on 8 cores took 85 s and produced 0 failures. It contained 2,607 text callouts, 475 schedules,
239 notes blocks, 178 legends and 100 title blocks; the router agreed on 71/71 PDF sheets.

## Limitations (so far)

* The router heuristic is tuned on synthetic samples only; it has not yet been measured on real drawings
  (planned: labelled sample in Milestone 4). Hybrid pages (a scan with a partial OCR text layer, or a
  vector sheet with a pasted raster detail) are classified as a whole page, not per region.
* The sample drawings in `data/samples` are generated, not real; the "CAD outlined text" sample imitates
  vectorised text with random strokes (the synthetic generator uses real glyph outlines instead).
* Synthetic sheets are cleaner and more regular than real drawings. There are no stamps, handwriting,
  colour, revision clouds, overlapping annotations or non-rectangular plans, and every room is a grid bay
  or two. Classes are imbalanced (about 26 text callouts per title block). How large the synthetic-to-real
  gap is gets measured on the real test set in Milestone 4.
* Japanese fonts extract "-" as U+2011 (non-breaking hyphen) from the text layer, so downstream text
  matching must normalise dashes.
* Outlined Japanese schedule sheets are the slowest to generate (up to ~40 s each), because every CJK glyph
  goes through an SVG round trip.
