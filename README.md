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
| 3. Baseline YOLO11-OBB | ✅ done (small-scale CPU training; full-scale recipe in `notebooks/train_colab.ipynb`) |
| 4. Real test set + tiled inference | ✅ done (23 public-domain / open-licence pages, AI-annotated) |
| 5. Table structure recognition + cell text assignment | ✅ done |
| 6. Ablation, error analysis, demo, final README | ✅ done |

**Scale caveat.** Everything below was trained and evaluated on a 4-core, 8 GB laptop without a GPU. The
models saw a few hundred synthetic sheets for 7 (baseline) and 6 (tile model) epochs, against 3,000 sheets
and 60 epochs in the intended recipe. Treat the detection numbers as a working lower bound, not as what this
design can reach.

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
pip install --no-deps ultralytics==8.4.202           # see requirements.txt: avoids an opencv-python clash
python data/make_samples.py                          # generate sample drawings
pytest                                               # unit tests
python -m src.pipeline --input data/samples --out outputs -v   # PDF/image -> JSON (detection + tables)
streamlit run app.py                                 # demo site
```

Reproducing the model and the numbers (small CPU scale; `notebooks/train_colab.ipynb` for full scale):

```bash
python -m data_gen.generate --out data/synth --n 400                    # ~0.6 MB/sheet
python -m data_gen.preview --root data/synth --n 8                      # eyeball the labels
python -m training.tile_dataset --src data/synth --out data/synth_tiles --tile 640 --max-tiles 6
python -m training.train --data data/synth/dataset.yaml --name full_640 --epochs 7 --batch 4 --fraction 0.5
python -m training.train --data data/synth_tiles/dataset.yaml --name tiles_640 --epochs 6 --batch 4 --subset 320
python -m eval.evaluate --data data/synth --split test --weights models/tiles_640.pt --name tiles_640
python -m eval.evaluate --real data/real --weights models/tiles_640.pt --name real_tiles_640   # needs data/real/raw
```

The committed results came from 2–3 epoch chunks: each chunk continued from the previous weights with
`--model models/<name>.pt --warmup 0`, and each tile chunk drew a fresh `--subset-seed`. This is because long-running
background jobs get stopped on an 8 GB machine.

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

## Layout detection (`src/detect.py`, `src/tiling.py`, `training/`)

* **Model:** Ultralytics YOLO11n-OBB (oriented boxes), fine-tuned from the COCO-pretrained `yolo11n-obb.pt` on
  the synthetic sheets. Augmentation is tuned for drawings: no flips (mirrored text is not a drawing), mild
  scale jitter, no colour jitter.
* **Fixed physical scale:** every page is resampled to `detect_dpi` (150) before detection, so a 300-DPI
  render of an outlined CAD sheet and a 200-DPI scan look like training data. DPI metadata implying a sheet
  smaller than A4 is ignored (a microfilm frame tagged 600 dpi would otherwise shrink to a thumbnail).
* **Tiled inference:** sheets are large (A1 at 150 DPI is ~5000 × 3500 px) and callouts are ~10 px tall, so
  detection runs on overlapping 640-px tiles at native resolution, plus one downscaled whole-sheet pass for
  regions larger than a tile. Pieces of one object cut by tile borders are merged back: same class and
  intersection-over-smaller ≥ 0.6 gives a union box; IoU ≥ 0.5 marks a duplicate (`src/tiling.py`).
* **Training data for tiles:** `training/tile_dataset.py` cuts sheets into 640-px tiles at 150 DPI and clips
  every label to the tile (kept if ≥ 50% visible, or if it covers a large part of the tile).
* **Two models**, both from `training/train.py`:
  * `full_640`: the baseline, trained on whole sheets letterboxed to 640 px;
  * `tiles_640`: trained on tiles.

  The committed `models/planscope_obb.pt` is the one chosen in Results below.

## Table structure (`src/table.py`)

Rule-based, because finish schedules are ruled grids:

1. Warp the detected oriented box to an upright crop resampled to 200 DPI, which handles scan skew.
2. Find ruling lines with long morphological openings, on a mask of "clearly darker than the paper". That
   catches faint, thin rules on faded scans, which Otsu thresholding misses.
3. Turn rule profiles into row and column separators. A thick band, such as a grey header fill merged with
   its rules, contributes both of its edges. Tables without vertical rules fall back to whitespace gaps
   between text columns.
4. A rule segment missing between two grid cells merges them, which recovers merged level cells (row spans).
5. On vector pages, assign text-layer words to cells. On raster pages, leave cells `null` for OCR.
6. Map the header row to canonical fields (room / floor / baseboard / wall / ceiling / remarks / level) using
   the spelling variants in `configs/vocab.yaml`: an exact match first, then the longest known spelling
   contained in the header ("天井仕上げ（既存）" → ceiling).

## Real held-out test set (`data/real`)

23 real pages, used only for testing (provenance and licences in `DATA.md`):

| Source | Pages | Format | Content |
|---|---|---|---|
| US National Archives, Governors Island (US Coast Guard), 1977–1995 | 12 | bilevel scans (200 DPI) | Room finish schedules and asbestos-abatement sheets |
| US Army Corps of Engineers | 1 | microfilm scan | Room finish schedule |
| Japan Ministry of the Environment (PDL1.0) | 7 | vector PDFs | Renovation drawings |
| Kobe City (政府標準利用規約) | 3 | vector PDFs | Renovation / asbestos specification pages |

They contain 433 labelled regions: 351 text callouts, 38 notes blocks, 19 title blocks, 18 schedule tables
and 7 legends. Labels follow the synthetic labelling rules. They were drawn by AI annotators
with `eval/label_tool.py`, which reads coordinates off gridded, zoomed renders and checks them on overlays,
and have not been reviewed by a human.

## Evaluation (`eval/`)

* `eval/evaluate.py` runs the **real pipeline** on held-out sheets (vector and outlined sheets from their PDFs,
  scans from their JPEGs) and scores the output JSON against GT:
  * detection: oriented-box IoU, per-class AP50 and AP50:95, and P/R/F1 at the configured confidence;
  * breakdowns by mode, language, sheet size and layout, plus a confusion matrix;
  * tables: cell P/R/F1, span accuracy, cell-text exact match, header accuracy, and exact-grid rate.

  `--oracle` feeds GT boxes into the pipeline, so table recognition is scored without detector errors.
* `eval/errors.py` draws GT against predictions on the worst pages and tallies misses, false positives and
  class confusions.
* `eval/label_tool.py` is the annotation helper for the real set (see `data/real/README.md`).

## Limitations

* **Compute scale.** The models were trained for 6–7 epochs on a few hundred synthetic sheets on a CPU.
  The real-set numbers mostly measure the synthetic-to-real gap at that scale.
* **The real test set is small and AI-annotated.** It has 23 pages, labelled in a single pass by an AI
  assistant with no human review (see `DATA.md`). It has no Japanese sheet with a full 仕上表, because none was
  found under a clean licence. Tables on real pages are scored for detection only; cell structure isn't
  annotated there.
* **The router** has been checked on the synthetic PDFs and on the real set's 10 vector pages. Hybrid pages
  (a scan with a partial OCR text layer, or a vector sheet with a pasted raster detail) are classified as a
  whole page, not per region.
* **Table structure is rule-based.** It assumes a single header row and a ruled or column-aligned grid.
  Multi-row headers, tables split across columns of a sheet, and matrix-style schedules (finish codes per room
  in a grid of check marks) aren't recovered as such.
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
