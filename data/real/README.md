# data/real — real held-out test set

Real drawing sheets used **only for testing** (never for training or tuning). Provenance and licences
are in `sources.json` (one entry per file) and summarised in the repository's `DATA.md`.

```
raw/         downloaded sheets (gitignored; re-download with the URLs in sources.json)
labels/      one JSON label file per page (committed)
sources.json provenance: URL, publisher, licence, retrieval date, extracted pages
previews/    scratch output of eval/label_tool.py (gitignored)
```

## Label format

```json
{"input": "raw/<file>", "page": 1, "attrs": {"lang": "en", "kind": "scan"},
 "width": 7200, "height": 4900, "dpi": 200, "route": "raster",
 "regions": [{"class": "title_block", "box": [x0, y0, x1, y1]},
             {"class": "text_callout", "polygon": [[x, y], [x, y], [x, y], [x, y]]}]}
```

Coordinates are full-resolution pixels of the page **as the pipeline renders it** (`src.document.load_pages`
with `configs/inference.yaml`), so `width`/`height`/`dpi` must match a fresh render; `eval/label_tool.py` checks
this. Classes and labelling rules are the same as for the synthetic data (`data_gen/components.py` docstring);
drawing lists and other ruled data tables count as `schedule_table`.

## How the labels were made

Single-pass annotation by an AI assistant with `eval/label_tool.py`: coordinates read from gridded,
zoomed renders and checked on overlays. **No human review and no second annotator**, so treat the real-set
numbers as indicative: boxes are approximate (about ±1% of the page), and on very dense plans not every
callout is labelled (the agents labelled at most ~40 per sheet; see the per-page notes in `DATA.md`). Table
cell structure is not annotated on real pages, so the real set measures detection only.

```bash
python -m eval.label_tool init  data/real raw/<file> --page 1
python -m eval.label_tool grid  data/real labels/<stem>_p1.json
python -m eval.label_tool check data/real labels/<stem>_p1.json
python -m eval.evaluate --real data/real --weights models/planscope_obb.pt --name real
```
