# data/

Scripts only — no datasets are committed (see `.gitignore`).

| Script | Output | Purpose |
|---|---|---|
| `make_samples.py` | `data/samples/` | 6 deterministic sample drawings covering every router path (vector JP/EN, scan PDF, scan PNG, CAD outlined-text, mixed multi-page). Used by the demo site and smoke tests. |
| `python -m data_gen.generate` | `data/synth/` | 3,000 synthetic sheets (default): images, YOLO-OBB labels, schema-valid GT JSON and per-sheet metadata. Settings: `configs/synth.yaml` |
| *(Milestone 4)* real-data scripts | `data/real/` | Download/annotation conversion for the held-out real test set (provenance in `DATA.md`) |

```bash
python data/make_samples.py --out data/samples --seed 42
python -m data_gen.generate --out data/synth            # add --n 50 for a quick run, --overwrite to regenerate
python -m data_gen.preview --root data/synth --n 8 --cells
```

`data/synth/` layout:

```
dataset.yaml                     Ultralytics OBB dataset file (absolute path + class names)
images/{train,val,test}/<id>.png|jpg
labels/{train,val,test}/<id>.txt  class x1 y1 x2 y2 x3 y3 x4 y4  (normalised, clockwise from the box's top-left)
gt/<id>.json                     ground truth in the pipeline's output format (src/schema.json)
meta/<id>.json                   generation parameters, router verdict, true text of every region/cell
pdf/<id>.pdf                     source PDF (vector and outlined sheets)
manifest.jsonl                   one line per sheet
summary.json                     counts per split / mode / language / class, router agreement, failures
previews/                        written by data_gen.preview
```

GT coordinates are pixels of the stored image at its DPI (`dpi` in `configs/synth.yaml`: 150 for vector sheets,
the same as the pipeline's `render.vector_dpi`, and 200 for scans and outlined sheets).
