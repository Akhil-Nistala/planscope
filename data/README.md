# data/

Scripts only — no datasets are committed (see `.gitignore`).

| Script | Output | Purpose |
|---|---|---|
| `make_samples.py` | `data/samples/` | 6 deterministic sample drawings covering every router path (vector JP/EN, scan PDF, scan PNG, CAD outlined-text, mixed multi-page). Used by the demo site and smoke tests. |
| *(Milestone 2)* `data_gen/` | `data/synth/` | ~3,000 synthetic sheets + YOLO-OBB labels + GT JSON |
| *(Milestone 4)* real-data scripts | `data/real/` | Download/annotation conversion for the held-out real test set (provenance in `DATA.md`) |

```bash
python data/make_samples.py --out data/samples --seed 42
```
