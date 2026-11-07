# Evaluation: oracle_tables

- pages: 46  ·  settings: `{"weights": null, "oracle": true, "tiling": true, "full_pass": true, "tile": 640, "imgsz": 1024, "full_imgsz": 1024, "conf": 0.25, "detect_dpi": 150, "tables": true, "data": "data\\synth", "split": "test"}`
- **mAP50 1.000**  ·  mAP50:95 1.000  ·  median page time 1.153 s

| class | GT | P | R | F1 | AP50 | AP50:95 |
|---|---|---|---|---|---|---|
| title_block | 46 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| schedule_table | 150 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| legend | 81 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| notes_block | 113 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| text_callout | 1406 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

**By mode**

| value | pages | mAP50 | mAP50:95 | callout recall |
|---|---|---|---|---|
| outlined | 9 | 1.000 | 1.000 | 1.000 |
| scan | 17 | 1.000 | 1.000 | 1.000 |
| vector | 20 | 1.000 | 1.000 | 1.000 |

**By lang**

| value | pages | mAP50 | mAP50:95 | callout recall |
|---|---|---|---|---|
| en | 14 | 1.000 | 1.000 | 1.000 |
| ja | 32 | 1.000 | 1.000 | 1.000 |

**By sheet_size**

| value | pages | mAP50 | mAP50:95 | callout recall |
|---|---|---|---|---|
| A1 | 8 | 1.000 | 1.000 | 1.000 |
| A2 | 13 | 1.000 | 1.000 | 1.000 |
| A3 | 25 | 1.000 | 1.000 | 1.000 |

**By layout**

| value | pages | mAP50 | mAP50:95 | callout recall |
|---|---|---|---|---|
| plan_only | 11 | 1.000 | 1.000 | 1.000 |
| plan_side | 28 | 1.000 | 1.000 | 1.000 |
| schedule_sheet | 7 | 1.000 | 1.000 | – |

**Tables** (matched GT schedules, IoU ≥ 0.5)

| matched / GT | cell P | cell R | cell F1 | span acc | text exact | header acc | exact grid |
|---|---|---|---|---|---|---|---|
| 150 / 150 | 0.972 | 0.898 | 0.933 | 1.000 | 1.000 | 1.000 | 0.860 |

**Confusion** (rows GT, columns predicted, IoU ≥ 0.5, class-agnostic matching)

| GT \ pred | title_block | schedule_table | legend | notes_block | text_callout | missed |
|---|---|---|---|---|---|---|
| title_block | 46 | 0 | 0 | 0 | 0 | 0 |
| schedule_table | 0 | 150 | 0 | 0 | 0 | 0 |
| legend | 0 | 0 | 81 | 0 | 0 | 0 |
| notes_block | 0 | 0 | 0 | 113 | 0 | 0 |
| text_callout | 0 | 0 | 0 | 0 | 1406 | 0 |
| background | 0 | 0 | 0 | 0 | 0 | 0 |
