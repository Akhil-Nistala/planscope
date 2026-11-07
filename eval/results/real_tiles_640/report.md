# Evaluation: real_tiles_640

- pages: 23  ·  settings: `{"weights": "models/tiles_640.pt", "oracle": false, "tiling": true, "full_pass": true, "tile": 640, "imgsz": 640, "full_imgsz": 1024, "conf": 0.25, "detect_dpi": 150, "tables": true, "data": "data\\real", "split": null}`
- **mAP50 0.012**  ·  mAP50:95 0.004  ·  median page time 9.364 s

| class | GT | P | R | F1 | AP50 | AP50:95 |
|---|---|---|---|---|---|---|
| title_block | 19 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| schedule_table | 18 | 0.025 | 0.167 | 0.043 | 0.018 | 0.004 |
| legend | 7 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| notes_block | 38 | 0.015 | 0.211 | 0.028 | 0.015 | 0.006 |
| text_callout | 351 | 0.123 | 0.140 | 0.131 | 0.025 | 0.008 |

**By lang**

| value | pages | mAP50 | mAP50:95 | callout recall |
|---|---|---|---|---|
| en | 13 | 0.015 | 0.005 | 0.126 |
| ja | 10 | 0.019 | 0.005 | 0.178 |

**By kind**

| value | pages | mAP50 | mAP50:95 | callout recall |
|---|---|---|---|---|
| scan | 13 | 0.015 | 0.005 | 0.126 |
| vector | 10 | 0.019 | 0.005 | 0.178 |

**Tables** (matched GT schedules, IoU ≥ 0.5)

| matched / GT | cell P | cell R | cell F1 | span acc | text exact | header acc | exact grid |
|---|---|---|---|---|---|---|---|
| 3 / 18 | – | – | – | – | – | – | 0.000 |

**Confusion** (rows GT, columns predicted, IoU ≥ 0.5, class-agnostic matching)

| GT \ pred | title_block | schedule_table | legend | notes_block | text_callout | missed |
|---|---|---|---|---|---|---|
| title_block | 0 | 1 | 0 | 5 | 0 | 13 |
| schedule_table | 0 | 3 | 0 | 1 | 0 | 14 |
| legend | 0 | 0 | 0 | 1 | 1 | 5 |
| notes_block | 0 | 1 | 0 | 8 | 0 | 29 |
| text_callout | 0 | 0 | 1 | 31 | 47 | 272 |
| background | 0 | 116 | 11 | 483 | 349 | 0 |
