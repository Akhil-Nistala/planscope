# Evaluation: real_full_640

- pages: 23  ·  settings: `{"weights": "models/full_640.pt", "oracle": false, "tiling": false, "full_pass": true, "tile": 640, "imgsz": 640, "full_imgsz": 640, "conf": 0.25, "detect_dpi": 150, "tables": true, "data": "data\\real", "split": null}`
- **mAP50 0.060**  ·  mAP50:95 0.028  ·  median page time 1.022 s

| class | GT | P | R | F1 | AP50 | AP50:95 |
|---|---|---|---|---|---|---|
| title_block | 19 | 0.167 | 0.053 | 0.080 | 0.026 | 0.021 |
| schedule_table | 18 | 0.070 | 0.500 | 0.122 | 0.105 | 0.049 |
| legend | 7 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| notes_block | 38 | 0.118 | 0.263 | 0.163 | 0.136 | 0.060 |
| text_callout | 351 | 0.303 | 0.057 | 0.096 | 0.032 | 0.012 |

**By lang**

| value | pages | mAP50 | mAP50:95 | callout recall |
|---|---|---|---|---|
| en | 13 | 0.107 | 0.053 | 0.061 |
| ja | 10 | 0.010 | 0.006 | 0.044 |

**By kind**

| value | pages | mAP50 | mAP50:95 | callout recall |
|---|---|---|---|---|
| scan | 13 | 0.107 | 0.053 | 0.061 |
| vector | 10 | 0.010 | 0.006 | 0.044 |

**Tables** (matched GT schedules, IoU ≥ 0.5)

| matched / GT | cell P | cell R | cell F1 | span acc | text exact | header acc | exact grid |
|---|---|---|---|---|---|---|---|
| 9 / 18 | – | – | – | – | – | – | 0.000 |

**Confusion** (rows GT, columns predicted, IoU ≥ 0.5, class-agnostic matching)

| GT \ pred | title_block | schedule_table | legend | notes_block | text_callout | missed |
|---|---|---|---|---|---|---|
| title_block | 0 | 11 | 0 | 4 | 0 | 4 |
| schedule_table | 0 | 8 | 1 | 2 | 0 | 7 |
| legend | 0 | 1 | 0 | 2 | 0 | 4 |
| notes_block | 0 | 2 | 0 | 10 | 1 | 25 |
| text_callout | 0 | 0 | 0 | 2 | 20 | 329 |
| background | 6 | 107 | 9 | 65 | 45 | 0 |
