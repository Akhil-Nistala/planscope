# Evaluation: full_640_whole_sheet

- pages: 46  ·  settings: `{"weights": "models/full_640.pt", "oracle": false, "tiling": false, "full_pass": true, "tile": 640, "imgsz": 640, "full_imgsz": 640, "conf": 0.25, "detect_dpi": 150, "tables": true, "data": "data\\synth", "split": "test"}`
- **mAP50 0.802**  ·  mAP50:95 0.520  ·  median page time 1.025 s

| class | GT | P | R | F1 | AP50 | AP50:95 |
|---|---|---|---|---|---|---|
| title_block | 46 | 0.824 | 0.609 | 0.700 | 0.596 | 0.476 |
| schedule_table | 150 | 0.851 | 0.953 | 0.899 | 0.944 | 0.728 |
| legend | 81 | 0.931 | 0.827 | 0.876 | 0.821 | 0.459 |
| notes_block | 113 | 0.818 | 0.956 | 0.882 | 0.944 | 0.690 |
| text_callout | 1406 | 0.787 | 0.770 | 0.778 | 0.706 | 0.247 |

**By mode**

| value | pages | mAP50 | mAP50:95 | callout recall |
|---|---|---|---|---|
| outlined | 9 | 0.814 | 0.557 | 0.765 |
| scan | 17 | 0.780 | 0.505 | 0.702 |
| vector | 20 | 0.827 | 0.537 | 0.836 |

**By lang**

| value | pages | mAP50 | mAP50:95 | callout recall |
|---|---|---|---|---|
| en | 14 | 0.775 | 0.513 | 0.761 |
| ja | 32 | 0.815 | 0.528 | 0.773 |

**By sheet_size**

| value | pages | mAP50 | mAP50:95 | callout recall |
|---|---|---|---|---|
| A1 | 8 | 0.673 | 0.427 | 0.559 |
| A2 | 13 | 0.823 | 0.533 | 0.792 |
| A3 | 25 | 0.859 | 0.570 | 0.862 |

**By layout**

| value | pages | mAP50 | mAP50:95 | callout recall |
|---|---|---|---|---|
| plan_only | 11 | 0.847 | 0.601 | 0.770 |
| plan_side | 28 | 0.787 | 0.512 | 0.769 |
| schedule_sheet | 7 | 0.798 | 0.553 | – |

**Tables** (matched GT schedules, IoU ≥ 0.5)

| matched / GT | cell P | cell R | cell F1 | span acc | text exact | header acc | exact grid |
|---|---|---|---|---|---|---|---|
| 143 / 150 | 0.806 | 0.150 | 0.253 | 0.999 | 0.952 | 0.600 | 0.014 |

**Confusion** (rows GT, columns predicted, IoU ≥ 0.5, class-agnostic matching)

| GT \ pred | title_block | schedule_table | legend | notes_block | text_callout | missed |
|---|---|---|---|---|---|---|
| title_block | 27 | 3 | 0 | 0 | 0 | 16 |
| schedule_table | 0 | 142 | 0 | 4 | 0 | 4 |
| legend | 0 | 0 | 67 | 0 | 0 | 14 |
| notes_block | 0 | 1 | 1 | 108 | 0 | 3 |
| text_callout | 0 | 0 | 0 | 0 | 1082 | 324 |
| background | 7 | 22 | 4 | 20 | 293 | 0 |
