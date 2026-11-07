# DATA.md — datasets and provenance

## 1. Synthetic training data (`data/synth`, generated, not committed)

Produced by `python -m data_gen.generate` (see README → Synthetic training data). Every image, label and GT
file is a deterministic function of `configs/synth.yaml` (seed 1234) and `configs/vocab.yaml`; nothing is
derived from third-party drawings. Fonts are the system's Japanese fonts (Yu Gothic / MS Gothic on Windows,
Noto CJK on Linux) and the PDF base-14 Latin fonts; they are embedded (subset) only in the generated PDFs.

The trained model in `models/` was trained on 400 generated sheets (320 train / 40 val / 40 test split by seed),
cut into 640-px tiles at 150 DPI for the tile model (`training/tile_dataset.py`).

## 2. Real held-out test set (`data/real`)

Used **only for testing**. 23 files, 23 labelled pages, 433 labelled regions
(title_block 19, schedule_table 18, legend 7, notes_block 38, text_callout 351). Raw files are not committed (re-download from the URLs in
`data/real/sources.json`, which also records retrieval date `2026-11-06`, direct file URLs and extracted pages);
labels are committed in `data/real/labels/`.

Licences, as stated by the publishers:

* **US National Archives (NARA), Governors Island drawings (US Coast Guard):** "No Copyright – United States"
  (rightsstatements.org NoC-US/1.0); several sheets were drawn by private A/E firms under contract to the Coast
  Guard and carry the firm's name. NARA marks them NoC-US; we rely on that statement.
* **US Army Corps of Engineers archive:** US federal government work, public domain.
* **Ministry of the Environment Japan (環境省, www.env.go.jp):** 公共データ利用規約 第1.0版 (PDL1.0, CC BY 4.0
  compatible). Drawings prepared by outside consultants; PDL's third-party-rights clause is a residual risk.
* **Kobe City (神戸市):** 政府標準利用規約 第2.0版 (CC BY 4.0 compatible). These are specification documents
  (特記仕様書), not CAD sheets.

Attribution for CC BY-compatible sources: 出典：環境省ウェブサイト / 出典：神戸市ウェブサイト (pages as listed below;
some pages were extracted from larger PDFs, which is an edit).

| file | lang | kind | publisher | licence | landing page | labels |
|---|---|---|---|---|---|---|
| `en_nara_govisland_bldg550_room_finish_schedule_1995.tiff` | en | image | U.S. National Archives and Records Administration | [No Copyright - United States](http://rightsstatements.org/vocab/NoC-US/1.0/) | [source](https://catalog.archives.gov/id/167074085) | TB 1 ST 3 NB 2 |
| `en_nara_govisland_bldg550_asbestos_abatement_1f_AA201_1995.tiff` | en | image | U.S. National Archives and Records Administration | [No Copyright - United States](http://rightsstatements.org/vocab/NoC-US/1.0/) | [source](https://catalog.archives.gov/id/167073537) | TB 1 ST 1 LG 1 NB 4 TC 17 |
| `en_nara_govisland_bldg844_general_instructions_asbestos_details_1993.tiff` | en | image | U.S. National Archives and Records Administration | [No Copyright - United States](http://rightsstatements.org/vocab/NoC-US/1.0/) | [source](https://catalog.archives.gov/id/167076991) | TB 1 NB 2 TC 12 |
| `en_nara_govisland_bldg9_finish_schedule_notes_1993.tiff` | en | image | U.S. National Archives and Records Administration | [No Copyright - United States](http://rightsstatements.org/vocab/NoC-US/1.0/) | [source](https://catalog.archives.gov/id/167062549) | TB 1 ST 1 NB 1 |
| `en_nara_govisland_bldg125_ops_room_door_finish_schedules_1979.tiff` | en | image | U.S. National Archives and Records Administration | [No Copyright - United States](http://rightsstatements.org/vocab/NoC-US/1.0/) | [source](https://catalog.archives.gov/id/167066217) | TB 1 ST 2 NB 2 TC 42 |
| `en_nara_govisland_bldg315_title_sheet_legend_demolition_notes_1986.tiff` | en | image | U.S. National Archives and Records Administration | [No Copyright - United States](http://rightsstatements.org/vocab/NoC-US/1.0/) | [source](https://catalog.archives.gov/id/167068733) | TB 1 ST 1 LG 1 NB 2 |
| `en_nara_govisland_bldg125_interior_renovation_demolition_plan_1990.tiff` | en | image | U.S. National Archives and Records Administration | [No Copyright - United States](http://rightsstatements.org/vocab/NoC-US/1.0/) | [source](https://catalog.archives.gov/id/167066091) | TB 1 LG 1 NB 1 TC 18 |
| `en_nara_govisland_bldg253_demolition_asbestos_removal_plot_plan_1991.tiff` | en | image | U.S. National Archives and Records Administration | [No Copyright - United States](http://rightsstatements.org/vocab/NoC-US/1.0/) | [source](https://catalog.archives.gov/id/167067777) | TB 1 TC 45 |
| `en_nara_govisland_bldg135A_renovation_plan_schedule_legend_1977.tiff` | en | image | U.S. National Archives and Records Administration | [No Copyright - United States](http://rightsstatements.org/vocab/NoC-US/1.0/) | [source](https://catalog.archives.gov/id/167066833) | TB 1 ST 2 LG 1 TC 19 |
| `en_nara_govisland_brickvillage_ph3_unit_plans_room_finish_1983.tiff` | en | image | U.S. National Archives and Records Administration | [No Copyright - United States](http://rightsstatements.org/vocab/NoC-US/1.0/) | [source](https://catalog.archives.gov/id/167061297) | TB 1 ST 1 NB 2 TC 58 |
| `en_nara_govisland_bldg333_courtroom_demolition_keyplan_notes.tiff` | en | image | U.S. National Archives and Records Administration | [No Copyright - United States](http://rightsstatements.org/vocab/NoC-US/1.0/) | [source](https://catalog.archives.gov/id/167069219) | TB 1 LG 1 NB 1 TC 16 |
| `en_nara_govisland_bldg333_interior_alterations_finish_schedule_1987.tiff` | en | image | U.S. National Archives and Records Administration | [No Copyright - United States](http://rightsstatements.org/vocab/NoC-US/1.0/) | [source](https://catalog.archives.gov/id/167069323) | TB 1 ST 1 NB 4 TC 34 |
| `en_usace_af_dormitory_21-01-101_sheet006_room_finish_schedule.jpeg` | en | image | U.S. Army Corps of Engineers | [Public domain](https://commons.wikimedia.org/wiki/Template:PD-USGov-Military-Army-USACE) | [source](https://usace.contentdm.oclc.org/digital/collection/p16021coll10/id/14865) | TB 1 ST 1 NB 2 |
| `jp_env_kokyogaien_gosui_jokashisetsu_kaishu_p02.pdf` | ja | vector pdf | 環境省 自然環境局 皇居外苑管理事務所 | [公共データ利用規約（第1.0版）(PDL1.0; CC BY 4.0 compatible)](https://www.env.go.jp/mail.html) | [source](https://www.env.go.jp/garden/kokyogaien/4_sup/index.html) | ST 1 |
| `jp_env_kokyogaien_gosui_jokashisetsu_kaishu_p03.pdf` | ja | vector pdf | 環境省 自然環境局 皇居外苑管理事務所 | [公共データ利用規約（第1.0版）(PDL1.0; CC BY 4.0 compatible)](https://www.env.go.jp/mail.html) | [source](https://www.env.go.jp/garden/kokyogaien/4_sup/index.html) | TB 1 NB 4 |
| `jp_env_kokyogaien_gosui_jokashisetsu_kaishu_p05.pdf` | ja | vector pdf | 環境省 自然環境局 皇居外苑管理事務所 | [公共データ利用規約（第1.0版）(PDL1.0; CC BY 4.0 compatible)](https://www.env.go.jp/mail.html) | [source](https://www.env.go.jp/garden/kokyogaien/4_sup/index.html) | TB 1 ST 2 LG 1 TC 63 |
| `jp_env_kokyogaien_gosui_jokashisetsu_kaishu_p06.pdf` | ja | vector pdf | 環境省 自然環境局 皇居外苑管理事務所 | [公共データ利用規約（第1.0版）(PDL1.0; CC BY 4.0 compatible)](https://www.env.go.jp/mail.html) | [source](https://www.env.go.jp/garden/kokyogaien/4_sup/index.html) | TB 1 TC 25 |
| `jp_env_kyotogyoen_nakayamatei_koshibei_kaishu_p01.pdf` | ja | vector pdf | 環境省 京都御苑管理事務所 | [公共データ利用規約（第1.0版）(PDL1.0; CC BY 4.0 compatible)](https://www.env.go.jp/mail.html) | [source](https://www.env.go.jp/garden/kyotogyoen/4_sup/sup_00050.html) | TB 1 ST 1 |
| `jp_env_kyotogyoen_nakayamatei_koshibei_kaishu_p02.pdf` | ja | vector pdf | 環境省 京都御苑管理事務所 | [公共データ利用規約（第1.0版）(PDL1.0; CC BY 4.0 compatible)](https://www.env.go.jp/mail.html) | [source](https://www.env.go.jp/garden/kyotogyoen/4_sup/sup_00050.html) | TB 1 NB 4 |
| `jp_env_kyotogyoen_nakayamatei_koshibei_kaishu_p13.pdf` | ja | vector pdf | 環境省 京都御苑管理事務所 | [公共データ利用規約（第1.0版）(PDL1.0; CC BY 4.0 compatible)](https://www.env.go.jp/mail.html) | [source](https://www.env.go.jp/garden/kyotogyoen/4_sup/sup_00050.html) | TB 1 ST 1 LG 1 NB 1 TC 2 |
| `jp_kobe_kenchiku_kaishu_tokkishiyosho_p01.pdf` | ja | vector pdf | 神戸市 建築技術管理委員会 | [政府標準利用規約（第2.0版）準拠](https://www.city.kobe.lg.jp/documents/19135/20170630041802-1.pdf) | [source](https://www.city.kobe.lg.jp/a31253/kurashi/machizukuri/institution/kentikugikan/tokkishiyousyo_eizen.html) | NB 2 |
| `jp_kobe_kenchiku_kaishu_tokkishiyosho_p48.pdf` | ja | vector pdf | 神戸市 建築技術管理委員会 | [政府標準利用規約（第2.0版）準拠](https://www.city.kobe.lg.jp/documents/19135/20170630041802-1.pdf) | [source](https://www.city.kobe.lg.jp/a31253/kurashi/machizukuri/institution/kentikugikan/tokkishiyousyo_eizen.html) | NB 2 |
| `jp_kobe_ishiwata_shori_tokkishiyosho_p01-04.pdf` | ja | vector pdf | 神戸市 建築技術管理委員会 | [政府標準利用規約（第2.0版）準拠](https://www.city.kobe.lg.jp/documents/19135/20170630041802-1.pdf) | [source](https://www.city.kobe.lg.jp/a31253/kurashi/machizukuri/institution/kentikugikan/tokkishiyousyo_eizen.html) | NB 2 |

Label abbreviations: TB title_block, ST schedule_table, LG legend, NB notes_block, TC text_callout.

**Known gaps of the real set.** No Japanese sheet with a full 仕上表 (finish schedule) was found under a clean
licence (candidates whose sites prohibit reuse were rejected; see `data/real/sources.json` notes and the
sourcing report in the README). English sheets are all bilevel 1970s–1990s scans; the Japanese pages are
vector PDFs, two of them A3 spec documents rather than drawings. 23 pages is small: per-class numbers on the
real set have wide confidence intervals.

**How the labels were made.** Single-pass annotation by an AI assistant using `eval/label_tool.py`
(gridded/zoomed renders, overlays checked; Japanese vector pages snapped to text-layer coordinates). No human
review, no second annotator. Conventions applied consistently across annotators: a table's own title row is
excluded even when drawn inside its frame; abbreviation lists without symbols are notes_block; drawing lists
are schedule_table; site/area names on plans are text_callout; dimension strings, grid bubbles, view titles,
north arrows and archive stamps are not labelled. Table cell structure is not annotated on real pages.
