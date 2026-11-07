# Small-scale CPU run of Milestones 3-6 (what produced the committed results). Sequential on purpose:
# a 4-core / 8 GB laptop cannot train and evaluate at the same time. Full scale: notebooks/train_colab.ipynb.
# Run from the repo root:  powershell -ExecutionPolicy Bypass -File scripts\run_small.ps1
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:YOLO_AUTOINSTALL = "false"
$py = ".\.venv\Scripts\python.exe"

function Step($name, [scriptblock]$cmd) {
    $t = Get-Date
    Write-Output "=== $name  ($(Get-Date -Format HH:mm:ss))"
    & $cmd 2>&1 | Out-File -Append -Encoding utf8 "logs\$name.log"
    Write-Output "    done in $([int]((Get-Date) - $t).TotalMinutes) min, exit $LASTEXITCODE"
}
New-Item -ItemType Directory -Force logs | Out-Null

# M5: table structure with GT boxes (no detector)
if (-not (Test-Path eval\results\oracle_tables\metrics.json)) {
    Step "eval_oracle" { & $py -m eval.evaluate --data data/synth --split test --oracle --name oracle_tables }
}
# M3: baseline (whole sheets letterboxed to 640) and tile model (640-px tiles at 150 DPI)
Step "train_full" { & $py -m training.train --data data/synth/dataset.yaml --name full_640 --imgsz 640 --epochs 4 --batch 4 --workers 0 --fraction 0.6 --patience 4 }
Step "train_tiles" { & $py -m training.train --data data/synth_tiles/dataset.yaml --name tiles_640 --imgsz 640 --epochs 5 --batch 4 --workers 0 --fraction 0.2 --patience 5 }
# M4/M6: end-to-end evaluation + ablation on the synthetic test split
Step "eval_full_notile" { & $py -m eval.evaluate --data data/synth --split test --weights models/full_640.pt --no-tiling --imgsz 640 --name full_640_whole_sheet }
Step "eval_tiles_tiled_full" { & $py -m eval.evaluate --data data/synth --split test --weights models/tiles_640.pt --name tiles_640_tiled_fullpass }
Step "eval_tiles_tiled_only" { & $py -m eval.evaluate --data data/synth --split test --weights models/tiles_640.pt --no-full-pass --name tiles_640_tiled_only }
Step "eval_tiles_notile" { & $py -m eval.evaluate --data data/synth --split test --weights models/tiles_640.pt --no-tiling --imgsz 640 --name tiles_640_whole_sheet }
# M4: real held-out set
Step "eval_real_tiles" { & $py -m eval.evaluate --real data/real --weights models/tiles_640.pt --name real_tiles_640 }
Step "eval_real_full" { & $py -m eval.evaluate --real data/real --weights models/full_640.pt --no-tiling --imgsz 640 --name real_full_640 }
# M6: error analysis
Step "errors_synth" { & $py -m eval.errors --run eval/results/tiles_640_tiled_fullpass --data data/synth --n 4 }
Step "errors_real" { & $py -m eval.errors --run eval/results/real_tiles_640 --real data/real --n 4 }
Write-Output "=== all done $(Get-Date -Format HH:mm:ss)"
