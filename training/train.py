"""Train a YOLO11-OBB layout detector (Ultralytics) and export the weights to ``models/``.

Usage::

    # full-sheet baseline: whole sheets letterboxed to imgsz
    python -m training.train --data data/synth/dataset.yaml --name full_640 --imgsz 640 --epochs 30
    # tile model: fixed-DPI tiles from training/tile_dataset.py
    python -m training.train --data data/synth_tiles/dataset.yaml --name tiles_640 --imgsz 640 --epochs 20

Writes ``runs/obb/<name>/`` (Ultralytics run dir), ``models/<name>.pt`` (best weights) and
``models/<name>.json`` (training settings + final validation metrics).

Augmentation is tuned for drawings: no horizontal/vertical flips (mirrored text is not a drawing),
mild scale jitter (the detector sees pages at a fixed DPI), no colour jitter (greyscale line work).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from pathlib import Path
from typing import Sequence

os.environ.setdefault("YOLO_AUTOINSTALL", "false")  # never pip-install from inside training (opencv clash)

REPO = Path(__file__).resolve().parents[1]

AUGMENT = dict(fliplr=0.0, flipud=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.2, scale=0.25, translate=0.1,
               degrees=0.0, mosaic=1.0, mixup=0.0)


def make_subset(data_yaml: Path, n_train: int, n_val: int, seed: int) -> Path:
    """Write a dataset yaml whose train/val are random image lists (unlike ``fraction``, which takes the
    first files, i.e. the first sheets). Used for chunked CPU training: a new train sample per chunk, a
    fixed val sample (seed 0) so chunks are comparable."""
    import random

    import yaml

    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(d["path"])
    exts = {".png", ".jpg", ".jpeg"}

    def sample(split: str, n: int, s: int) -> Path:
        files = sorted(str(p) for p in (root / d[split]).iterdir() if p.suffix.lower() in exts)
        if n and n < len(files):
            files = random.Random(s).sample(files, n)
        lst = root / f"_{split}_subset.txt"
        lst.write_text("\n".join(files) + "\n", encoding="utf-8")
        return lst

    d["train"] = str(sample("train", n_train, seed))
    d["val"] = str(sample("val", n_val, 0))
    out = root / "_subset.yaml"
    out.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    return out


def last_metrics(run_dir: Path) -> dict[str, float]:
    """Final row of Ultralytics' results.csv (validation metrics per epoch)."""
    f = run_dir / "results.csv"
    if not f.exists():
        return {}
    rows = list(csv.DictReader(f.open(encoding="utf-8")))
    return {k.strip(): float(v) for k, v in rows[-1].items() if v.strip()} if rows else {}


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="yolo11n-obb.pt", help="pretrained checkpoint or model yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of the training set to use")
    ap.add_argument("--warmup", type=float, default=3.0, help="warmup epochs (0 when continuing from a checkpoint)")
    ap.add_argument("--subset", type=int, default=0, help="train on N random training images (0 = all)")
    ap.add_argument("--val-subset", type=int, default=0, help="validate on N random val images (0 = all)")
    ap.add_argument("--subset-seed", type=int, default=0, help="seed of the random subset (vary per chunk)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = all cores)")
    ap.add_argument("--project", type=Path, default=REPO / "runs" / "obb")
    args = ap.parse_args(argv)

    import torch
    from ultralytics import YOLO
    from ultralytics.utils import torch_utils

    if args.device == "cpu":
        # Ultralytics resets CPU training to min(8, cpu_count - 1) threads; on a 4-core / 8-thread laptop
        # that oversubscribes the cores (measured slower). Default to physical cores (logical / 2).
        n = args.threads or max(1, (os.cpu_count() or 2) // 2)
        torch_utils.NUM_THREADS = n
        torch.set_num_threads(n)
    t0 = time.time()
    data = args.data.resolve()
    if args.subset:
        data = make_subset(data, args.subset, args.val_subset, args.subset_seed)
    model = YOLO(args.model)
    model.train(data=str(data), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
                workers=args.workers, device=args.device, project=str(args.project), name=args.name,
                exist_ok=True, patience=args.patience, fraction=args.fraction, seed=0, plots=False,
                warmup_epochs=args.warmup,
                amp=args.device != "cpu", verbose=False, **AUGMENT)
    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    (REPO / "models").mkdir(exist_ok=True)
    shutil.copy2(best, REPO / "models" / f"{args.name}.pt")
    meta = {"name": args.name, "data": str(args.data), "base_model": args.model, "imgsz": args.imgsz,
            "epochs_requested": args.epochs, "batch": args.batch, "fraction": args.fraction,
            "subset": args.subset, "val_subset": args.val_subset, "subset_seed": args.subset_seed,
            "device": args.device, "threads": torch.get_num_threads(), "augment": AUGMENT,
            "train_hours": round((time.time() - t0) / 3600, 2), "final_val": last_metrics(run_dir),
            "run_dir": str(run_dir.relative_to(REPO)) if run_dir.is_relative_to(REPO) else str(run_dir)}
    (REPO / "models" / f"{args.name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
