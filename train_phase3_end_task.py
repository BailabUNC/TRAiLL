#!/usr/bin/env python3
"""Phase III (manuscript): 5-fold train frozen / partially unfrozen encoder + dropout MLP on baseline numpy windows.

Loads encoder weights from ``PHASE3_ENCODER_BEST`` (expects a dict with an ``\"enc\"`` state_dict).
Trains on ``data/baseline_arrays/filter_features1.npy`` + ``filter_labels1.npy`` by default (same as
``eval/phase3/confusion_and_roc.py``). Saves the best-val fold bundle under ``checkpoints/phase3/`` and
per-fold accuracies under ``outputs/loso/phase3/results/``. Run from repo root.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from path_config import (
    OUTPUTS_PHASE3_ARRAYS,
    PHASE3_BASELINE_FEATURES1,
    PHASE3_BASELINE_LABELS1,
    PHASE3_DIR,
    ROOT,
    first_existing,
    PHASE3_ENCODER_BEST,
)
from tools.phase3_end_task_lib import NpyTSDataset, run_fold_manuscript


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-III end-task 5-fold training (manuscript recipe).")
    parser.add_argument(
        "--features",
        type=str,
        default=first_existing(*PHASE3_BASELINE_FEATURES1),
        help="Path to (N,L,C) feature .npy",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=first_existing(*PHASE3_BASELINE_LABELS1),
        help="Path to (N,) label .npy",
    )
    parser.add_argument(
        "--encoder-ckpt",
        type=str,
        default=first_existing(*PHASE3_ENCODER_BEST),
        help="Full checkpoint containing key 'enc'",
    )
    parser.add_argument(
        "--out-checkpoint-dir",
        type=str,
        default=str(ROOT / PHASE3_DIR),
        help="Directory for best-fold ``end_task_mlp_drop_F*.pth``",
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default=str(OUTPUTS_PHASE3_ARRAYS),
        help="Directory for numpy metrics export",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="Print paths and exit.")
    args = parser.parse_args()

    if args.dry_run:
        print(f"[dry-run] features={args.features}")
        print(f"[dry-run] labels={args.labels}")
        print(f"[dry-run] encoder_ckpt={args.encoder_ckpt}")
        print(f"[dry-run] out_checkpoint_dir={args.out_checkpoint_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ds = NpyTSDataset(args.features, args.labels)
    pack = torch.load(args.encoder_ckpt, map_location=device)
    enc_ckpt = pack["enc"] if isinstance(pack, dict) and "enc" in pack else pack

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    fold_metrics: list[float] = []
    saved_models: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(ds)), ds.y.numpy())):
        acc, state = run_fold_manuscript(
            train_idx,
            val_idx,
            fold,
            ds,
            enc_ckpt,
            device,
            max_epochs=args.max_epochs,
            patience=args.patience,
        )
        fold_metrics.append(acc)
        saved_models.append(state)
        print(f"Fold {fold} best val_acc = {acc:.3%}\n")

    fold_metrics_arr = np.array(fold_metrics, dtype=np.float64)
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    np.save(metrics_dir / "phase3_end_task_manuscript_fold_val_acc.npy", fold_metrics_arr)

    print("========== 5-fold (manuscript) ==========")
    print("Acc per fold:", [f"{a:.3%}" for a in fold_metrics])
    print(f"Mean ± std: {fold_metrics_arr.mean():.3%} ± {fold_metrics_arr.std():.3%}")

    best_fold = int(fold_metrics_arr.argmax())
    out_dir = Path(args.out_checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    acc_best = fold_metrics_arr[best_fold]
    out_path = out_dir / f"end_task_mlp_drop_F{best_fold}_acc{acc_best:.3f}.pth"
    torch.save(saved_models[best_fold], out_path)
    print(f"Saved best-fold bundle to {out_path}")


if __name__ == "__main__":
    main()
