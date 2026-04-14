from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from path_config import ROOT
from experiment.loso_phase_utils import load_split, leakage_check, train_phase1_fold


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LOSO Phase I only across fold1/2/3.")
    parser.add_argument("--loso-dir", type=str, default="data/loso")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 1
        args.patience = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loso_dir = ROOT / args.loso_dir
    out_dir = ROOT / "outputs" / "phase1" / "arrays"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"phase": "phase1", "device": str(device), "smoke": bool(args.smoke), "folds": {}}
    for fold in ("fold1", "fold2", "fold3"):
        print(f"\n===== {fold} / phase1 =====")
        inner_train = load_split(loso_dir / f"{fold}_inner_train.pt")
        inner_val = load_split(loso_dir / f"{fold}_inner_val.pt")
        outer_test = load_split(loso_dir / f"{fold}_outer_test.pt")
        leakage_check(inner_train, inner_val, outer_test)

        phase1_ckpt = ROOT / "checkpoints" / "phase1" / f"loso_{fold}_autoencoder.pth"
        metrics = train_phase1_fold(
            inner_train["features"],
            inner_val["features"],
            phase1_ckpt,
            device,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        fold_result = {
            "phase1": metrics,
            "outer_test_subject_ids": sorted(set(outer_test["subject_ids"].tolist())),
            "inner_train_samples": int(inner_train["features"].shape[0]),
            "inner_val_samples": int(inner_val["features"].shape[0]),
        }
        summary["folds"][fold] = fold_result

        with (out_dir / f"loso_{fold}_phase1_results.json").open("w", encoding="utf-8") as f:
            json.dump(fold_result, f, indent=2)

    vals = [summary["folds"][f]["phase1"]["best_val_loss"] for f in ("fold1", "fold2", "fold3")]
    summary["aggregate"] = {"best_val_loss_mean": float(sum(vals) / len(vals))}
    with (out_dir / "loso_phase1_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nPhase I LOSO completed.")


if __name__ == "__main__":
    main()
