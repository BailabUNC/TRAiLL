from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from path_config import ROOT
from experiment.loso.loso_phase_utils import load_split, leakage_check, train_phase2_fold


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LOSO Phase II only across fold1/2/3.")
    parser.add_argument("--loso-dir", type=str, default="data/loso")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Optional subfolder under outputs/loso/phase2/runs/<tag>/results to avoid overwriting default outputs.",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 1
        args.patience = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loso_dir = ROOT / args.loso_dir
    out_dir = (
        ROOT / "outputs" / "loso" / "phase2" / "runs" / args.run_tag / "results"
        if args.run_tag
        else ROOT / "outputs" / "loso" / "phase2" / "results"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"phase": "phase2", "device": str(device), "smoke": bool(args.smoke), "folds": {}}
    for fold in ("fold1", "fold2", "fold3"):
        print(f"\n===== {fold} / phase2 =====")
        inner_train = load_split(loso_dir / f"{fold}_inner_train.pt")
        inner_val = load_split(loso_dir / f"{fold}_inner_val.pt")
        outer_test = load_split(loso_dir / f"{fold}_outer_test.pt")
        leakage_check(inner_train, inner_val, outer_test)

        phase1_ckpt = ROOT / "checkpoints" / "phase1" / f"loso_{fold}_autoencoder.pth"
        if not phase1_ckpt.exists():
            raise FileNotFoundError(f"Missing phase1 checkpoint for {fold}: {phase1_ckpt}")

        phase2_proj = ROOT / "checkpoints" / "phase2" / "simsiam" / f"loso_{fold}_proj_head.pth"
        phase2_pred = ROOT / "checkpoints" / "phase2" / "simsiam" / f"loso_{fold}_pred_head.pth"
        phase2_enc = ROOT / "checkpoints" / "phase2" / "simsiam" / f"loso_{fold}_encoder.pth"
        metrics = train_phase2_fold(
            inner_train["features"],
            inner_val["features"],
            phase1_ckpt,
            phase2_proj,
            phase2_pred,
            phase2_enc,
            device,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        fold_result = {
            "phase2": metrics,
            "outer_test_subject_ids": sorted(set(outer_test["subject_ids"].tolist())),
            "inner_train_samples": int(inner_train["features"].shape[0]),
            "inner_val_samples": int(inner_val["features"].shape[0]),
        }
        summary["folds"][fold] = fold_result

        with (out_dir / f"loso_{fold}_phase2_results.json").open("w", encoding="utf-8") as f:
            json.dump(fold_result, f, indent=2)

    vals = [summary["folds"][f]["phase2"]["best_val_loss"] for f in ("fold1", "fold2", "fold3")]
    summary["aggregate"] = {"best_val_loss_mean": float(sum(vals) / len(vals))}
    with (out_dir / "loso_phase2_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nPhase II LOSO completed.")


if __name__ == "__main__":
    main()
