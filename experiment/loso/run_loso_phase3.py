from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from path_config import ROOT
from experiment.loso.loso_phase_utils import load_split, leakage_check, train_phase3_and_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LOSO Phase III only across fold1/2/3.")
    parser.add_argument("--loso-dir", type=str, default="data/loso")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--encoder-ckpt", type=str, default="checkpoints/phase3/model_best.pth")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 1
        args.patience = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loso_dir = ROOT / args.loso_dir
    out_dir = ROOT / "outputs" / "loso" / "phase3" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {"phase": "phase3", "device": str(device), "smoke": bool(args.smoke), "folds": {}}
    for fold in ("fold1", "fold2", "fold3"):
        print(f"\n===== {fold} / phase3 =====")
        inner_train = load_split(loso_dir / f"{fold}_inner_train.pt")
        inner_val = load_split(loso_dir / f"{fold}_inner_val.pt")
        outer_test = load_split(loso_dir / f"{fold}_outer_test.pt")
        leakage_check(inner_train, inner_val, outer_test)

        phase3_ckpt = ROOT / "checkpoints" / "phase3" / f"loso_{fold}_end_task.pth"
        metrics = train_phase3_and_eval(
            inner_train["features"],
            inner_train["labels"],
            inner_val["features"],
            inner_val["labels"],
            outer_test["features"],
            outer_test["labels"],
            ROOT / args.encoder_ckpt,
            phase3_ckpt,
            device,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
        )

        fold_result = {
            "phase3": metrics,
            "outer_test_subject_ids": sorted(set(outer_test["subject_ids"].tolist())),
            "inner_train_samples": int(inner_train["features"].shape[0]),
            "inner_val_samples": int(inner_val["features"].shape[0]),
            "outer_test_samples": int(outer_test["features"].shape[0]),
        }
        summary["folds"][fold] = fold_result
        with (out_dir / f"loso_{fold}_results.json").open("w", encoding="utf-8") as f:
            json.dump(fold_result, f, indent=2)

    wacc = [summary["folds"][f]["phase3"]["window_acc"] for f in ("fold1", "fold2", "fold3")]
    ifacc = [summary["folds"][f]["phase3"]["instance_acc"] for f in ("fold1", "fold2", "fold3")]
    summary["aggregate"] = {
        "window_acc_mean": float(np.mean(wacc)),
        "window_acc_std": float(np.std(wacc)),
        "instance_acc_mean": float(np.mean(ifacc)),
        "instance_acc_std": float(np.std(ifacc)),
    }
    with (out_dir / "loso_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nPhase III LOSO completed.")


if __name__ == "__main__":
    main()
