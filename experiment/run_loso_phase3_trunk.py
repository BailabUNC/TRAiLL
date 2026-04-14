from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from path_config import ROOT
from experiment.loso_phase3_trunk import train_phase3_trunk_and_eval
from experiment.loso_phase_utils import load_split, leakage_check


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LOSO Phase III (trunk+pool): uses per-fold Phase-II encoder trunk, not Encoder1DCNN."
    )
    parser.add_argument("--loso-dir", type=str, default="data/loso")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = 1
        args.patience = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loso_dir = ROOT / args.loso_dir
    out_dir = ROOT / "outputs" / "phase3" / "arrays"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "phase": "phase3_trunk",
        "device": str(device),
        "smoke": bool(args.smoke),
        "folds": {},
    }
    for fold in ("fold1", "fold2", "fold3"):
        print(f"\n===== {fold} / phase3-trunk =====")
        inner_train = load_split(loso_dir / f"{fold}_inner_train.pt")
        inner_val = load_split(loso_dir / f"{fold}_inner_val.pt")
        outer_test = load_split(loso_dir / f"{fold}_outer_test.pt")
        leakage_check(inner_train, inner_val, outer_test)

        enc_ckpt = ROOT / "checkpoints" / "phase2" / "simsiam" / f"loso_{fold}_encoder.pth"
        if not enc_ckpt.exists():
            raise FileNotFoundError(
                f"Missing Phase-II encoder for {fold}: {enc_ckpt}\n"
                "Re-run: python experiment/run_loso_phase2.py (writes loso_*_encoder.pth)."
            )

        out_ckpt = ROOT / "checkpoints" / "phase3" / f"loso_{fold}_trunk_end_task.pth"
        metrics = train_phase3_trunk_and_eval(
            inner_train["features"],
            inner_train["labels"],
            inner_val["features"],
            inner_val["labels"],
            outer_test["features"],
            outer_test["labels"],
            enc_ckpt,
            out_ckpt,
            device,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
        )

        fold_result = {
            "phase3_trunk": metrics,
            "encoder_ckpt": str(enc_ckpt.relative_to(ROOT)),
            "outer_test_subject_ids": sorted(set(outer_test["subject_ids"].tolist())),
            "inner_train_samples": int(inner_train["features"].shape[0]),
            "inner_val_samples": int(inner_val["features"].shape[0]),
            "outer_test_samples": int(outer_test["features"].shape[0]),
        }
        summary["folds"][fold] = fold_result
        with (out_dir / f"loso_{fold}_phase3_trunk_results.json").open("w", encoding="utf-8") as f:
            json.dump(fold_result, f, indent=2)

    wacc = [summary["folds"][f]["phase3_trunk"]["window_acc"] for f in ("fold1", "fold2", "fold3")]
    ifacc = [summary["folds"][f]["phase3_trunk"]["instance_acc"] for f in ("fold1", "fold2", "fold3")]
    summary["aggregate"] = {
        "window_acc_mean": float(np.mean(wacc)),
        "window_acc_std": float(np.std(wacc)),
        "instance_acc_mean": float(np.mean(ifacc)),
        "instance_acc_std": float(np.std(ifacc)),
    }
    with (out_dir / "loso_phase3_trunk_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nPhase III (trunk) LOSO completed.")


if __name__ == "__main__":
    main()
