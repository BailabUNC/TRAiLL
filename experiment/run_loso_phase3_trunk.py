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
    parser.add_argument("--epochs", type=int, default=50, help="Default is tuned for stronger fine-tuning.")
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-head", type=float, default=3e-3)
    parser.add_argument("--lr-enc", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument(
        "--unfreeze-from",
        type=int,
        default=0,
        help="Encoder block index to start unfreezing (0–4). 0 = full trunk; 3 = last two blocks only (conservative).",
    )
    parser.add_argument(
        "--stage2-epoch",
        type=int,
        default=None,
        help="Progressive unfreeze switch epoch (>=2). Disabled when omitted.",
    )
    parser.add_argument(
        "--stage2-unfreeze-from",
        type=int,
        default=None,
        help="Encoder block index for stage2 unfreeze (0–4). Requires --stage2-epoch.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Clip joint encoder+head gradients; 0 disables.",
    )
    parser.add_argument(
        "--conservative",
        action="store_true",
        help="Use the original mild recipe (partial unfreeze, higher WD, smaller head).",
    )
    parser.add_argument(
        "--val-select",
        choices=("mean", "min_subj"),
        default="min_subj",
        help="Early-stop score on inner val: pooled mean acc, or min acc across train subjects (closer to LOSO stress).",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="Cross-entropy label smoothing (0 disables). Mild default to reduce overconfident val fits.",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.12,
        help="Beta(alpha,alpha) mixup on pooled trunk features during train; 0 disables.",
    )
    parser.add_argument(
        "--class-weighted-ce",
        action="store_true",
        help="Enable inverse-frequency class-weighted cross-entropy on inner-train labels.",
    )
    parser.add_argument(
        "--class-weight-power",
        type=float,
        default=1.0,
        help="Exponent for inverse-frequency class weights (1.0 = exact inverse frequency).",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--run-tag",
        type=str,
        default="",
        help="Optional subfolder under outputs/phase3/runs/<tag>/arrays to avoid overwriting default outputs.",
    )
    args = parser.parse_args()

    if args.conservative:
        args.epochs = 20
        args.patience = 5
        args.lr_head = 1e-3
        args.lr_enc = 1e-4
        args.weight_decay = 3e-4
        args.dropout = 0.3
        args.hidden_dim = 128
        args.unfreeze_from = 3
        args.stage2_epoch = None
        args.stage2_unfreeze_from = None
        args.max_grad_norm = 0.0
        args.val_select = "mean"
        args.label_smoothing = 0.0
        args.mixup_alpha = 0.0
        args.class_weighted_ce = False
        args.class_weight_power = 1.0

    if args.smoke:
        args.epochs = 1
        args.patience = 1

    if (args.stage2_epoch is None) != (args.stage2_unfreeze_from is None):
        raise ValueError("Provide both --stage2-epoch and --stage2-unfreeze-from together, or neither.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loso_dir = ROOT / args.loso_dir
    out_dir = (
        ROOT / "outputs" / "phase3" / "runs" / args.run_tag / "arrays"
        if args.run_tag
        else ROOT / "outputs" / "phase3" / "arrays"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    max_grad_norm = args.max_grad_norm if args.max_grad_norm > 0 else None

    summary = {
        "phase": "phase3_trunk",
        "device": str(device),
        "smoke": bool(args.smoke),
        "conservative": bool(args.conservative),
        "hparams": {
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr_head": args.lr_head,
            "lr_enc": args.lr_enc,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "hidden_dim": args.hidden_dim,
            "unfreeze_from": args.unfreeze_from,
            "stage2_epoch": args.stage2_epoch,
            "stage2_unfreeze_from": args.stage2_unfreeze_from,
            "max_grad_norm": max_grad_norm,
            "val_select": args.val_select,
            "label_smoothing": args.label_smoothing,
            "mixup_alpha": args.mixup_alpha,
            "class_weighted_ce": bool(args.class_weighted_ce),
            "class_weight_power": args.class_weight_power,
        },
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
            lr_head=args.lr_head,
            lr_enc=args.lr_enc,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            hidden_dim=args.hidden_dim,
            unfreeze_from=args.unfreeze_from,
            stage2_epoch=args.stage2_epoch,
            stage2_unfreeze_from=args.stage2_unfreeze_from,
            max_grad_norm=max_grad_norm,
            val_subject_ids=inner_val["subject_ids"],
            val_select_metric=args.val_select,
            label_smoothing=args.label_smoothing,
            mixup_alpha=args.mixup_alpha,
            class_weighted_ce=bool(args.class_weighted_ce),
            class_weight_power=args.class_weight_power,
            test_instance_ids=outer_test.get("instance_ids"),
            val_instance_ids=inner_val.get("instance_ids"),
        )

        fold_result = {
            "phase3_trunk": metrics,
            "encoder_ckpt": str(enc_ckpt.relative_to(ROOT)),
            "outer_test_subject_ids": sorted(set(outer_test["subject_ids"].tolist())),
            "inner_train_samples": int(inner_train["features"].shape[0]),
            "inner_val_samples": int(inner_val["features"].shape[0]),
            "outer_test_samples": int(outer_test["features"].shape[0]),
        }
        td = metrics.get("test_diagnostics") or {}
        mb = td.get("majority_baseline") or {}
        pc = td.get("prediction_collapse") or {}
        print(
            f"  [diagnostics] random_baseline≈{td.get('random_uniform_baseline_acc', 0):.3f} "
            f"majority_baseline={mb.get('majority_baseline_test_acc', 0):.3f} "
            f"top_pred_freq={pc.get('top_pred_class_freq', 0):.3f}"
        )
        summary["folds"][fold] = fold_result
        with (out_dir / f"loso_{fold}_phase3_trunk_results.json").open("w", encoding="utf-8") as f:
            json.dump(fold_result, f, indent=2)

    wacc = [summary["folds"][f]["phase3_trunk"]["window_acc"] for f in ("fold1", "fold2", "fold3")]
    ifacc = [summary["folds"][f]["phase3_trunk"]["instance_acc"] for f in ("fold1", "fold2", "fold3")]
    peak_tr = [
        summary["folds"][f]["phase3_trunk"]["train_window_acc_peak_during_fit"]
        for f in ("fold1", "fold2", "fold3")
    ]
    summary["aggregate"] = {
        "window_acc_mean": float(np.mean(wacc)),
        "window_acc_std": float(np.std(wacc)),
        "instance_acc_mean": float(np.mean(ifacc)),
        "instance_acc_std": float(np.std(ifacc)),
        "train_window_acc_peak_mean": float(np.mean(peak_tr)),
        "train_window_acc_peak_std": float(np.std(peak_tr)),
    }
    with (out_dir / "loso_phase3_trunk_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nPhase III (trunk) LOSO completed.")


if __name__ == "__main__":
    main()
