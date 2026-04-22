from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from path_config import ROOT
from experiment.loso.loso_phase_utils import load_split, remap_labels


def _flatten(x_t_c: torch.Tensor) -> np.ndarray:
    return x_t_c.reshape(x_t_c.shape[0], -1).cpu().numpy()


def _fold_metrics(
    inner_train: Dict[str, torch.Tensor],
    inner_val: Dict[str, torch.Tensor],
    outer_test: Dict[str, torch.Tensor],
    hidden_dim: int,
    max_iter: int,
    seed: int,
) -> Dict[str, object]:
    tr_y, _va_y, te_y = remap_labels(inner_train["labels"], inner_val["labels"], outer_test["labels"])
    x_tr = _flatten(inner_train["features"])
    x_te = _flatten(outer_test["features"])
    y_tr = tr_y.numpy()
    y_te = te_y.numpy()

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(hidden_dim,),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    max_iter=max_iter,
                    random_state=seed,
                    early_stopping=True,
                    n_iter_no_change=20,
                    validation_fraction=0.2,
                ),
            ),
        ]
    )
    clf.fit(x_tr, y_tr)
    pred = clf.predict(x_te)
    cm = confusion_matrix(y_te, pred, labels=np.arange(24))
    per_class_f1 = f1_score(y_te, pred, labels=np.arange(24), average=None, zero_division=0)
    return {
        "window_acc": float(accuracy_score(y_te, pred)),
        "window_macro_f1": float(f1_score(y_te, pred, average="macro", zero_division=0)),
        "instance_acc": float(accuracy_score(y_te, pred)),
        "instance_macro_f1": float(f1_score(y_te, pred, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "per_class_f1": per_class_f1.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LOSO naive MLP baseline on data/loso splits.")
    parser.add_argument("--loso-dir", type=str, default="data/loso")
    parser.add_argument("--out-dir", type=str, default="outputs/loso/naive/results")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    loso_dir = ROOT / args.loso_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "phase": "naive_mlp",
        "folds": {},
        "hparams": {
            "hidden_dim": args.hidden_dim,
            "max_iter": args.max_iter,
            "seed": args.seed,
        },
    }

    for fold in ("fold1", "fold2", "fold3"):
        inner_train = load_split(loso_dir / f"{fold}_inner_train.pt")
        inner_val = load_split(loso_dir / f"{fold}_inner_val.pt")
        outer_test = load_split(loso_dir / f"{fold}_outer_test.pt")
        metrics = _fold_metrics(
            inner_train=inner_train,
            inner_val=inner_val,
            outer_test=outer_test,
            hidden_dim=args.hidden_dim,
            max_iter=args.max_iter,
            seed=args.seed,
        )
        payload = {
            "naive_mlp": metrics,
            "outer_test_subject_ids": sorted(set(outer_test["subject_ids"].tolist())),
            "inner_train_samples": int(inner_train["features"].shape[0]),
            "inner_val_samples": int(inner_val["features"].shape[0]),
            "outer_test_samples": int(outer_test["features"].shape[0]),
        }
        summary["folds"][fold] = payload
        with (out_dir / f"loso_{fold}_naive_mlp_results.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    accs: List[float] = [summary["folds"][f]["naive_mlp"]["window_acc"] for f in ("fold1", "fold2", "fold3")]
    f1s: List[float] = [summary["folds"][f]["naive_mlp"]["window_macro_f1"] for f in ("fold1", "fold2", "fold3")]
    summary["aggregate"] = {
        "window_acc_mean": float(np.mean(accs)),
        "window_acc_std": float(np.std(accs)),
        "window_macro_f1_mean": float(np.mean(f1s)),
        "window_macro_f1_std": float(np.std(f1s)),
    }
    with (out_dir / "loso_naive_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote naive LOSO summary to {out_dir / 'loso_naive_summary.json'}")


if __name__ == "__main__":
    main()
