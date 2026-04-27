from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib as mpl
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiment.loso.loso_phase3_trunk import build_phase2_encoder
from path_config import ROOT

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"] = 8


def to_channels_first(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 1).contiguous()


def load_features_labels(path: Path) -> Dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if "features" not in data or "labels" not in data:
        raise KeyError(f"{path} must contain keys: features, labels")
    x = data["features"].float()
    y = data["labels"].long()
    if x.ndim != 3:
        raise ValueError(f"{path} features must be [N,T,C], got {tuple(x.shape)}")
    if y.ndim != 1:
        raise ValueError(f"{path} labels must be [N], got {tuple(y.shape)}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"{path} features/labels length mismatch")
    return {"features": x, "labels": y}


def build_label_mapping(train_labels: torch.Tensor) -> Dict[int, int]:
    uniq = torch.unique(train_labels).tolist()
    uniq_sorted = sorted(int(v) for v in uniq)
    return {old: i for i, old in enumerate(uniq_sorted)}


def load_fold_classifier(ckpt_path: Path, device: torch.device) -> nn.Sequential:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "clf" not in ckpt:
        raise KeyError(f"{ckpt_path} missing 'clf' state dict.")
    state = ckpt["clf"]
    hidden_dim = state["0.weight"].shape[0]
    in_dim = state["0.weight"].shape[1]
    num_classes = state["3.weight"].shape[0]
    dropout_p = float(state.get("2.p", 0.0))
    clf = nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout_p),
        nn.Linear(hidden_dim, num_classes),
    ).to(device)
    clf.load_state_dict(state)
    clf.eval()
    return clf


def evaluate_fold(
    fold: str,
    dataset_x: torch.Tensor,
    dataset_y: torch.Tensor,
    loso_dir: Path,
    encoder_dir: Path,
    phase3_dir: Path,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    inner_train = load_features_labels(loso_dir / f"{fold}_inner_train.pt")
    old2new = build_label_mapping(inner_train["labels"])

    kept_idx: List[int] = [i for i, yy in enumerate(dataset_y.tolist()) if int(yy) in old2new]
    dropped_idx: List[int] = [i for i, yy in enumerate(dataset_y.tolist()) if int(yy) not in old2new]
    if not kept_idx:
        raise RuntimeError(f"{fold}: no evaluable samples after label-space filtering.")

    idx_tensor = torch.as_tensor(kept_idx, dtype=torch.long)
    x = dataset_x.index_select(0, idx_tensor)
    y_raw = dataset_y.index_select(0, idx_tensor)
    y = torch.tensor([old2new[int(v)] for v in y_raw.tolist()], dtype=torch.long)
    x_cf = to_channels_first(x)
    dl = DataLoader(TensorDataset(x_cf, y), batch_size=batch_size, shuffle=False, drop_last=False)

    enc_ckpt = encoder_dir / f"loso_{fold}_encoder.pth"
    model_ckpt = phase3_dir / f"loso_{fold}_trunk_end_task.pth"
    if not enc_ckpt.exists():
        raise FileNotFoundError(f"Missing encoder ckpt for {fold}: {enc_ckpt}")
    if not model_ckpt.exists():
        raise FileNotFoundError(f"Missing phase3 ckpt for {fold}: {model_ckpt}")

    enc = build_phase2_encoder(device, enc_ckpt)
    state = torch.load(model_ckpt, map_location=device, weights_only=False)
    enc_state = state.get("enc")
    if enc_state is None:
        raise KeyError(f"{model_ckpt} missing 'enc' state dict.")
    enc.load_state_dict(enc_state)
    enc.eval()
    clf = load_fold_classifier(model_ckpt, device)

    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device)
            logits = clf(enc(xb).mean(dim=2))
            pred = logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(pred)
            y_true.extend(yb.tolist())

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    return {
        "fold": fold,
        "num_total": int(dataset_y.shape[0]),
        "num_evaluable": int(len(kept_idx)),
        "num_dropped_oov_labels": int(len(dropped_idx)),
        "coverage": float(len(kept_idx) / max(int(dataset_y.shape[0]), 1)),
        "window_acc": float(accuracy_score(y_true_arr, y_pred_arr)),
        "window_macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "window_bal_acc": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
    }


def evaluate_dataset(
    dataset_rel: str,
    group_name: str,
    loso_dir: Path,
    encoder_dir: Path,
    phase3_dir: Path,
    batch_size: int,
    device: torch.device,
) -> Dict[str, object]:
    ds = load_features_labels(ROOT / dataset_rel)
    x = ds["features"]
    y = ds["labels"]
    folds = ["fold1", "fold2", "fold3"]
    per_fold = [
        evaluate_fold(
            fold=fold,
            dataset_x=x,
            dataset_y=y,
            loso_dir=loso_dir,
            encoder_dir=encoder_dir,
            phase3_dir=phase3_dir,
            batch_size=batch_size,
            device=device,
        )
        for fold in folds
    ]
    return {
        "name": group_name,
        "dataset": dataset_rel,
        "features_shape": list(x.shape),
        "num_classes_raw": int(torch.unique(y).numel()),
        "raw_label_min": int(y.min().item()),
        "raw_label_max": int(y.max().item()),
        "folds": per_fold,
        "mean": {
            "window_acc": float(np.mean([r["window_acc"] for r in per_fold])),
            "window_macro_f1": float(np.mean([r["window_macro_f1"] for r in per_fold])),
            "window_bal_acc": float(np.mean([r["window_bal_acc"] for r in per_fold])),
            "coverage": float(np.mean([r["coverage"] for r in per_fold])),
        },
    }


def plot_grouped_metrics(results: List[Dict[str, object]], out_png: Path, out_svg: Path, dpi: int) -> None:
    fold_names = ["fold1", "fold2", "fold3"]
    acc = []
    f1 = []
    bal = []
    for fold_name in fold_names:
        fold_acc = []
        fold_f1 = []
        fold_bal = []
        for r in results:
            folds = r["folds"]
            hit = next((f for f in folds if f["fold"] == fold_name), None)
            if hit is None:
                raise KeyError(f"Missing {fold_name} in dataset result: {r['name']}")
            fold_acc.append(float(hit["window_acc"]) * 100.0)
            fold_f1.append(float(hit["window_macro_f1"]) * 100.0)
            fold_bal.append(float(hit["window_bal_acc"]) * 100.0)
        acc.append(float(np.mean(fold_acc)))
        f1.append(float(np.mean(fold_f1)))
        bal.append(float(np.mean(fold_bal)))

    x = np.arange(len(fold_names))
    width = 0.22
    colors = ["#9499df", "#e29693", "#8dbfa7"]

    fig, ax = plt.subplots(figsize=(3.2, 5.5))
    ax.bar(x - width, acc, width=width, color=colors[0], alpha=0.75, edgecolor="white", label="Accuracy")
    ax.bar(x, f1, width=width, color=colors[1], alpha=0.75, edgecolor="white", label="Macro-F1")
    ax.bar(x + width, bal, width=width, color=colors[2], alpha=0.75, edgecolor="white", label="Balanced Acc")

    ax.set_xticks(x, fold_names)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=8)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", direction="in")
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    fig.savefig(out_svg)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="General LOSO saved-model evaluation and grouped-metric plotting."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "data/.processed/concatenated_dataset-krushna-letters-group_1.pt",
            "data/.processed/concatenated_dataset-krushna-letters-group_2.pt",
            "data/.processed/concatenated_dataset-krushna-letters-group_3.pt",
        ],
        help="Dataset .pt files to evaluate.",
    )
    parser.add_argument(
        "--group-names",
        nargs="+",
        default=["Krushna G1", "Krushna G2", "Krushna G3"],
        help="Group labels displayed on the x-axis.",
    )
    parser.add_argument("--loso-dir", type=str, default="data/loso")
    parser.add_argument("--encoder-dir", type=str, default="checkpoints/phase2/simsiam")
    parser.add_argument("--phase3-dir", type=str, default="checkpoints/phase3")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-dir", type=str, default="outputs/generalization")
    parser.add_argument("--out-stem", type=str, default="loso_generalization_grouped_metrics")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.datasets) != len(args.group_names):
        raise ValueError("--datasets and --group-names must have the same length.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loso_dir = ROOT / args.loso_dir
    encoder_dir = ROOT / args.encoder_dir
    phase3_dir = ROOT / args.phase3_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for ds_rel, group_name in zip(args.datasets, args.group_names):
        row = evaluate_dataset(
            dataset_rel=ds_rel,
            group_name=group_name,
            loso_dir=loso_dir,
            encoder_dir=encoder_dir,
            phase3_dir=phase3_dir,
            batch_size=args.batch_size,
            device=device,
        )
        results.append(row)
        mean = row["mean"]
        print(
            f"{group_name}: acc={mean['window_acc']:.4f}, macro_f1={mean['window_macro_f1']:.4f}, "
            f"bal_acc={mean['window_bal_acc']:.4f}, coverage={mean['coverage']:.2%}"
        )

    png = out_dir / f"{args.out_stem}.png"
    svg = out_dir / f"{args.out_stem}.svg"
    out_json = out_dir / f"{args.out_stem}.json"
    plot_grouped_metrics(results, png, svg, args.dpi)

    payload = {
        "title": "LOSO saved-model generalization across external groups",
        "device": str(device),
        "datasets": results,
        "figure": str(png.relative_to(ROOT)),
        "figure_svg": str(svg.relative_to(ROOT)),
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {png}")
    print(f"Wrote {svg}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
