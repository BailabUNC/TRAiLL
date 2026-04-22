#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib
import matplotlib as mpl
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from path_config import ROOT

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"] = 8


def _load_ssl_cm(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    cm = np.asarray(payload["phase3_trunk"]["test_diagnostics"]["confusion_matrix"], dtype=np.float64)
    if cm.shape != (24, 24):
        raise ValueError(f"Expected 24x24 SSL confusion matrix, got {cm.shape} from {path}")
    return cm


def _load_naive_cm(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    cm = np.asarray(payload["naive_mlp"]["confusion_matrix"], dtype=np.float64)
    if cm.shape != (24, 24):
        raise ValueError(f"Expected 24x24 naive confusion matrix, got {cm.shape} from {path}")
    return cm


def _per_class_f1_from_cm(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2.0 * tp + fp + fn
    return np.divide(2.0 * tp, denom, out=np.zeros_like(tp, dtype=np.float64), where=denom > 0)


def _resolve_inputs(paths: Sequence[str]) -> List[Path]:
    return [Path(p) if Path(p).is_absolute() else ROOT / p for p in paths]


def _sorted_view(
    ssl_f1: np.ndarray,
    naive_f1: np.ndarray,
    delta_f1: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.arange(len(delta_f1))
    if mode == "descending":
        order = np.argsort(-delta_f1)
    return order, ssl_f1[order], naive_f1[order], delta_f1[order]


def _gesture_alphabet_labels() -> List[str]:
    # TRAiLL gestures map 1..24 -> A..Z excluding J and Z.
    return [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in {"J", "Z"}]


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-gesture benefit of SSL under LOSO (pooled across 3 folds).")
    parser.add_argument(
        "--ssl-inputs",
        nargs="*",
        type=str,
        default=[],
        help="Optional SSL fold result JSONs. Defaults to outputs/loso/phase3/results/loso_fold*_phase3_trunk_results.json",
    )
    parser.add_argument(
        "--naive-inputs",
        nargs="*",
        type=str,
        default=[],
        help="Optional naive fold result JSONs. Defaults to outputs/loso/naive/results/loso_fold*_naive_mlp_results.json",
    )
    parser.add_argument(
        "--sort",
        type=str,
        default="descending",
        choices=("descending", "canonical"),
        help="Bar order for gestures.",
    )
    parser.add_argument("--out-dir", type=str, default="outputs/loso/phase3/plots")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    if args.ssl_inputs:
        ssl_files = _resolve_inputs(args.ssl_inputs)
    else:
        base = ROOT / "outputs" / "loso" / "phase3" / "results"
        ssl_files = [base / f"loso_fold{i}_phase3_trunk_results.json" for i in (1, 2, 3)]
    if args.naive_inputs:
        naive_files = _resolve_inputs(args.naive_inputs)
    else:
        base = ROOT / "outputs" / "loso" / "naive" / "results"
        naive_files = [base / f"loso_fold{i}_naive_mlp_results.json" for i in (1, 2, 3)]

    for p in ssl_files + naive_files:
        if not p.exists():
            raise FileNotFoundError(f"Missing fold file: {p}")

    pooled_ssl_cm = np.sum([_load_ssl_cm(p) for p in ssl_files], axis=0)
    pooled_naive_cm = np.sum([_load_naive_cm(p) for p in naive_files], axis=0)

    ssl_f1 = _per_class_f1_from_cm(pooled_ssl_cm)
    naive_f1 = _per_class_f1_from_cm(pooled_naive_cm)
    delta_f1 = ssl_f1 - naive_f1

    order, ssl_sorted, naive_sorted, delta_sorted = _sorted_view(ssl_f1, naive_f1, delta_f1, args.sort)
    alphabet = _gesture_alphabet_labels()
    class_ids = (order + 1).tolist()
    labels = [alphabet[i] for i in order]

    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep figure size aligned with compact confusion-matrix style.
    fig, ax = plt.subplots(figsize=(2.46, 2.46))
    x = np.arange(24)
    colors = np.where(delta_sorted >= 0.0, "#4C78A8", "#E15759")
    ax.bar(x, delta_sorted, color=colors, width=0.72, edgecolor="none")
    ax.axhline(0.0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", direction="in", labelsize=7)
    ax.set_ylabel("ΔF1 score")
    ax.set_xlabel("ASL gesture")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_min = float(np.min(delta_sorted))
    y_max = float(np.max(delta_sorted))
    y_pad = max(0.006, 0.022 * (y_max - y_min if y_max > y_min else 1.0))
    for xi, yi, letter in zip(x, delta_sorted, labels):
        if yi >= 0.0:
            ax.text(xi, yi + y_pad, letter, ha="center", va="bottom", fontsize=5)
        else:
            label_y = yi - y_pad
            label_y = min(label_y, -0.012)
            ax.text(xi, label_y, letter, ha="center", va="top", fontsize=5)

    ax.set_ylim(y_min - 3.2 * y_pad, max(0.3, y_max + 1.2 * y_pad))
    ax.set_yticks([0.0, 0.1, 0.2, 0.3])
    fig.tight_layout()

    png_path = out_dir / "loso_per_gesture_delta_f1.png"
    svg_path = out_dir / "loso_per_gesture_delta_f1.svg"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", transparent=True)
    fig.savefig(svg_path, bbox_inches="tight", transparent=True)
    plt.close(fig)

    json_path = out_dir / "loso_per_gesture_delta_f1.json"
    payload = {
        "title_meaning": "Per-gesture benefit of SSL under LOSO",
        "metric": "per_class_f1_from_pooled_confusion",
        "sort_mode": args.sort,
        "class_order_1_indexed": class_ids,
        "class_labels": labels,
        "ssl_per_class_f1": ssl_sorted.tolist(),
        "naive_per_class_f1": naive_sorted.tolist(),
        "delta_f1": delta_sorted.tolist(),
        "source_ssl_folds": [str(p.relative_to(ROOT)) for p in ssl_files],
        "source_naive_folds": [str(p.relative_to(ROOT)) for p in naive_files],
        "figure": str(png_path.relative_to(ROOT)),
        "figure_svg": str(svg_path.relative_to(ROOT)),
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
