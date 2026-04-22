#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from path_config import ROOT

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams["font.size"] = 8


def _load_ssl(ssl_summary: Path) -> Dict[int, Dict[str, float]]:
    with ssl_summary.open("r", encoding="utf-8") as f:
        data = json.load(f)
    by_subj: Dict[int, Dict[str, float]] = {}
    for fold in ("fold1", "fold2", "fold3"):
        row = data["folds"][fold]
        subj = int(row["outer_test_subject_ids"][0])
        m = row["phase3_trunk"]
        by_subj[subj] = {
            "accuracy": float(m["window_acc"]),
            "macro_f1": float(m["window_macro_f1"]),
        }
    return by_subj


def _load_naive(naive_summary: Path) -> Dict[int, Dict[str, float]]:
    with naive_summary.open("r", encoding="utf-8") as f:
        data = json.load(f)
    by_subj: Dict[int, Dict[str, float]] = {}
    for fold in ("fold1", "fold2", "fold3"):
        row = data["folds"][fold]
        subj = int(row["outer_test_subject_ids"][0])
        m = row["naive_mlp"]
        by_subj[subj] = {
            "accuracy": float(m["window_acc"]),
            "macro_f1": float(m["window_macro_f1"]),
        }
    return by_subj


def main() -> None:
    parser = argparse.ArgumentParser(description="Subject-wise held-out performance under 3-fold LOSO.")
    parser.add_argument(
        "--ssl-summary",
        type=str,
        default="outputs/loso/phase3/results/loso_phase3_trunk_summary.json",
        help="Phase3 SSL trunk summary JSON.",
    )
    parser.add_argument(
        "--naive-summary",
        type=str,
        default="outputs/loso/naive/results/loso_naive_summary.json",
        help="Naive MLP summary JSON (generate via experiment/naive/run_loso_naive_mlp.py).",
    )
    parser.add_argument("--out-dir", type=str, default="outputs/loso/per_subject")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()

    ssl_path = ROOT / args.ssl_summary
    naive_path = ROOT / args.naive_summary
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not naive_path.exists():
        raise FileNotFoundError(
            f"Missing naive summary: {naive_path}. "
            "Run: python experiment/naive/run_loso_naive_mlp.py"
        )

    ssl = _load_ssl(ssl_path)
    naive = _load_naive(naive_path)
    subjects = [1, 2, 3]

    ssl_vals = [ssl[s]["accuracy"] for s in subjects]
    naive_vals = [naive[s]["accuracy"] for s in subjects]
    categories = ["Subject 1", "Subject 2", "Subject 3"]
    naive_plot_pct = [v * 100.0 for v in naive_vals]
    ssl_plot_pct = [v * 100.0 for v in ssl_vals]
    x = np.arange(len(categories))
    width = 0.24
    colors = ["#9499df", "#e29693"]
    fig, ax = plt.subplots(figsize=(2.55, 5.5))
    bars_naive = ax.bar(
        x - width / 2,
        naive_plot_pct,
        width=width,
        label="Naive MLP",
        color=colors[0],
        alpha=0.7,
        edgecolor="white",
    )
    bars_ssl = ax.bar(
        x + width / 2,
        ssl_plot_pct,
        width=width,
        label="SSL pipeline",
        color=colors[1],
        alpha=0.7,
        edgecolor="white",
    )

    ax.set_xticks(x, categories)
    ax.set_ylim(40, 100)
    ax.set_ylabel("Classification accuracy (%)")
    ax.set_yticks(np.arange(40, 101, 20))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=8)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", direction="in")
    fig.tight_layout()

    png = out_dir / "loso_per_subject_accuracy.png"
    fig.savefig(png, dpi=args.dpi)
    plt.close(fig)

    payload = {
        "metric": "accuracy_percent",
        "subjects": subjects,
        "categories": categories,
        "naive_mlp": {
            "per_subject": naive_vals,
            "per_subject_percent": [v * 100.0 for v in naive_vals],
        },
        "ssl_pipeline": {
            "per_subject": ssl_vals,
            "per_subject_percent": [v * 100.0 for v in ssl_vals],
        },
        "title": "Subject-wise held-out performance under 3-fold LOSO",
        "figure": str(png.relative_to(ROOT)),
    }
    out_json = out_dir / "loso_per_subject_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {png}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
