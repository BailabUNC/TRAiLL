#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

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


def _load_fold_cm(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    cm = np.asarray(payload["phase3_trunk"]["test_diagnostics"]["confusion_matrix"], dtype=np.float64)
    if cm.shape != (24, 24):
        raise ValueError(f"Expected 24x24 confusion matrix, got {cm.shape} from {path}")
    return cm


def _row_normalize(cm: np.ndarray) -> np.ndarray:
    row_sum = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=np.float64), where=row_sum > 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-subject confusion structure under LOSO (SSL pooled).")
    parser.add_argument(
        "--inputs",
        nargs="*",
        type=str,
        default=[],
        help="Optional list of loso_fold*_phase3_trunk_results.json files. Defaults to outputs/loso/phase3/results.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/loso/phase3/plots",
        help="Directory to write pooled confusion figure and JSON.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    if args.inputs:
        files: List[Path] = [Path(p) if Path(p).is_absolute() else ROOT / p for p in args.inputs]
    else:
        base = ROOT / "outputs" / "loso" / "phase3" / "results"
        files = [base / f"loso_fold{i}_phase3_trunk_results.json" for i in (1, 2, 3)]

    for p in files:
        if not p.exists():
            raise FileNotFoundError(f"Missing fold file: {p}")

    cms = [_load_fold_cm(p) for p in files]
    pooled_cm = np.sum(cms, axis=0)
    pooled_norm = _row_normalize(pooled_cm)

    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Match compact square look used by fig4-signlang/confusion_matrix_best_fold.svg
    fig, ax = plt.subplots(figsize=(2.46, 2.46))
    im = ax.imshow(pooled_norm, cmap="Blues", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="equal")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    tick_pos = [0, 5, 10, 15, 20, 23]
    tick_lbl = [str(i + 1) for i in tick_pos]
    ax.set_xticks(tick_pos, tick_lbl)
    ax.set_yticks(tick_pos, tick_lbl)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", direction="in")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized proportion")
    fig.tight_layout()

    png_path = out_dir / "loso_pooled_confusion_row_normalized.png"
    svg_path = out_dir / "loso_pooled_confusion_row_normalized.svg"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight", transparent=True)
    fig.savefig(svg_path, bbox_inches="tight", transparent=True)
    plt.close(fig)

    out_json = out_dir / "loso_pooled_confusion_row_normalized.json"
    payload = {
        "title_meaning": "Cross-subject confusion structure under LOSO",
        "source_folds": [str(p.relative_to(ROOT)) for p in files],
        "shape": [24, 24],
        "pooled_confusion_matrix": pooled_cm.tolist(),
        "pooled_row_normalized_confusion": pooled_norm.tolist(),
        "figure": str(png_path.relative_to(ROOT)),
        "figure_svg": str(svg_path.relative_to(ROOT)),
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {png_path}")
    print(f"Wrote {svg_path}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
