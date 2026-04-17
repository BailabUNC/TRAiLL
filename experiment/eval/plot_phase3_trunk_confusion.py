#!/usr/bin/env python3
"""Plot confusion matrices from ``loso_*_phase3_trunk_results.json`` (val + test diagnostics)."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, List, Literal

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


Split = Literal["test", "val"]


def _load_confusion_matrix(payload: dict[str, Any], split: Split) -> np.ndarray:
    sub = "test_diagnostics" if split == "test" else "val_diagnostics"
    try:
        cm = payload["phase3_trunk"][sub]["confusion_matrix"]
    except KeyError as e:
        hint = "Run: python experiment/run_loso_phase3_trunk.py (val_diagnostics added in recent runs)."
        raise KeyError(
            f"Expected keys phase3_trunk -> {sub} -> confusion_matrix. {hint}"
        ) from e
    arr = np.asarray(cm, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"confusion_matrix must be square 2D, got shape {arr.shape}")
    return arr


def _nnz_pred_columns(cm: np.ndarray) -> int:
    """How many predicted classes receive at least one sample (column sum > 0)."""
    return int((cm.sum(axis=0) > 0).sum())


def plot_confusion_matrix(
    cm: np.ndarray,
    *,
    title: str,
    out_path: Path,
    dpi: int,
) -> None:
    """Two panels: log1p(count) (avoids one-hot column washing out the rest) and row-normalized recall."""
    n = cm.shape[0]
    fig_w = max(10.0, min(22.0, 0.7 * n + 6))
    fig_h = max(5.0, min(12.0, 0.35 * n + 3))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(fig_w, fig_h))

    pred_cols = _nnz_pred_columns(cm)
    fig.suptitle(
        f"{title}\n"
        f"(non-zero predicted columns: {pred_cols}/{n} — vertical bands = most predictions fall in few classes)",
        fontsize=10,
    )

    log_cm = np.log1p(cm.astype(np.float64))
    im0 = ax0.imshow(log_cm, cmap="Blues", aspect="equal", interpolation="nearest")
    ax0.set_xlabel("Predicted")
    ax0.set_ylabel("True")
    ax0.set_title("log1p(count)")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    row_sum = cm.sum(axis=1, keepdims=True).astype(np.float64)
    row_norm = np.divide(
        cm.astype(np.float64), row_sum, out=np.zeros_like(cm, dtype=np.float64), where=row_sum > 0
    )
    im1 = ax1.imshow(row_norm, cmap="Blues", aspect="equal", interpolation="nearest", vmin=0, vmax=1)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.set_title("Row-normalized (recall per true class)")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ticks = list(range(0, n, 2)) if n > 20 else list(range(n))
    for ax in (ax0, ax1):
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def default_result_paths(repo_root: Path) -> List[Path]:
    base = repo_root / "outputs" / "phase3" / "arrays"
    if not base.is_dir():
        return []
    paths = sorted(base.glob("loso_fold*_phase3_trunk_results.json"))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot confusion matrix heatmaps from Phase-III trunk LOSO JSON outputs."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=str,
        default=[],
        help="Paths to loso_*_phase3_trunk_results.json (default: all under outputs/phase3/arrays/).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/phase3/plots",
        help="Directory for PNG files (created if missing).",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--split",
        choices=("both", "test", "val"),
        default="both",
        help="Which split(s) to plot (val requires val_diagnostics in JSON).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    if args.inputs:
        paths = [Path(p) if Path(p).is_absolute() else repo_root / p for p in args.inputs]
    else:
        paths = default_result_paths(repo_root)
        if not paths:
            raise SystemExit(
                f"No default JSON files found under {repo_root / 'outputs' / 'phase3' / 'arrays'}. "
                "Pass explicit paths: python experiment/eval/plot_phase3_trunk_confusion.py path/to/loso_fold1_....json"
            )

    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else repo_root / args.out_dir

    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        stem = path.stem
        m = re.match(r"(loso_fold\d+)_phase3_trunk_results", stem)
        tag = m.group(1) if m else stem

        splits: List[Split] = []
        if args.split in ("both", "test"):
            splits.append("test")
        if args.split in ("both", "val"):
            splits.append("val")

        for sp in splits:
            try:
                cm = _load_confusion_matrix(payload, sp)
            except KeyError as err:
                if sp == "val":
                    print(f"  skip {tag} val: {err}")
                    continue
                raise
            suffix = "" if sp == "test" else "_val"
            out_path = out_dir / f"{tag}_confusion_matrix{suffix}.png"
            title = f"{tag} — {sp} confusion ({cm.shape[0]}×{cm.shape[1]})\n{path.name}"
            plot_confusion_matrix(cm, title=title, out_path=out_path, dpi=args.dpi)
            print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
