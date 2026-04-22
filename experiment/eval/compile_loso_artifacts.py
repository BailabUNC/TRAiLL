#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from path_config import ROOT
from train_phase1_autoencoder import CNNAutoencoderStrided512
from train_phase2_simsiam import (
    CNNAutoencoderStrided512 as Phase2AE,
    Phase1Encoder,
    PredictionHead,
    ProjectionHead,
    augment_1d,
    simsiam_loss,
)


def _load_split(path: Path) -> Dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    out = {
        "features": data["features"].float(),  # [N,T,C]
        "labels": data["labels"].long(),
        "subject_ids": data["subject_ids"].long(),
    }
    if "instance_ids" in data:
        out["instance_ids"] = data["instance_ids"].long()
    return out


def _to_cf(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 1).contiguous()


def _batched_recon_loss(
    model: nn.Module, x_tc: torch.Tensor, criterion: nn.Module, device: torch.device, batch_size: int = 128
) -> float:
    model.eval()
    x = _to_cf(x_tc)
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for i in range(0, x.size(0), batch_size):
            xb = x[i: i + batch_size].to(device)
            xr = model(xb)
            loss = criterion(xr, xb)
            total_loss += float(loss.item()) * xb.size(0)
            total_n += xb.size(0)
    return total_loss / max(total_n, 1)


def _phase1_results(loso_dir: Path, phase1_plot_dir: Path, phase1_results_dir: Path, device: torch.device) -> None:
    criterion = nn.SmoothL1Loss()
    out: Dict[str, object] = {"phase": "phase1_diagnostics", "folds": {}}
    example_folds = ("fold1", "fold2")

    for fold in ("fold1", "fold2", "fold3"):
        train = _load_split(loso_dir / f"{fold}_inner_train.pt")
        val = _load_split(loso_dir / f"{fold}_inner_val.pt")
        test = _load_split(loso_dir / f"{fold}_outer_test.pt")

        ckpt = ROOT / "checkpoints" / "phase1" / f"loso_{fold}_autoencoder.pth"
        model = CNNAutoencoderStrided512(in_channels=48, latent_channels=32).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))

        train_loss = _batched_recon_loss(model, train["features"], criterion, device)
        val_loss = _batched_recon_loss(model, val["features"], criterion, device)
        heldout_loss = _batched_recon_loss(model, test["features"], criterion, device)

        fold_out = {
            "inner_train_reconstruction_loss": train_loss,
            "inner_val_reconstruction_loss": val_loss,
            "held_out_subject_reconstruction_loss": heldout_loss,
            "held_out_subject_ids": sorted(set(test["subject_ids"].tolist())),
        }

        if fold in example_folds:
            x = _to_cf(test["features"][0:1]).to(device)
            with torch.no_grad():
                xr = model(x).cpu().numpy()[0]  # [C,T]
            x_np = x.cpu().numpy()[0]

            # 3D overlay for all 48 channels:
            # x-axis = channel index, y-axis = time index, z-axis = signal value.
            c, t = x_np.shape
            ch_idx = np.arange(c)
            time_idx = np.arange(t)
            x_grid, z_grid = np.meshgrid(ch_idx, time_idx, indexing="ij")
            raw_vs_recon_path = phase1_plot_dir / f"{fold}_heldout_raw_vs_recon_3d.png"
            fig = plt.figure(figsize=(11, 6))
            ax = fig.add_subplot(111, projection="3d")
            recon_baseline = 1.0
            xr_shifted = xr + recon_baseline
            ax.plot_surface(
                x_grid,
                z_grid,
                x_np,
                cmap="Blues",
                alpha=0.65,
                linewidth=0,
                antialiased=False,
            )
            ax.plot_surface(
                x_grid,
                z_grid,
                xr_shifted,
                cmap="Oranges",
                alpha=0.45,
                linewidth=0,
                antialiased=False,
            )
            ax.set_title(
                f"{fold} held-out subject: raw (blue) vs recon+{recon_baseline:.1f} (orange)"
            )
            ax.set_xlabel("channel")
            ax.set_ylabel("time")
            ax.set_zlabel("value")
            fig.tight_layout()
            fig.savefig(raw_vs_recon_path, dpi=150)
            plt.close(fig)

            raw_mean = x_np.mean(axis=0)
            rec_mean = xr.mean(axis=0)
            raw_fft = np.abs(np.fft.rfft(raw_mean))
            rec_fft = np.abs(np.fft.rfft(rec_mean))
            n_bins = min(120, raw_fft.shape[0])
            spectrum_path = phase1_plot_dir / f"{fold}_heldout_lowfreq_spectrum_compare.png"
            plt.figure(figsize=(8, 3))
            plt.plot(raw_fft[:n_bins], label="raw mean-spectrum", linewidth=1.0)
            plt.plot(rec_fft[:n_bins], label="recon mean-spectrum", linewidth=1.0, alpha=0.9)
            plt.title(f"{fold} held-out low-frequency spectrum (first {n_bins} bins)")
            plt.xlabel("fft bin")
            plt.ylabel("magnitude")
            plt.legend()
            plt.tight_layout()
            plt.savefig(spectrum_path, dpi=150)
            plt.close()

            fold_out["example_plots"] = {
                "raw_vs_recon_3d": str(raw_vs_recon_path.relative_to(ROOT)),
                "lowfreq_spectrum_compare": str(spectrum_path.relative_to(ROOT)),
            }

        out["folds"][fold] = fold_out

    with (phase1_results_dir / "loso_phase1_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def _phase2_load_models(
    fold: str, device: torch.device
) -> Tuple[nn.Module, nn.Module, nn.Module]:
    p1_ckpt = ROOT / "checkpoints" / "phase1" / f"loso_{fold}_autoencoder.pth"
    p2_proj = ROOT / "checkpoints" / "phase2" / "simsiam" / f"loso_{fold}_proj_head.pth"
    p2_pred = ROOT / "checkpoints" / "phase2" / "simsiam" / f"loso_{fold}_pred_head.pth"
    p2_enc = ROOT / "checkpoints" / "phase2" / "simsiam" / f"loso_{fold}_encoder.pth"

    ae = Phase2AE(in_channels=48, latent_channels=64).to(device)
    ckpt = torch.load(p1_ckpt, map_location=device)
    filtered = {k: v for k, v in ckpt.items() if not k.startswith("enc5")}
    ae.load_state_dict(filtered, strict=False)
    encoder = Phase1Encoder(ae).to(device)
    encoder.load_state_dict(torch.load(p2_enc, map_location=device))

    proj = ProjectionHead(in_dim=64, hidden_dim=128, out_dim=64).to(device)
    proj.load_state_dict(torch.load(p2_proj, map_location=device))

    pred = PredictionHead(in_dim=64, hidden_dim=128, out_dim=64).to(device)
    pred.load_state_dict(torch.load(p2_pred, map_location=device))

    encoder.eval()
    proj.eval()
    pred.eval()
    return encoder, proj, pred


def _embed(encoder: nn.Module, proj: nn.Module, x_cf: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        h = encoder(x_cf.to(device)).mean(dim=2)
        z = proj(h)
        z = nn.functional.normalize(z, dim=1)
    return z.cpu()


def _random_pair_mean_cos(z: torch.Tensor, labels: torch.Tensor, same_class: bool, n_pairs: int = 1024) -> float:
    rng = random.Random(7)
    y = labels.numpy()
    idxs = list(range(z.size(0)))
    cos_vals: List[float] = []

    for _ in range(n_pairs * 8):
        i = rng.choice(idxs)
        j = rng.choice(idxs)
        if i == j:
            continue
        cond = y[i] == y[j]
        if cond != same_class:
            continue
        cos_vals.append(float(torch.dot(z[i], z[j]).item()))
        if len(cos_vals) >= n_pairs:
            break
    return float(np.mean(cos_vals)) if cos_vals else float("nan")


def _simsiam_dataset_loss(
    encoder: nn.Module,
    proj: nn.Module,
    pred: nn.Module,
    x_cf: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> float:
    total_loss = 0.0
    total_n = 0
    with torch.no_grad():
        for i in range(0, x_cf.size(0), batch_size):
            xb = x_cf[i: i + batch_size]
            x1 = torch.stack([augment_1d(xx) for xx in xb]).to(device)
            x2 = torch.stack([augment_1d(xx) for xx in xb]).to(device)
            h1 = encoder(x1).mean(dim=2)
            h2 = encoder(x2).mean(dim=2)
            z1 = proj(h1)
            z2 = proj(h2)
            p1 = pred(z1)
            p2 = pred(z2)
            loss = simsiam_loss(p1, z2, p2, z1)
            total_loss += float(loss.item()) * xb.size(0)
            total_n += xb.size(0)
    return total_loss / max(total_n, 1)


def _phase2_results(loso_dir: Path, phase2_plot_dir: Path, phase2_results_dir: Path, device: torch.device) -> None:
    out: Dict[str, object] = {"phase": "phase2_diagnostics", "folds": {}}

    for fold in ("fold1", "fold2", "fold3"):
        train = _load_split(loso_dir / f"{fold}_inner_train.pt")
        val = _load_split(loso_dir / f"{fold}_inner_val.pt")
        x_train_cf = _to_cf(train["features"])
        x_val_cf = _to_cf(val["features"])
        y_val = val["labels"]

        encoder, proj, pred = _phase2_load_models(fold, device)
        train_loss = _simsiam_dataset_loss(encoder, proj, pred, x_train_cf, device)
        val_loss = _simsiam_dataset_loss(encoder, proj, pred, x_val_cf, device)

        z_clean = _embed(encoder, proj, x_val_cf, device)
        aug1 = torch.stack([augment_1d(xx) for xx in x_val_cf])
        aug2 = torch.stack([augment_1d(xx) for xx in x_val_cf])
        z1 = _embed(encoder, proj, aug1, device)
        z2 = _embed(encoder, proj, aug2, device)
        same_inst_aug_cos = float((z1 * z2).sum(dim=1).mean().item())
        same_cls_cos = _random_pair_mean_cos(z_clean, y_val, same_class=True, n_pairs=1024)
        diff_cls_cos = _random_pair_mean_cos(z_clean, y_val, same_class=False, n_pairs=1024)

        out["folds"][fold] = {
            "self_supervised_train_loss": train_loss,
            "self_supervised_val_loss": val_loss,
            "same_instance_augmented_pair_cosine_mean": same_inst_aug_cos,
            "different_instance_same_class_cosine_mean": same_cls_cos,
            "different_class_cosine_mean": diff_cls_cos,
        }

    fold1 = _load_split(loso_dir / "fold1_inner_val.pt")
    encoder, proj, _pred = _phase2_load_models("fold1", device)
    z = _embed(encoder, proj, _to_cf(fold1["features"]), device).numpy()
    y = fold1["labels"].numpy()
    pca = PCA(n_components=2, random_state=7)
    pts = pca.fit_transform(z)
    emb_path = phase2_plot_dir / "fold1_inner_val_embedding_pca.png"
    plt.figure(figsize=(7, 6))
    plt.scatter(pts[:, 0], pts[:, 1], c=y, s=12, cmap="tab20", alpha=0.8)
    plt.title("Phase II embedding visualization (fold1 inner-val, PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(emb_path, dpi=180)
    plt.close()
    out["embedding_plot"] = str(emb_path.relative_to(ROOT))

    with (phase2_results_dir / "loso_phase2_diagnostics.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def _macro_f1_from_cm(cm: np.ndarray) -> float:
    row = cm.sum(axis=1)
    col = cm.sum(axis=0)
    diag = np.diag(cm)
    precision = np.divide(diag, col, out=np.zeros_like(diag, dtype=np.float64), where=col > 0)
    recall = np.divide(diag, row, out=np.zeros_like(diag, dtype=np.float64), where=row > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(recall), where=(precision + recall) > 0)
    return float(np.mean(f1))


def _phase3_pooled_results(phase3_results_dir: Path) -> Dict[str, object]:
    payloads = []
    for fold in ("fold1", "fold2", "fold3"):
        p = phase3_results_dir / f"loso_{fold}_phase3_trunk_results.json"
        with p.open("r", encoding="utf-8") as f:
            payloads.append((fold, json.load(f)))

    fold_metrics: Dict[str, object] = {}
    cms = []
    for fold, data in payloads:
        td = data["phase3_trunk"]["test_diagnostics"]
        fold_metrics[fold] = {
            "accuracy": td["window_acc"],
            "macro_f1": td["window_macro_f1"],
            "balanced_accuracy": td["window_bal_acc"],
            "per_class_recall": td["per_class_recall"],
            "held_out_subject_ids": data["outer_test_subject_ids"],
        }
        cms.append(np.asarray(td["confusion_matrix"], dtype=np.int64))

    pooled_cm = np.sum(cms, axis=0)
    total = pooled_cm.sum()
    acc = float(np.trace(pooled_cm) / max(total, 1))
    row = pooled_cm.sum(axis=1)
    recall = np.divide(np.diag(pooled_cm), row, out=np.zeros_like(row, dtype=np.float64), where=row > 0)
    bal_acc = float(np.mean(recall))
    macro_f1 = _macro_f1_from_cm(pooled_cm)
    return {
        "fold_metrics": fold_metrics,
        "pooled": {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "balanced_accuracy": bal_acc,
            "per_class_recall": recall.tolist(),
            "confusion_matrix": pooled_cm.tolist(),
        },
    }


def _write_pooled_plot(phase3_plot_dir: Path, pooled_cm: np.ndarray) -> None:
    pooled_path = phase3_plot_dir / "pooled_confusion_matrix_outer_test_held_out.png"
    plt.figure(figsize=(7, 6))
    plt.imshow(np.log1p(pooled_cm), cmap="Blues", aspect="equal")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("Pooled held-out confusion matrix (log1p counts)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(pooled_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute missing LOSO diagnostics into outputs/loso/phase*/results JSON files.")
    parser.add_argument("--loso-dir", type=str, default="data/loso")
    parser.add_argument(
        "--phase3-results-dir",
        type=str,
        default="outputs/loso/phase3/results",
        help="Directory containing loso_fold*_phase3_trunk_results.json used for pooled metrics.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/loso",
        help="Root path containing phase1/phase2/phase3 subfolders where LOSO diagnostics and plots are written.",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loso_dir = ROOT / args.loso_dir

    out_root = Path(args.output_root)
    out_root = out_root if out_root.is_absolute() else ROOT / out_root
    phase1_results_dir = out_root / "phase1" / "results"
    phase2_results_dir = out_root / "phase2" / "results"
    phase3_results_dir = out_root / "phase3" / "results"
    phase1_plot_dir = out_root / "phase1" / "plots"
    phase2_plot_dir = out_root / "phase2" / "plots"
    phase3_plot_dir = out_root / "phase3" / "plots"
    phase3_source_results_dir = Path(args.phase3_results_dir)
    phase3_source_results_dir = (
        phase3_source_results_dir
        if phase3_source_results_dir.is_absolute()
        else ROOT / phase3_source_results_dir
    )
    for p in (
        phase1_results_dir,
        phase2_results_dir,
        phase3_results_dir,
        phase1_plot_dir,
        phase2_plot_dir,
        phase3_plot_dir,
    ):
        p.mkdir(parents=True, exist_ok=True)

    _phase1_results(loso_dir, phase1_plot_dir, phase1_results_dir, device)
    _phase2_results(loso_dir, phase2_plot_dir, phase2_results_dir, device)
    phase3 = _phase3_pooled_results(phase3_source_results_dir)

    pooled_cm = np.asarray(phase3["pooled"]["confusion_matrix"], dtype=np.int64)
    _write_pooled_plot(phase3_plot_dir, pooled_cm)
    with (phase3_results_dir / "loso_phase3_pooled_confusion.json").open("w", encoding="utf-8") as f:
        json.dump(phase3, f, indent=2)

    print(f"Wrote {phase1_results_dir / 'loso_phase1_diagnostics.json'}")
    print(f"Wrote {phase2_results_dir / 'loso_phase2_diagnostics.json'}")
    print(f"Wrote {phase3_results_dir / 'loso_phase3_pooled_confusion.json'}")


if __name__ == "__main__":
    main()
