from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from path_config import ROOT
from experiment.loso.loso_phase_utils import load_split, remap_labels


def _flatten(x_t_c: torch.Tensor) -> np.ndarray:
    return x_t_c.reshape(x_t_c.shape[0], -1).cpu().numpy()


def _zscore_fit_transform(x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = x_train.mean(axis=0, keepdims=True)
    sigma = x_train.std(axis=0, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return (x_train - mu) / sigma, (x_val - mu) / sigma, (x_test - mu) / sigma


class NaiveMLP(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        # Fixed random linear projection to mimic a frozen latent extractor.
        self.proj = nn.Linear(in_dim, latent_dim, bias=False)
        nn.init.kaiming_normal_(self.proj.weight, nonlinearity="linear")
        for p in self.proj.parameters():
            p.requires_grad_(False)

        self.head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.proj(x)
        return self.head(z)


ValSelectMetric = Literal["mean", "min_subj"]


def _val_accuracy_by_subject(
    model: NaiveMLP,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    val_subject_ids: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> Tuple[float, float]:
    model.eval()
    dl = DataLoader(
        TensorDataset(x_val, y_val, val_subject_ids),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    per_sid_correct: Dict[int, int] = {}
    per_sid_total: Dict[int, int] = {}
    with torch.no_grad():
        for xb, yb, sid in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb).argmax(1)
            ok = pred == yb
            for i in range(xb.size(0)):
                s = int(sid[i].item())
                per_sid_correct[s] = per_sid_correct.get(s, 0) + int(ok[i].item())
                per_sid_total[s] = per_sid_total.get(s, 0) + 1
    mean_acc = float(sum(per_sid_correct.values()) / max(sum(per_sid_total.values()), 1))
    if not per_sid_total:
        return mean_acc, mean_acc
    min_subj_acc = float(
        min(per_sid_correct[s] / max(per_sid_total[s], 1) for s in sorted(per_sid_total.keys()))
    )
    return mean_acc, min_subj_acc


def _parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _fold_metrics(
    inner_train: Dict[str, torch.Tensor],
    inner_val: Dict[str, torch.Tensor],
    outer_test: Dict[str, torch.Tensor],
    device: torch.device,
    latent_dim: int,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    patience: int,
    batch_size: int,
    lr_head: float,
    lr_enc: float,
    weight_decay: float,
    val_select_metric: ValSelectMetric,
    trainable_proj: bool,
) -> Dict[str, object]:
    tr_y, va_y, te_y = remap_labels(inner_train["labels"], inner_val["labels"], outer_test["labels"])
    x_tr = _flatten(inner_train["features"])
    x_va = _flatten(inner_val["features"])
    x_te = _flatten(outer_test["features"])

    x_tr, x_va, x_te = _zscore_fit_transform(x_tr, x_va, x_te)
    tr_x_t = torch.from_numpy(x_tr).float()
    va_x_t = torch.from_numpy(x_va).float()
    te_x_t = torch.from_numpy(x_te).float()
    tr_y_t = tr_y.long()
    va_y_t = va_y.long()
    te_y_t = te_y.long()

    in_dim = int(tr_x_t.shape[1])
    num_classes = int(tr_y_t.max().item() + 1)
    model = NaiveMLP(
        in_dim=in_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)
    for p in model.proj.parameters():
        p.requires_grad_(bool(trainable_proj))
    optim_groups = [{"params": list(model.head.parameters()), "lr": lr_head}]
    proj_trainable_params = [p for p in model.proj.parameters() if p.requires_grad]
    if proj_trainable_params:
        optim_groups.append({"params": proj_trainable_params, "lr": lr_enc})
    optimizer = torch.optim.AdamW(optim_groups, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = nn.CrossEntropyLoss()

    dl_tr = DataLoader(TensorDataset(tr_x_t, tr_y_t), batch_size=batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(TensorDataset(va_x_t, va_y_t), batch_size=batch_size, shuffle=False, drop_last=False)
    dl_te = DataLoader(TensorDataset(te_x_t, te_y_t), batch_size=batch_size, shuffle=False, drop_last=False)

    best_val = -1.0
    bad = 0
    best_state = None
    val_sid = inner_val["subject_ids"].long()
    n_val_subj = int(torch.unique(val_sid).numel())
    effective_val_metric: ValSelectMetric = val_select_metric
    if val_select_metric == "min_subj" and n_val_subj < 2:
        effective_val_metric = "mean"

    for _ep in range(1, epochs + 1):
        model.train()
        for xb, yb in dl_tr:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_mean, val_min_subj = _val_accuracy_by_subject(
            model=model,
            x_val=va_x_t,
            y_val=va_y_t,
            val_subject_ids=val_sid,
            device=device,
            batch_size=batch_size,
        )
        val_select = val_min_subj if effective_val_metric == "min_subj" else val_mean
        if val_select > best_val + 1e-6:
            best_val = val_select
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for xb, yb in dl_te:
            xb = xb.to(device)
            pred = model(xb).argmax(1).cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(yb.numpy().tolist())

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=np.arange(num_classes))
    per_class_f1 = f1_score(y_true_arr, y_pred_arr, labels=np.arange(num_classes), average=None, zero_division=0)
    return {
        "window_acc": float(accuracy_score(y_true_arr, y_pred_arr)),
        "window_macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "window_bal_acc": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "instance_acc": float(accuracy_score(y_true_arr, y_pred_arr)),
        "instance_macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "val_best_acc": float(best_val),
        "val_select_metric": effective_val_metric,
        "param_count": _parameter_count(model),
        "latent_dim": int(latent_dim),
        "projection": "frozen_random_linear",
        "trainable_projection": bool(trainable_proj),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LOSO naive MLP baseline (raw flattened input) with phase3-like training recipe."
    )
    parser.add_argument("--loso-dir", type=str, default="data/loso")
    parser.add_argument("--out-dir", type=str, default="outputs/loso/naive/results")
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--patience", type=int, default=220)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-head", type=float, default=5e-3)
    parser.add_argument("--lr-enc", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--val-select", choices=("mean", "min_subj"), default="min_subj")
    parser.add_argument(
        "--trainable-proj",
        action="store_true",
        help="If set, unfreeze projection layer and train it with --lr-enc.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loso_dir = ROOT / args.loso_dir
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "phase": "naive_mlp",
        "device": str(device),
        "folds": {},
        "hparams": {
            "latent_dim": args.latent_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "lr_head": args.lr_head,
            "lr_enc": args.lr_enc,
            "weight_decay": args.weight_decay,
            "val_select": args.val_select,
            "trainable_proj": bool(args.trainable_proj),
            "seed": args.seed,
            "input_representation": "flattened_raw_48x512_then_frozen_linear_projection",
            "normalization": "zscore_train_stats_per_fold",
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
            device=device,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr_head=args.lr_head,
            lr_enc=args.lr_enc,
            weight_decay=args.weight_decay,
            val_select_metric=args.val_select,
            trainable_proj=bool(args.trainable_proj),
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
    bals: List[float] = [summary["folds"][f]["naive_mlp"]["window_bal_acc"] for f in ("fold1", "fold2", "fold3")]
    summary["aggregate"] = {
        "window_acc_mean": float(np.mean(accs)),
        "window_acc_std": float(np.std(accs)),
        "window_macro_f1_mean": float(np.mean(f1s)),
        "window_macro_f1_std": float(np.std(f1s)),
        "window_bal_acc_mean": float(np.mean(bals)),
        "window_bal_acc_std": float(np.std(bals)),
    }
    with (out_dir / "loso_naive_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote naive LOSO summary to {out_dir / 'loso_naive_summary.json'}")


if __name__ == "__main__":
    main()
