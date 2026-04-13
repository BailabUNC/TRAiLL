"""Shared Phase-III end-task pieces: numpy baseline windows, 1D-CNN encoder, fold training loops.

Used by ``train_phase3_end_task.py`` (manuscript-style) and ``experiments/phase3_end_task_*.py`` variants.
Encoder architecture and tensor layout must stay aligned with ``eval/phase3/confusion_and_roc.py``.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset


class NpyTSDataset(Dataset):
    """``features``: (N, L, C) float32; ``labels``: (N,) int — remapped to 0..K-1."""

    def __init__(self, feat_npy: str, label_npy: str):
        X = np.load(feat_npy).astype(np.float32)
        y = np.load(label_npy).astype(np.int64)
        uniq = np.unique(y)
        self.old2new = {old: i for i, old in enumerate(uniq)}
        y = np.vectorize(self.old2new.get)(y)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx].permute(1, 0)
        return x, self.y[idx]


class Encoder1DCNN(nn.Module):
    def __init__(self, C: int, embed_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(C, 128, 7, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, embed_dim, 3, padding=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_classifier(num_classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, num_classes),
    )


def run_fold_manuscript(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    fold_id: int,
    ds: NpyTSDataset,
    enc_ckpt: Dict[str, torch.Tensor],
    device: torch.device | str,
    *,
    max_epochs: int = 30,
    patience: int = 5,
    unfreeze_module_indices: Tuple[int, ...] = (4, 6),
    head_dropout: float = 0.3,
    lr_head: float = 1e-3,
    lr_enc_ft: float = 1e-4,
    weight_decay: float = 3e-4,
) -> Tuple[float, Dict[str, Any]]:
    """Freeze encoder, unfreeze selected ``Sequential`` indices, train MLP head (matches shipped ``end_task_*`` bundle)."""
    dl_train = DataLoader(Subset(ds, train_idx), batch_size=32, shuffle=True, drop_last=True)
    dl_val = DataLoader(Subset(ds, val_idx), batch_size=128, shuffle=False)

    enc = Encoder1DCNN(C=48, embed_dim=256).to(device)
    enc.load_state_dict(enc_ckpt)
    for p in enc.parameters():
        p.requires_grad_(False)
    for idx in unfreeze_module_indices:
        for p in enc.net[idx].parameters():
            p.requires_grad_(True)

    num_classes = len(ds.old2new)
    clf = _build_classifier(num_classes, head_dropout).to(device)

    head_params = list(clf.parameters())
    enc_params_ft = [p for p in enc.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": lr_head},
            {"params": enc_params_ft, "lr": lr_enc_ft},
        ],
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_state, bad_epochs = 0.0, None, 0

    for ep in range(1, max_epochs + 1):
        enc.train()
        clf.train()
        for x, y in dl_train:
            z = enc(x.to(device))
            loss = criterion(clf(z), y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        enc.eval()
        clf.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in dl_val:
                preds = clf(enc(x.to(device))).argmax(1).cpu()
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / max(total, 1)

        print(
            f"[fold {fold_id} | ep {ep:02d}] val_acc={acc:.3%} "
            f"lr_head={optimizer.param_groups[0]['lr']:.4f}"
        )

        if acc > best_acc + 1e-4:
            best_acc = acc
            best_state = {"enc": copy.deepcopy(enc.state_dict()), "clf": copy.deepcopy(clf.state_dict())}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stop at epoch {ep}")
                break

    assert best_state is not None
    return best_acc, best_state


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def run_fold_mixup_progressive_unfreeze(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    fold_id: int,
    ds: NpyTSDataset,
    enc_ckpt: Dict[str, torch.Tensor],
    device: torch.device | str,
    *,
    max_epochs: int = 50,
    patience: int = 10,
    unfreeze_module_indices: Tuple[int, ...] = (4, 5),
    head_dropout: float = 0.5,
    mixup_alpha: float = 0.4,
    progressive_unfreeze_epoch: int = 5,
    lr_head: float = 1e-3,
    lr_enc_ft: float = 1e-4,
    lr_enc_rest: float = 5e-5,
    weight_decay: float = 3e-4,
) -> Tuple[float, Dict[str, Any]]:
    """Mixup + cosine schedule; partial unfreeze then full encoder after ``progressive_unfreeze_epoch`` (archive ``phase1_2_cnn``)."""
    dl_train = DataLoader(Subset(ds, train_idx), batch_size=32, shuffle=True, drop_last=True)
    dl_val = DataLoader(Subset(ds, val_idx), batch_size=128, shuffle=False)

    enc = Encoder1DCNN(C=48, embed_dim=256).to(device)
    enc.load_state_dict(enc_ckpt)
    for p in enc.parameters():
        p.requires_grad_(False)
    for idx in unfreeze_module_indices:
        for p in enc.net[idx].parameters():
            p.requires_grad_(True)

    num_classes = len(ds.old2new)
    clf = _build_classifier(num_classes, head_dropout).to(device)

    head_params = list(clf.parameters())
    enc_params_ft = [p for p in enc.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": lr_head},
            {"params": enc_params_ft, "lr": lr_enc_ft},
        ],
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc, best_state, bad_epochs = 0.0, None, 0

    for ep in range(1, max_epochs + 1):
        if ep == progressive_unfreeze_epoch:
            newly_unfrozen = []
            for p in enc.parameters():
                if not p.requires_grad:
                    p.requires_grad_(True)
                    newly_unfrozen.append(p)
            optimizer.add_param_group({"params": newly_unfrozen, "lr": lr_enc_rest})
            print(f"[fold {fold_id}] unfreeze remaining encoder params: {len(newly_unfrozen)}")

        enc.train()
        clf.train()
        for x, y in dl_train:
            x, y_a, y_b, lam = mixup_data(x.to(device), y.to(device), alpha=mixup_alpha)
            out = clf(enc(x))
            loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        enc.eval()
        clf.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in dl_val:
                preds = clf(enc(x.to(device))).argmax(1).cpu()
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / max(total, 1)

        print(
            f"[fold {fold_id:>2} | ep {ep:>2}] val_acc={acc:6.2%} "
            f"lr_head={optimizer.param_groups[0]['lr']:.4f}"
        )

        if acc > best_acc + 1e-4:
            best_acc = acc
            best_state = {"enc": copy.deepcopy(enc.state_dict()), "clf": copy.deepcopy(clf.state_dict())}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stop at epoch {ep} (best={best_acc:6.2%})")
                break

    assert best_state is not None
    return best_acc, best_state
