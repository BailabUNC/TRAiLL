"""LOSO Phase III variant: Phase-II CNN trunk + temporal mean pool + MLP classifier.

Does not use ``tools/phase3_end_task_lib.Encoder1DCNN`` or manuscript numpy baseline.
Encoder weights are loaded from per-fold Phase-II checkpoints (``loso_*_encoder.pth``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from train_phase2_simsiam import CNNAutoencoderStrided512 as Phase2AE, Phase1Encoder

from experiment.loso_phase_utils import remap_labels, to_channels_first

_TRUNK_LATENT = 64


def build_phase2_encoder(device: torch.device, encoder_ckpt: Path) -> Phase1Encoder:
    ae = Phase2AE(in_channels=48, latent_channels=_TRUNK_LATENT).to(device)
    enc = Phase1Encoder(ae).to(device)
    state = torch.load(encoder_ckpt, map_location=device, weights_only=False)
    enc.load_state_dict(state)
    return enc


def train_phase3_trunk_and_eval(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    encoder_ckpt: Path,
    out_model: Path,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
) -> Dict[str, float]:
    tr_y, va_y, te_y = remap_labels(train_y, val_y, test_y)

    xtr = to_channels_first(train_x)
    xva = to_channels_first(val_x)
    xte = to_channels_first(test_x)

    dl_tr = DataLoader(TensorDataset(xtr, tr_y), batch_size=batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(TensorDataset(xva, va_y), batch_size=batch_size, shuffle=False, drop_last=False)
    dl_te = DataLoader(TensorDataset(xte, te_y), batch_size=batch_size, shuffle=False, drop_last=False)

    enc = build_phase2_encoder(device, encoder_ckpt)
    for p in enc.parameters():
        p.requires_grad_(False)
    for idx in (3, 4):
        for p in enc.encoder[idx].parameters():
            p.requires_grad_(True)

    num_classes = int(tr_y.max().item() + 1)
    clf = nn.Sequential(
        nn.Linear(_TRUNK_LATENT, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": list(clf.parameters()), "lr": 1e-3},
            {"params": [p for p in enc.parameters() if p.requires_grad], "lr": 1e-4},
        ],
        weight_decay=3e-4,
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_acc = 0.0
    bad = 0
    best_state = None

    for ep in range(1, epochs + 1):
        enc.train()
        clf.train()
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            h = enc(x).mean(dim=2)
            loss = criterion(clf(h), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        enc.eval()
        clf.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(device), y.to(device)
                pred = clf(enc(x).mean(dim=2)).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        val_acc = correct / max(total, 1)
        print(f"[phase3-trunk] ep {ep:02d}: val_acc={val_acc:.4f}")

        if val_acc > best_acc + 1e-6:
            best_acc = val_acc
            best_state = {"enc": enc.state_dict(), "clf": clf.state_dict()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    assert best_state is not None
    out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_model)

    enc.load_state_dict(best_state["enc"])
    clf.load_state_dict(best_state["clf"])
    enc.eval()
    clf.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for x, y in dl_te:
            x = x.to(device)
            pred = clf(enc(x).mean(dim=2)).argmax(1).cpu().numpy()
            y_true.extend(y.numpy().tolist())
            y_pred.extend(pred.tolist())

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    window_acc = accuracy_score(y_true_arr, y_pred_arr)
    window_f1 = f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
    window_bal = balanced_accuracy_score(y_true_arr, y_pred_arr)

    return {
        "val_best_acc": float(best_acc),
        "window_acc": float(window_acc),
        "window_macro_f1": float(window_f1),
        "window_bal_acc": float(window_bal),
        "instance_acc": float(window_acc),
        "instance_macro_f1": float(window_f1),
        "instance_bal_acc": float(window_bal),
    }
