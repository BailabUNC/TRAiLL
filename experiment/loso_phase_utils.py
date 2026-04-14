from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from train_phase1_autoencoder import CNNAutoencoderStrided512
from train_phase2_simsiam import (
    CNNAutoencoderStrided512 as Phase2AE,
    Phase1Encoder,
    ProjectionHead,
    PredictionHead,
    simsiam_loss,
    augment_1d,
)
from tools.phase3_end_task_lib import Encoder1DCNN


def load_split(path: Path) -> Dict[str, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "features": data["features"].float(),  # [N,T,C]
        "labels": data["labels"].long(),       # [N]
        "subject_ids": data["subject_ids"].long(),
    }


def leakage_check(train: Dict[str, torch.Tensor], val: Dict[str, torch.Tensor], test: Dict[str, torch.Tensor]) -> None:
    train_sid = set(train["subject_ids"].unique().tolist())
    val_sid = set(val["subject_ids"].unique().tolist())
    test_sid = set(test["subject_ids"].unique().tolist())
    if train_sid.intersection(test_sid):
        raise RuntimeError("Leakage: train contains held-out subject")
    if val_sid.intersection(test_sid):
        raise RuntimeError("Leakage: val contains held-out subject")


def to_channels_first(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 1).contiguous()


def train_phase1_fold(
    train_x: torch.Tensor,
    val_x: torch.Tensor,
    out_ckpt: Path,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
) -> Dict[str, float]:
    model = CNNAutoencoderStrided512(in_channels=48, latent_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()

    xtr = to_channels_first(train_x)
    xva = to_channels_first(val_x)
    dl_tr = DataLoader(TensorDataset(xtr), batch_size=batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(TensorDataset(xva), batch_size=batch_size, shuffle=False, drop_last=False)

    best_val = float("inf")
    bad = 0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        for (x,) in dl_tr:
            x = x.to(device)
            noisy = x + torch.randn_like(x) * 0.2
            xr = model(noisy)
            loss = criterion(xr, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * x.size(0)
            tr_n += x.size(0)

        model.eval()
        va_loss = 0.0
        va_n = 0
        with torch.no_grad():
            for (x,) in dl_va:
                x = x.to(device)
                noisy = x + torch.randn_like(x) * 0.2
                xr = model(noisy)
                loss = criterion(xr, x)
                va_loss += loss.item() * x.size(0)
                va_n += x.size(0)

        tr_avg = tr_loss / max(tr_n, 1)
        va_avg = va_loss / max(va_n, 1)
        print(f"[phase1] ep {ep:02d}: train_loss={tr_avg:.5f}, val_loss={va_avg:.5f}")

        if va_avg < best_val - 1e-6:
            best_val = va_avg
            best_state = model.state_dict()
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    assert best_state is not None
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_ckpt)
    return {"best_val_loss": best_val}


class SimSiamValDataset(torch.utils.data.Dataset):
    def __init__(self, x: torch.Tensor):
        self.x = x

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        xx = self.x[idx]
        return augment_1d(xx), augment_1d(xx)


def train_phase2_fold(
    train_x: torch.Tensor,
    val_x: torch.Tensor,
    phase1_ckpt: Path,
    out_proj: Path,
    out_pred: Path,
    out_encoder: Path,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
) -> Dict[str, float]:
    train_cf = to_channels_first(train_x)
    val_cf = to_channels_first(val_x)

    ae = Phase2AE(in_channels=48, latent_channels=64).to(device)
    ckpt = torch.load(phase1_ckpt, map_location=device)
    filtered = {k: v for k, v in ckpt.items() if not k.startswith("enc5")}
    ae.load_state_dict(filtered, strict=False)
    encoder = Phase1Encoder(ae).to(device)
    for p in encoder.parameters():
        p.requires_grad_(False)
    for name, p in encoder.named_parameters():
        if any(name.startswith(f"encoder.{i}") for i in (3, 4)):
            p.requires_grad_(True)

    proj = ProjectionHead(in_dim=64, hidden_dim=128, out_dim=64).to(device)
    pred = PredictionHead(in_dim=64, hidden_dim=128, out_dim=64).to(device)

    optimizer = torch.optim.AdamW(
        list([p for p in encoder.parameters() if p.requires_grad]) + list(proj.parameters()) + list(pred.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    dl_tr = DataLoader(SimSiamValDataset(train_cf), batch_size=batch_size, shuffle=True, drop_last=True)
    dl_va = DataLoader(SimSiamValDataset(val_cf), batch_size=batch_size, shuffle=False, drop_last=False)

    best_val = float("inf")
    bad = 0
    best_proj = None
    best_pred = None
    best_enc = None

    for ep in range(1, epochs + 1):
        encoder.train()
        proj.train()
        pred.train()
        tr_loss = 0.0
        tr_n = 0
        for x1, x2 in dl_tr:
            x1, x2 = x1.to(device), x2.to(device)
            h1 = encoder(x1).mean(dim=2)
            h2 = encoder(x2).mean(dim=2)
            z1 = proj(h1)
            z2 = proj(h2)
            p1 = pred(z1)
            p2 = pred(z2)
            loss = simsiam_loss(p1, z2, p2, z1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * x1.size(0)
            tr_n += x1.size(0)

        encoder.eval()
        proj.eval()
        pred.eval()
        va_loss = 0.0
        va_n = 0
        with torch.no_grad():
            for x1, x2 in dl_va:
                x1, x2 = x1.to(device), x2.to(device)
                h1 = encoder(x1).mean(dim=2)
                h2 = encoder(x2).mean(dim=2)
                z1 = proj(h1)
                z2 = proj(h2)
                p1 = pred(z1)
                p2 = pred(z2)
                loss = simsiam_loss(p1, z2, p2, z1)
                va_loss += loss.item() * x1.size(0)
                va_n += x1.size(0)

        tr_avg = tr_loss / max(tr_n, 1)
        va_avg = va_loss / max(va_n, 1)
        print(f"[phase2] ep {ep:02d}: train_loss={tr_avg:.5f}, val_loss={va_avg:.5f}")

        if va_avg < best_val - 1e-6:
            best_val = va_avg
            best_proj = proj.state_dict()
            best_pred = pred.state_dict()
            best_enc = encoder.state_dict()
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    assert best_proj is not None and best_pred is not None and best_enc is not None
    out_proj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_proj, out_proj)
    torch.save(best_pred, out_pred)
    torch.save(best_enc, out_encoder)
    return {"best_val_loss": best_val}


def remap_labels(train_y: torch.Tensor, val_y: torch.Tensor, test_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    uniq = torch.unique(train_y)
    old2new = {int(v): i for i, v in enumerate(uniq.tolist())}
    mapper = np.vectorize(lambda z: old2new[int(z)])
    train_new = torch.from_numpy(mapper(train_y.numpy())).long()
    val_new = torch.from_numpy(mapper(val_y.numpy())).long()
    test_new = torch.from_numpy(mapper(test_y.numpy())).long()
    return train_new, val_new, test_new


def train_phase3_and_eval(
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

    pack = torch.load(encoder_ckpt, map_location=device)
    enc_ckpt = pack["enc"] if isinstance(pack, dict) and "enc" in pack else pack

    enc = Encoder1DCNN(C=48, embed_dim=256).to(device)
    enc.load_state_dict(enc_ckpt)
    for p in enc.parameters():
        p.requires_grad_(False)
    for idx in (4, 6):
        for p in enc.net[idx].parameters():
            p.requires_grad_(True)

    clf = nn.Sequential(
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, int(tr_y.max().item() + 1)),
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
            out = clf(enc(x))
            loss = criterion(out, y)
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
                pred = clf(enc(x)).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        val_acc = correct / max(total, 1)
        print(f"[phase3] ep {ep:02d}: val_acc={val_acc:.4f}")

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

    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in dl_te:
            x = x.to(device)
            pred = clf(enc(x)).argmax(1).cpu().numpy()
            y_true.extend(y.numpy().tolist())
            y_pred.extend(pred.tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    window_acc = accuracy_score(y_true, y_pred)
    window_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    window_bal = balanced_accuracy_score(y_true, y_pred)

    inst_acc = window_acc
    inst_f1 = window_f1
    inst_bal = window_bal

    return {
        "val_best_acc": float(best_acc),
        "window_acc": float(window_acc),
        "window_macro_f1": float(window_f1),
        "window_bal_acc": float(window_bal),
        "instance_acc": float(inst_acc),
        "instance_macro_f1": float(inst_f1),
        "instance_bal_acc": float(inst_bal),
    }
