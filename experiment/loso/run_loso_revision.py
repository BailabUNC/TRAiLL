from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from path_config import ROOT
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

    # In current tensors, each row corresponds to one segmented instance/window.
    window_acc = accuracy_score(y_true, y_pred)
    window_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    window_bal = balanced_accuracy_score(y_true, y_pred)

    # Instance-level equals row-level here (one row per instance after preprocessing).
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 3-fold LOSO experiment on prepared data/loso splits.")
    parser.add_argument("--loso-dir", type=str, default="data/loso")
    parser.add_argument("--phase1-epochs", type=int, default=20)
    parser.add_argument("--phase2-epochs", type=int, default=20)
    parser.add_argument("--phase3-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--phase1-lr", type=float, default=1e-3)
    parser.add_argument("--phase2-lr", type=float, default=2e-3)
    parser.add_argument("--phase3-encoder-ckpt", type=str, default="checkpoints/phase3/model_best.pth")
    parser.add_argument("--smoke", action="store_true", help="Run very short settings to validate pipeline wiring.")
    args = parser.parse_args()

    if args.smoke:
        args.phase1_epochs = 1
        args.phase2_epochs = 1
        args.phase3_epochs = 1
        args.patience = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loso_dir = ROOT / args.loso_dir
    out_metrics_dir = ROOT / "outputs" / "loso" / "phase3" / "results"
    out_metrics_dir.mkdir(parents=True, exist_ok=True)

    summary = {"folds": {}, "device": str(device), "smoke": bool(args.smoke)}

    for fold in ("fold1", "fold2", "fold3"):
        print(f"\n===== {fold} =====")
        outer_train = load_split(loso_dir / f"{fold}_outer_train.pt")
        outer_test = load_split(loso_dir / f"{fold}_outer_test.pt")
        inner_train = load_split(loso_dir / f"{fold}_inner_train.pt")
        inner_val = load_split(loso_dir / f"{fold}_inner_val.pt")
        leakage_check(inner_train, inner_val, outer_test)

        phase1_ckpt = ROOT / "checkpoints" / "phase1" / f"loso_{fold}_autoencoder.pth"
        phase2_proj = ROOT / "checkpoints" / "phase2" / "simsiam" / f"loso_{fold}_proj_head.pth"
        phase2_pred = ROOT / "checkpoints" / "phase2" / "simsiam" / f"loso_{fold}_pred_head.pth"
        phase2_enc = ROOT / "checkpoints" / "phase2" / "simsiam" / f"loso_{fold}_encoder.pth"
        phase3_ckpt = ROOT / "checkpoints" / "phase3" / f"loso_{fold}_end_task.pth"

        p1 = train_phase1_fold(
            inner_train["features"],
            inner_val["features"],
            phase1_ckpt,
            device,
            epochs=args.phase1_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.phase1_lr,
        )
        p2 = train_phase2_fold(
            inner_train["features"],
            inner_val["features"],
            phase1_ckpt,
            phase2_proj,
            phase2_pred,
            phase2_enc,
            device,
            epochs=args.phase2_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.phase2_lr,
        )
        p3 = train_phase3_and_eval(
            inner_train["features"],
            inner_train["labels"],
            inner_val["features"],
            inner_val["labels"],
            outer_test["features"],
            outer_test["labels"],
            ROOT / args.phase3_encoder_ckpt,
            phase3_ckpt,
            device,
            epochs=args.phase3_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
        )

        fold_result = {
            "phase1": p1,
            "phase2": p2,
            "phase3": p3,
            "outer_test_subject_ids": sorted(set(outer_test["subject_ids"].tolist())),
            "outer_train_subject_ids": sorted(set(outer_train["subject_ids"].tolist())),
            "inner_train_samples": int(inner_train["features"].shape[0]),
            "inner_val_samples": int(inner_val["features"].shape[0]),
            "outer_test_samples": int(outer_test["features"].shape[0]),
        }
        summary["folds"][fold] = fold_result

        fold_out = out_metrics_dir / f"loso_{fold}_results.json"
        with fold_out.open("w", encoding="utf-8") as f:
            json.dump(fold_result, f, indent=2)

    wacc = [summary["folds"][f]["phase3"]["window_acc"] for f in ("fold1", "fold2", "fold3")]
    ifacc = [summary["folds"][f]["phase3"]["instance_acc"] for f in ("fold1", "fold2", "fold3")]
    summary["aggregate"] = {
        "window_acc_mean": float(np.mean(wacc)),
        "window_acc_std": float(np.std(wacc)),
        "instance_acc_mean": float(np.mean(ifacc)),
        "instance_acc_std": float(np.std(ifacc)),
    }
    with (out_metrics_dir / "loso_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nLOSO run completed. Summary:")
    print(json.dumps(summary["aggregate"], indent=2))


if __name__ == "__main__":
    main()
