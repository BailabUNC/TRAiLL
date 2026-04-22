"""LOSO Phase III variant: Phase-II CNN trunk + temporal mean pool + MLP classifier.

Does not use ``tools/phase3_end_task_lib.Encoder1DCNN`` or manuscript numpy baseline.
Encoder weights are loaded from per-fold Phase-II checkpoints (``loso_*_encoder.pth``).

**Val vs held-out test gap (LOSO):** inner val only contains *train* subjects while the outer
test is a *new* subject, so pooled val accuracy is often optimistic. Mitigations implemented here:

- optional **min-subject** inner val score (early stop on the worst train subject's val acc);
- **label smoothing** and **mixup on pooled features** to reduce overfitting to val-subject mix.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from train_phase2_simsiam import CNNAutoencoderStrided512 as Phase2AE, Phase1Encoder

from experiment.loso.loso_phase_utils import remap_labels, to_channels_first
from experiment.loso.phase3_loso_diagnostics import build_split_diagnostics, build_test_diagnostics

_TRUNK_LATENT = 64


def _set_encoder_trainable(enc: Phase1Encoder, unfreeze_from: int) -> None:
    """Unfreeze ``encoder[i]`` for ``i in [unfreeze_from, 5)``. ``unfreeze_from=0`` trains the full trunk."""
    for p in enc.parameters():
        p.requires_grad_(False)
    for idx in range(max(0, unfreeze_from), 5):
        for p in enc.encoder[idx].parameters():
            p.requires_grad_(True)


ValSelectMetric = Literal["mean", "min_subj"]


def _val_accuracy_by_subject(
    enc: Phase1Encoder,
    clf: nn.Module,
    xva: torch.Tensor,
    va_y: torch.Tensor,
    val_sid: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> Tuple[float, float, Dict[int, float]]:
    """Returns (mean_acc, min_per_subject_acc, acc_per_subject_id)."""
    enc.eval()
    clf.eval()
    dl = DataLoader(TensorDataset(xva, va_y, val_sid), batch_size=batch_size, shuffle=False, drop_last=False)
    per_sid_correct: Dict[int, int] = {}
    per_sid_total: Dict[int, int] = {}
    with torch.no_grad():
        for x, y, sid in dl:
            x = x.to(device)
            y = y.to(device)
            sid = sid.to(device)
            pred = clf(enc(x).mean(dim=2)).argmax(1)
            ok = pred == y
            for i in range(x.size(0)):
                s = int(sid[i].item())
                per_sid_correct[s] = per_sid_correct.get(s, 0) + int(ok[i].item())
                per_sid_total[s] = per_sid_total.get(s, 0) + 1
    accs: List[float] = []
    detail: Dict[int, float] = {}
    for s in sorted(per_sid_total.keys()):
        t = per_sid_total[s]
        c = per_sid_correct.get(s, 0)
        a = c / max(t, 1)
        detail[s] = a
        accs.append(a)
    mean_acc = float(sum(per_sid_correct.values()) / max(sum(per_sid_total.values()), 1))
    min_subj = float(min(accs)) if accs else mean_acc
    return mean_acc, min_subj, detail


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
    *,
    lr_head: float = 3e-3,
    lr_enc: float = 5e-4,
    weight_decay: float = 1e-4,
    dropout: float = 0.1,
    hidden_dim: int = 256,
    unfreeze_from: int = 0,
    stage2_epoch: Optional[int] = None,
    stage2_unfreeze_from: Optional[int] = None,
    max_grad_norm: Optional[float] = 1.0,
    val_subject_ids: Optional[torch.Tensor] = None,
    val_select_metric: ValSelectMetric = "min_subj",
    label_smoothing: float = 0.0,
    mixup_alpha: float = 0.0,
    class_weighted_ce: bool = False,
    class_weight_power: float = 1.0,
    test_instance_ids: Optional[torch.Tensor] = None,
    val_instance_ids: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    if not 0 <= unfreeze_from <= 4:
        raise ValueError(f"unfreeze_from must be in [0, 4], got {unfreeze_from}")
    if stage2_unfreeze_from is not None and not 0 <= stage2_unfreeze_from <= 4:
        raise ValueError(f"stage2_unfreeze_from must be in [0, 4], got {stage2_unfreeze_from}")
    if stage2_epoch is not None and stage2_epoch <= 1:
        raise ValueError(f"stage2_epoch must be >= 2 when set, got {stage2_epoch}")

    tr_y, va_y, te_y = remap_labels(train_y, val_y, test_y)

    xtr = to_channels_first(train_x)
    xva = to_channels_first(val_x)
    xte = to_channels_first(test_x)

    val_sid = val_subject_ids.long() if val_subject_ids is not None else None
    n_val_subj = int(torch.unique(val_sid).numel()) if val_sid is not None else 0
    effective_val_metric: ValSelectMetric = val_select_metric
    if val_select_metric == "min_subj" and (val_sid is None or n_val_subj < 2):
        effective_val_metric = "mean"

    dl_tr = DataLoader(TensorDataset(xtr, tr_y), batch_size=batch_size, shuffle=True, drop_last=False)
    dl_te = DataLoader(TensorDataset(xte, te_y), batch_size=batch_size, shuffle=False, drop_last=False)

    enc = build_phase2_encoder(device, encoder_ckpt)
    _set_encoder_trainable(enc, unfreeze_from)
    stage2_applied = False

    num_classes = int(tr_y.max().item() + 1)
    clf = nn.Sequential(
        nn.Linear(_TRUNK_LATENT, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": list(clf.parameters()), "lr": lr_head},
            {"params": [p for p in enc.parameters() if p.requires_grad], "lr": lr_enc},
        ],
        weight_decay=weight_decay,
    )
    class_weights: Optional[torch.Tensor] = None
    if class_weighted_ce:
        # Inverse-frequency class weights, normalized to mean=1 for stable scale.
        counts = torch.bincount(tr_y, minlength=num_classes).float()
        inv = torch.pow(counts.clamp_min(1.0), -class_weight_power)
        class_weights = (inv / inv.mean()).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_select = -1.0
    best_mean = 0.0
    best_min_subj = 0.0
    bad = 0
    best_state = None
    max_train_acc_epoch = 0.0

    def _train_set_accuracy() -> float:
        enc.eval()
        clf.eval()
        c = t = 0
        with torch.no_grad():
            for x, y in DataLoader(TensorDataset(xtr, tr_y), batch_size=batch_size, shuffle=False):
                x = x.to(device)
                y = y.to(device)
                pred = clf(enc(x).mean(dim=2)).argmax(1)
                c += int((pred == y).sum().item())
                t += int(y.size(0))
        enc.train()
        clf.train()
        return c / max(t, 1)

    for ep in range(1, epochs + 1):
        if (
            not stage2_applied
            and stage2_epoch is not None
            and stage2_unfreeze_from is not None
            and ep >= stage2_epoch
        ):
            _set_encoder_trainable(enc, stage2_unfreeze_from)
            stage2_applied = True
            print(
                f"[phase3-trunk] progressive unfreeze at ep {ep:02d}: "
                f"{unfreeze_from} -> {stage2_unfreeze_from}"
            )

        enc.train()
        clf.train()
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            if mixup_alpha > 0 and x.size(0) >= 2:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                idx = torch.randperm(x.size(0), device=device)
                h = enc(x).mean(dim=2)
                h = lam * h + (1.0 - lam) * h[idx]
                out = clf(h)
                yb = y[idx]
                loss = lam * criterion(out, y) + (1.0 - lam) * criterion(out, yb)
            else:
                h = enc(x).mean(dim=2)
                loss = criterion(clf(h), y)
            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm is not None and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(list(clf.parameters()) + list(enc.parameters()), max_grad_norm)
            optimizer.step()
        scheduler.step()

        if val_sid is not None:
            val_mean, val_min_subj, _detail = _val_accuracy_by_subject(enc, clf, xva, va_y, val_sid, device, batch_size)
        else:
            enc.eval()
            clf.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in DataLoader(TensorDataset(xva, va_y), batch_size=batch_size, shuffle=False):
                    x, y = x.to(device), y.to(device)
                    pred = clf(enc(x).mean(dim=2)).argmax(1)
                    correct += (pred == y).sum().item()
                    total += y.size(0)
            val_mean = correct / max(total, 1)
            val_min_subj = val_mean

        val_select = val_min_subj if effective_val_metric == "min_subj" else val_mean
        tag = "min_subj" if effective_val_metric == "min_subj" else "mean"
        tr_acc_ep = _train_set_accuracy()
        if tr_acc_ep > max_train_acc_epoch:
            max_train_acc_epoch = tr_acc_ep
        print(
            f"[phase3-trunk] ep {ep:02d}: train_acc={tr_acc_ep:.4f} val_mean={val_mean:.4f} val_min_subj={val_min_subj:.4f} "
            f"val_select({tag})={val_select:.4f}"
        )

        if val_select > best_select + 1e-6:
            best_select = val_select
            best_mean = val_mean
            best_min_subj = val_min_subj
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

    train_correct = 0
    train_total = 0
    with torch.no_grad():
        for x, y in DataLoader(TensorDataset(xtr, tr_y), batch_size=batch_size, shuffle=False):
            x = x.to(device)
            y = y.to(device)
            pred = clf(enc(x).mean(dim=2)).argmax(1)
            train_correct += int((pred == y).sum().item())
            train_total += int(y.size(0))
    train_window_acc = train_correct / max(train_total, 1)

    y_va_true: list[int] = []
    y_va_pred: list[int] = []
    with torch.no_grad():
        for x, y in DataLoader(TensorDataset(xva, va_y), batch_size=batch_size, shuffle=False):
            x = x.to(device)
            pred = clf(enc(x).mean(dim=2)).argmax(1).cpu().numpy()
            y_va_true.extend(y.numpy().tolist())
            y_va_pred.extend(pred.tolist())
    y_va_true_arr = np.array(y_va_true)
    y_va_pred_arr = np.array(y_va_pred)
    val_inst_np = val_instance_ids.cpu().numpy() if val_instance_ids is not None else None
    val_diag = build_split_diagnostics(
        split_name="val",
        y_true=y_va_true_arr,
        y_pred=y_va_pred_arr,
        train_y_remapped=tr_y.numpy(),
        num_classes=num_classes,
        split_instance_ids=val_inst_np,
    )
    if not val_diag["label_coverage"]["ok"]:
        print(
            "[phase3-trunk] WARNING: val contains remapped labels not seen in train: "
            f"{val_diag['label_coverage'].get('val_labels_not_in_train', [])}"
        )

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
    inst_np = test_instance_ids.cpu().numpy() if test_instance_ids is not None else None

    diag = build_test_diagnostics(
        y_true=y_true_arr,
        y_pred=y_pred_arr,
        train_y_remapped=tr_y.numpy(),
        num_classes=num_classes,
        test_instance_ids=inst_np,
    )
    if not diag["label_coverage"]["ok"]:
        print(
            "[phase3-trunk] WARNING: test contains remapped labels not seen in train: "
            f"{diag['label_coverage']['test_labels_not_in_train']}"
        )

    out: Dict[str, Any] = {
        "val_best_select": float(best_select),
        "val_best_mean_acc": float(best_mean),
        "val_best_min_subj_acc": float(best_min_subj),
        "val_select_metric": effective_val_metric,
        "val_best_acc": float(best_select),
        "train_window_acc": float(train_window_acc),
        "train_window_acc_peak_during_fit": float(max_train_acc_epoch),
        "window_acc": float(diag["window_acc"]),
        "window_macro_f1": float(diag["window_macro_f1"]),
        "window_bal_acc": float(diag["window_bal_acc"]),
        "instance_acc": float(diag["instance_acc"]),
        "instance_macro_f1": float(diag["instance_macro_f1"]),
        "instance_bal_acc": float(diag["instance_bal_acc"]),
        "test_diagnostics": diag,
        "val_diagnostics": val_diag,
        "train_hparams": {
            "lr_head": lr_head,
            "lr_enc": lr_enc,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "hidden_dim": hidden_dim,
            "unfreeze_from": unfreeze_from,
            "stage2_epoch": stage2_epoch,
            "stage2_unfreeze_from": stage2_unfreeze_from,
            "max_grad_norm": max_grad_norm,
            "val_select_metric_requested": val_select_metric,
            "label_smoothing": label_smoothing,
            "mixup_alpha": mixup_alpha,
            "class_weighted_ce": class_weighted_ce,
            "class_weight_power": class_weight_power,
        },
    }
    return out
