"""Phase-III LOSO diagnostics: label coverage, baselines, confusion matrix, optional instance vote (val/test)."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

SplitName = Literal["test", "val"]

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def verify_split_label_coverage(tr_y: np.ndarray, split_y: np.ndarray, split_name: SplitName) -> Dict[str, Any]:
    """Ensure every remapped label on val/test split appears in training."""
    tr_set = set(tr_y.tolist())
    sp_set = set(split_y.tolist())
    missing = sorted(sp_set - tr_set)
    base = {"num_train_classes": len(tr_set), "ok": len(missing) == 0}
    if split_name == "test":
        return {**base, "num_test_classes_seen": len(sp_set), "test_labels_not_in_train": missing}
    return {**base, "num_val_classes_seen": len(sp_set), "val_labels_not_in_train": missing}


def verify_remapped_label_coverage(tr_y: np.ndarray, te_y: np.ndarray) -> Dict[str, Any]:
    """Backward-compatible alias for test split."""
    return verify_split_label_coverage(tr_y, te_y, "test")


def majority_class_baseline(train_y: np.ndarray, y_true_test: np.ndarray) -> Dict[str, float]:
    """Accuracy if we always predict the most frequent training class."""
    if train_y.size == 0:
        return {"majority_class": -1.0, "majority_baseline_test_acc": 0.0}
    bc = np.bincount(train_y.astype(np.int64), minlength=int(train_y.max()) + 1)
    maj = int(bc.argmax())
    acc = float((y_true_test.astype(np.int64) == maj).mean())
    return {"majority_class": float(maj), "majority_baseline_test_acc": acc}


def prediction_collapse_score(y_pred: np.ndarray, num_classes: int) -> Dict[str, float]:
    """Share of predictions in the single most common class (high => collapsed)."""
    if y_pred.size == 0:
        return {"top_pred_class_freq": 0.0, "normalized_entropy": 0.0}
    bc = np.bincount(y_pred.astype(np.int64), minlength=num_classes)
    total = bc.sum()
    top = float(bc.max() / max(total, 1))
    p = bc[bc > 0] / total
    h = -float((p * np.log(p + 1e-12)).sum())
    h_norm = h / max(np.log(max(num_classes, 2)), 1e-12)
    return {"top_pred_class_freq": top, "normalized_entropy": float(h_norm)}


def confusion_and_per_class(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> Dict[str, Any]:
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true.astype(np.int64), y_pred.astype(np.int64), labels=labels)
    p, r, f1, sup = precision_recall_fscore_support(
        y_true.astype(np.int64),
        y_pred.astype(np.int64),
        labels=labels,
        average=None,
        zero_division=0,
    )
    return {
        "confusion_matrix": cm.tolist(),
        "per_class_precision": p.tolist(),
        "per_class_recall": r.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": sup.astype(int).tolist(),
    }


def instance_majority_vote(
    y_true: np.ndarray, y_pred: np.ndarray, instance_ids: np.ndarray, num_classes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """One row per instance: label = first true label in group; pred = mode of preds in group."""
    if instance_ids.size != len(y_true) or instance_ids.size != len(y_pred):
        raise ValueError("instance_ids length must match y_true and y_pred")
    inst_to_indices: Dict[int, List[int]] = {}
    for i, gid in enumerate(instance_ids.astype(np.int64).tolist()):
        inst_to_indices.setdefault(gid, []).append(i)
    y_inst: List[int] = []
    p_inst: List[int] = []
    for _gid, idxs in sorted(inst_to_indices.items(), key=lambda x: x[0]):
        yt = y_true[idxs]
        y0 = int(yt[0])
        if not np.all(yt == y0):
            # Inconsistent labels in one group — keep first, still count once
            pass
        y_inst.append(y0)
        pv = y_pred[idxs].astype(np.int64)
        counts = np.bincount(pv, minlength=num_classes)
        p_inst.append(int(counts.argmax()))
    return np.array(y_inst, dtype=np.int64), np.array(p_inst, dtype=np.int64)


def build_split_diagnostics(
    *,
    split_name: SplitName,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_y_remapped: np.ndarray,
    num_classes: int,
    split_instance_ids: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    cov = verify_split_label_coverage(train_y_remapped, y_true, split_name)
    maj = majority_class_baseline(train_y_remapped, y_true)
    coll = prediction_collapse_score(y_pred, num_classes)
    cm_block = confusion_and_per_class(y_true, y_pred, num_classes)

    out: Dict[str, Any] = {
        "split": split_name,
        "num_classes_head": num_classes,
        "random_uniform_baseline_acc": float(1.0 / max(num_classes, 1)),
        "window_acc": float(accuracy_score(y_true, y_pred)),
        "window_bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "window_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "label_coverage": cov,
        "majority_baseline": maj,
        "prediction_collapse": coll,
        **cm_block,
    }

    split_label = "test" if split_name == "test" else "val"
    if split_instance_ids is None:
        out["instance_vote"] = {
            "applied": False,
            "reason": (
                f"no instance_ids in {split_label} split; each row is one TRAiLL instance (window==instance)."
            ),
        }
        out["instance_acc"] = out["window_acc"]
        out["instance_bal_acc"] = out["window_bal_acc"]
        out["instance_macro_f1"] = out["window_macro_f1"]
    else:
        n_unique = len(np.unique(split_instance_ids.astype(np.int64)))
        multi = n_unique < len(split_instance_ids)
        yi, pi = instance_majority_vote(y_true, y_pred, split_instance_ids, num_classes)
        out["instance_vote"] = {
            "applied": True,
            "num_groups": int(n_unique),
            "multi_window_groups": bool(multi),
        }
        out["instance_acc"] = float(accuracy_score(yi, pi))
        out["instance_bal_acc"] = float(balanced_accuracy_score(yi, pi))
        out["instance_macro_f1"] = float(f1_score(yi, pi, average="macro", zero_division=0))
        out["instance_confusion_matrix"] = confusion_matrix(
            yi, pi, labels=list(range(num_classes))
        ).tolist()

    return out


def build_test_diagnostics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_y_remapped: np.ndarray,
    num_classes: int,
    test_instance_ids: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Diagnostics on the held-out test split (same schema as ``build_split_diagnostics``)."""
    return build_split_diagnostics(
        split_name="test",
        y_true=y_true,
        y_pred=y_pred,
        train_y_remapped=train_y_remapped,
        num_classes=num_classes,
        split_instance_ids=test_instance_ids,
    )
