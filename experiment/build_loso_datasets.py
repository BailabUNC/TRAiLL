from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np
import torch

from sklearn.model_selection import StratifiedShuffleSplit

sys.path.append(str(Path(__file__).resolve().parents[1]))
from path_config import ROOT


@dataclass(frozen=True)
class SubjectConfig:
    name: str
    processed_pt: str


SUBJECTS: Dict[str, SubjectConfig] = {
    "S1": SubjectConfig(
        name="daniela_group_1",
        processed_pt="data/.processed/concatenated_dataset-daniela-letters-group_1.pt",
    ),
    "S2": SubjectConfig(
        name="daniela_group_2",
        processed_pt="data/.processed/concatenated_dataset-daniela-letters-group_2.pt",
    ),
    "S3": SubjectConfig(
        name="daniela_group_3",
        processed_pt="data/.processed/concatenated_dataset-daniela-letters-group_3.pt",
    ),
}

FOLDS: List[Tuple[str, List[str], str]] = [
    ("fold1", ["S2", "S3"], "S1"),
    ("fold2", ["S1", "S3"], "S2"),
    ("fold3", ["S1", "S2"], "S3"),
]


def _load_subject_tensor(key: str) -> Dict[str, torch.Tensor]:
    cfg = SUBJECTS[key]
    path = ROOT / cfg.processed_pt
    if not path.exists():
        raise FileNotFoundError(f"Missing subject dataset: {path}")

    data = torch.load(path, map_location="cpu", weights_only=False)
    features = data["features"].float()
    labels = data["labels"].long()
    if features.ndim != 3:
        raise ValueError(f"{path} features must be [N,T,C], got {tuple(features.shape)}")
    if labels.ndim != 1:
        raise ValueError(f"{path} labels must be [N], got {tuple(labels.shape)}")
    if features.shape[0] != labels.shape[0]:
        raise ValueError(f"{path} feature/label length mismatch")

    subject_idx = int(key[1])  # S1 -> 1
    subject_ids = torch.full((features.shape[0],), subject_idx, dtype=torch.long)
    return {
        "features": features,
        "labels": labels,
        "subject_ids": subject_ids,
        "subject_key": key,
        "subject_name": cfg.name,
        "source_path": str(path),
    }


def _concat_parts(parts: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "features": torch.cat([p["features"] for p in parts], dim=0),
        "labels": torch.cat([p["labels"] for p in parts], dim=0),
        "subject_ids": torch.cat([p["subject_ids"] for p in parts], dim=0),
    }


def _subset_pack(pack: Dict[str, torch.Tensor], idx: np.ndarray) -> Dict[str, torch.Tensor]:
    tidx = torch.as_tensor(idx, dtype=torch.long)
    return {
        "features": pack["features"].index_select(0, tidx),
        "labels": pack["labels"].index_select(0, tidx),
        "subject_ids": pack["subject_ids"].index_select(0, tidx),
    }


def _label_hist(y: torch.Tensor) -> Dict[str, int]:
    vals, cnts = torch.unique(y, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(vals.tolist(), cnts.tolist())}


def _make_summary(name: str, pack: Dict[str, torch.Tensor], fold: str) -> Dict[str, object]:
    x = pack["features"]
    y = pack["labels"]
    sid = pack["subject_ids"]
    return {
        "name": name,
        "fold": fold,
        "shape": list(x.shape),  # [N,T,C]
        "num_samples": int(x.shape[0]),
        "num_timesteps": int(x.shape[1]),
        "num_channels": int(x.shape[2]),
        "label_hist": _label_hist(y),
        "subject_id_hist": _label_hist(sid),
    }


def main() -> None:
    out_dir = ROOT / "data" / "loso"
    out_dir.mkdir(parents=True, exist_ok=True)

    subject_data = {k: _load_subject_tensor(k) for k in SUBJECTS}

    manifest: Dict[str, object] = {
        "split_rule": "Outer LOSO by subject (S1/S2/S3), inner stratified 80/20 split on outer train.",
        "subjects": {
            key: {
                "name": SUBJECTS[key].name,
                "source_pt": SUBJECTS[key].processed_pt,
            }
            for key in SUBJECTS
        },
        "folds": {},
    }

    for fold_name, train_subjects, test_subject in FOLDS:
        outer_train_pack = _concat_parts([subject_data[s] for s in train_subjects])
        outer_test_pack = _concat_parts([subject_data[test_subject]])

        y_train = outer_train_pack["labels"].numpy()
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        idx_train, idx_val = next(splitter.split(np.arange(len(y_train)), y_train))
        inner_train_pack = _subset_pack(outer_train_pack, idx_train)
        inner_val_pack = _subset_pack(outer_train_pack, idx_val)

        # Leakage guard: subject-wise isolation between outer train and outer test.
        train_subj = set(inner_train_pack["subject_ids"].unique().tolist()) | set(
            inner_val_pack["subject_ids"].unique().tolist()
        )
        test_subj = set(outer_test_pack["subject_ids"].unique().tolist())
        if train_subj.intersection(test_subj):
            raise RuntimeError(f"Leakage detected in {fold_name}: subject overlap train vs test")

        fold_prefix = out_dir / fold_name
        outer_train_path = fold_prefix.with_name(f"{fold_name}_outer_train.pt")
        outer_test_path = fold_prefix.with_name(f"{fold_name}_outer_test.pt")
        inner_train_path = fold_prefix.with_name(f"{fold_name}_inner_train.pt")
        inner_val_path = fold_prefix.with_name(f"{fold_name}_inner_val.pt")

        torch.save(outer_train_pack, outer_train_path)
        torch.save(outer_test_pack, outer_test_path)
        torch.save(inner_train_pack, inner_train_path)
        torch.save(inner_val_pack, inner_val_path)

        fold_info = {
            "outer_train_subjects": train_subjects,
            "outer_test_subject": test_subject,
            "files": {
                "outer_train": str(outer_train_path.relative_to(ROOT)),
                "outer_test": str(outer_test_path.relative_to(ROOT)),
                "inner_train": str(inner_train_path.relative_to(ROOT)),
                "inner_val": str(inner_val_path.relative_to(ROOT)),
            },
            "summaries": {
                "outer_train": _make_summary("outer_train", outer_train_pack, fold_name),
                "outer_test": _make_summary("outer_test", outer_test_pack, fold_name),
                "inner_train": _make_summary("inner_train", inner_train_pack, fold_name),
                "inner_val": _make_summary("inner_val", inner_val_pack, fold_name),
            },
        }
        manifest["folds"][fold_name] = fold_info

    manifest_path = out_dir / "loso_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    readme_path = out_dir / "README.md"
    readme_lines = [
        "# LOSO Prepared Datasets",
        "",
        "This folder stores fold-ready datasets for outer-subject LOSO and inner train/val splits.",
        "",
        "## File format",
        "",
        "Each `.pt` file is a dictionary with:",
        "",
        "- `features`: `FloatTensor [N, T, C]`",
        "- `labels`: `LongTensor [N]`",
        "- `subject_ids`: `LongTensor [N]` (`1`=S1, `2`=S2, `3`=S3)",
        "",
        "## Subject mapping",
        "",
        "- `S1` -> `daniela_group_1`",
        "- `S2` -> `daniela_group_2`",
        "- `S3` -> `daniela_group_3`",
        "",
        "## Folds",
        "",
        "- `fold1`: train on `S2+S3`, test on `S1`",
        "- `fold2`: train on `S1+S3`, test on `S2`",
        "- `fold3`: train on `S1+S2`, test on `S3`",
        "",
        "Inner split is stratified 80/20 over outer-train labels.",
        "",
        "See `loso_manifest.json` for per-file sample counts and label distributions.",
        "",
    ]
    readme_path.write_text("\n".join(readme_lines), encoding="utf-8")

    print(f"Saved LOSO datasets and manifest to {out_dir}")


if __name__ == "__main__":
    main()
