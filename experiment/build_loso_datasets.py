from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys

import numpy as np
import torch

from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from path_config import ROOT
from traill_data_preparation.traill_loso_session_resplit import mix_and_split_sessions


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


def _load_subject_tensor(key: str) -> Dict[str, Any]:
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


def _load_mixed_session_tensor(key: str, path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing mixed session dataset: {path}")
    data = torch.load(path, map_location="cpu", weights_only=False)
    features = data["features"].float()
    labels = data["labels"].long()
    subject_ids = data["subject_ids"].long()
    if features.ndim != 3:
        raise ValueError(f"{path} features must be [N,T,C], got {tuple(features.shape)}")
    if labels.ndim != 1:
        raise ValueError(f"{path} labels must be [N], got {tuple(labels.shape)}")
    if features.shape[0] != labels.shape[0]:
        raise ValueError(f"{path} feature/label length mismatch")
    if subject_ids.shape[0] != features.shape[0]:
        raise ValueError(f"{path} subject_ids length mismatch")

    out: Dict[str, Any] = {
        "features": features,
        "labels": labels,
        "subject_ids": subject_ids,
        "subject_key": key,
        "subject_name": f"mixed_session_{key}",
        "source_path": str(path),
    }
    if "instance_ids" in data:
        out["instance_ids"] = data["instance_ids"].long()
    if "origin_group_ids" in data:
        out["origin_group_ids"] = data["origin_group_ids"].long()
    return out


def _concat_parts(parts: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    base: Dict[str, torch.Tensor] = {
        "features": torch.cat([p["features"] for p in parts], dim=0),
        "labels": torch.cat([p["labels"] for p in parts], dim=0),
        "subject_ids": torch.cat([p["subject_ids"] for p in parts], dim=0),
    }
    if any("instance_ids" in p for p in parts):
        chunks: List[torch.Tensor] = []
        off = 0
        for p in parts:
            n = int(p["features"].shape[0])
            if "instance_ids" in p:
                chunks.append(p["instance_ids"].long() + off)
            else:
                chunks.append(torch.arange(n, dtype=torch.long) + off)
            off += n
        base["instance_ids"] = torch.cat(chunks, dim=0)
    if all("origin_group_ids" in p for p in parts):
        base["origin_group_ids"] = torch.cat([p["origin_group_ids"].long() for p in parts], dim=0)
    elif any("origin_group_ids" in p for p in parts):
        raise ValueError("origin_group_ids must be present for every part or for none.")
    return base


def _subset_pack(pack: Dict[str, torch.Tensor], idx: np.ndarray) -> Dict[str, torch.Tensor]:
    tidx = torch.as_tensor(idx, dtype=torch.long)
    out: Dict[str, torch.Tensor] = {
        "features": pack["features"].index_select(0, tidx),
        "labels": pack["labels"].index_select(0, tidx),
        "subject_ids": pack["subject_ids"].index_select(0, tidx),
    }
    for opt in ("instance_ids", "origin_group_ids"):
        if opt in pack:
            out[opt] = pack[opt].index_select(0, tidx)
    return out


def _label_hist(y: torch.Tensor) -> Dict[str, int]:
    vals, cnts = torch.unique(y, return_counts=True)
    return {str(int(v)): int(c) for v, c in zip(vals.tolist(), cnts.tolist())}


def _make_summary(name: str, pack: Dict[str, torch.Tensor], fold: str) -> Dict[str, object]:
    x = pack["features"]
    y = pack["labels"]
    sid = pack["subject_ids"]
    summary: Dict[str, object] = {
        "name": name,
        "fold": fold,
        "shape": list(x.shape),  # [N,T,C]
        "num_samples": int(x.shape[0]),
        "num_timesteps": int(x.shape[1]),
        "num_channels": int(x.shape[2]),
        "label_hist": _label_hist(y),
        "subject_id_hist": _label_hist(sid),
    }
    if "origin_group_ids" in pack:
        summary["origin_group_id_hist"] = _label_hist(pack["origin_group_ids"])
    return summary


def _write_folds(
    out_dir: Path,
    subject_data: Dict[str, Dict[str, Any]],
    manifest: Dict[str, object],
    readme_lines: List[str],
    inner_split_seed: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest["folds"] = {}

    for fold_name, train_subjects, test_subject in FOLDS:
        outer_train_pack = _concat_parts([subject_data[s] for s in train_subjects])
        outer_test_pack = _concat_parts([subject_data[test_subject]])

        y_train = outer_train_pack["labels"].numpy()
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=inner_split_seed)
        idx_train, idx_val = next(splitter.split(np.arange(len(y_train)), y_train))
        inner_train_pack = _subset_pack(outer_train_pack, idx_train)
        inner_val_pack = _subset_pack(outer_train_pack, idx_val)

        train_subj = set(inner_train_pack["subject_ids"].unique().tolist()) | set(
            inner_val_pack["subject_ids"].unique().tolist()
        )
        test_subj = set(outer_test_pack["subject_ids"].unique().tolist())
        if train_subj.intersection(test_subj):
            raise RuntimeError(f"Leakage detected in {fold_name}: subject overlap train vs test")

        outer_train_path = out_dir / f"{fold_name}_outer_train.pt"
        outer_test_path = out_dir / f"{fold_name}_outer_test.pt"
        inner_train_path = out_dir / f"{fold_name}_inner_train.pt"
        inner_val_path = out_dir / f"{fold_name}_inner_val.pt"

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
    readme_path.write_text("\n".join(readme_lines), encoding="utf-8")
    print(f"Saved LOSO datasets and manifest to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/loso fold .pt files for LOSO.")
    parser.add_argument(
        "--mixed",
        action="store_true",
        help="Shuffle-merge daniela group_1/2/3, split into 3 pseudo-sessions, then same LOSO folds.",
    )
    parser.add_argument("--input-dir", type=str, default="data/.processed")
    parser.add_argument("--glob", type=str, default="concatenated_dataset-daniela-letters-group_*.pt")
    parser.add_argument("--output-dir", type=str, default="data/loso")
    parser.add_argument("--mix-seed", type=int, default=42, help="RNG for pseudo-session assignment.")
    parser.add_argument(
        "--inner-seed",
        type=int,
        default=42,
        help="RNG for stratified 80/20 inner train/val on outer-train windows.",
    )
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir

    if args.mixed:
        mix_manifest = mix_and_split_sessions(args.input_dir, args.glob, out_dir, args.mix_seed)
        subject_data = {
            "S1": _load_mixed_session_tensor("S1", out_dir / "mixed_session1.pt"),
            "S2": _load_mixed_session_tensor("S2", out_dir / "mixed_session2.pt"),
            "S3": _load_mixed_session_tensor("S3", out_dir / "mixed_session3.pt"),
        }
        manifest: Dict[str, object] = {
            "split_rule": (
                "Outer LOSO by pseudo-session (S1–S3) after mixing daniela group_1/2/3; "
                "inner stratified 80/20 on outer-train windows."
            ),
            "mix_sessions": mix_manifest,
            "subjects": {
                k: {
                    "name": subject_data[k]["subject_name"],
                    "source_pt": subject_data[k]["source_path"],
                }
                for k in subject_data
            },
        }
        readme_lines = [
            "# LOSO Prepared Datasets (mixed pseudo-sessions)",
            "",
            "Group `concatenated_dataset-daniela-letters-group_*.pt` files were **shuffled and split**",
            "into three pseudo-sessions (`mixed_session1.pt` … `mixed_session3.pt`).",
            "LOSO then treats `S1`/`S2`/`S3` as those sessions (not raw recording groups).",
            "",
            "Provenance: `origin_group_ids` in each `.pt` is `1/2/3` for original daniela group.",
            "",
            "## File format",
            "",
            "Each fold `.pt` dictionary includes:",
            "",
            "- `features`, `labels`, `subject_ids`",
            "- `instance_ids` (optional): offset so IDs are unique across concatenated sessions",
            "- `origin_group_ids` (optional): original daniela group index per window",
            "",
            "## Folds",
            "",
            "- `fold1`: train on `S2+S3`, test on `S1`",
            "- `fold2`: train on `S1+S3`, test on `S2`",
            "- `fold3`: train on `S1+S2`, test on `S3`",
            "",
            "Inner split: stratified 80/20 over outer-train **labels**.",
            "",
            "See `loso_manifest.json` and `mixed_sessions_manifest.json`.",
            "",
        ]
    else:
        subject_data = {k: _load_subject_tensor(k) for k in SUBJECTS}
        manifest = {
            "split_rule": "Outer LOSO by subject (S1/S2/S3), inner stratified 80/20 split on outer train.",
            "subjects": {
                key: {
                    "name": SUBJECTS[key].name,
                    "source_pt": SUBJECTS[key].processed_pt,
                }
                for key in SUBJECTS
            },
        }
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

    _write_folds(out_dir, subject_data, manifest, readme_lines, inner_split_seed=args.inner_seed)


if __name__ == "__main__":
    main()
