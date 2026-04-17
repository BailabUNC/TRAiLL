from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

PathLike = Union[str, Path]


def _load_group_pt(path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or "features" not in data or "labels" not in data:
        raise ValueError(f"{path} must be a dict with keys 'features' and 'labels'.")
    x = data["features"].float()
    y = data["labels"].long()
    if x.ndim != 3 or y.ndim != 1:
        raise ValueError(f"{path} has invalid shape: features={tuple(x.shape)} labels={tuple(y.shape)}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"{path} feature/label length mismatch.")
    return x, y


def _save_split(path: Path, features: torch.Tensor, labels: torch.Tensor, session_id: int, origin_group_ids: torch.Tensor) -> None:
    payload = {
        "features": features,
        "labels": labels,
        "subject_ids": torch.full((features.shape[0],), session_id, dtype=torch.long),
        "instance_ids": torch.arange(features.shape[0], dtype=torch.long),
        "origin_group_ids": origin_group_ids.long(),
    }
    torch.save(payload, path)


def _hist_counts(values: torch.Tensor) -> Dict[str, int]:
    uniq, cnt = torch.unique(values, return_counts=True)
    return {str(int(u)): int(c) for u, c in zip(uniq.tolist(), cnt.tolist())}


def mix_and_split_sessions(
    input_dir: PathLike,
    glob_pattern: str,
    output_dir: PathLike,
    seed: int,
) -> Dict[str, Any]:
    """
    Concatenate daniela group_1/2/3 ``.pt`` files, shuffle, split into 3 pseudo-sessions,
    write ``mixed_session{1,2,3}.pt`` and ``mixed_sessions_manifest.json`` under ``output_dir``.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    group_files = sorted(input_dir.glob(glob_pattern))
    if len(group_files) != 3:
        raise ValueError(
            f"Expected exactly 3 input files under {input_dir} matching {glob_pattern}, got {len(group_files)}."
        )

    x_blocks: List[torch.Tensor] = []
    y_blocks: List[torch.Tensor] = []
    g_blocks: List[torch.Tensor] = []
    input_meta = []

    for group_idx, path in enumerate(group_files, start=1):
        x, y = _load_group_pt(path)
        x_blocks.append(x)
        y_blocks.append(y)
        g_blocks.append(torch.full((x.shape[0],), group_idx, dtype=torch.long))
        input_meta.append(
            {
                "file": str(path.resolve()),
                "num_samples": int(x.shape[0]),
                "label_hist": _hist_counts(y),
            }
        )

    X = torch.cat(x_blocks, dim=0)
    y = torch.cat(y_blocks, dim=0)
    origin_gid = torch.cat(g_blocks, dim=0)

    n = X.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    splits = np.array_split(perm, 3)

    manifest: Dict[str, Any] = {
        "description": "Mixed group_1/2/3 then random split into 3 pseudo sessions.",
        "seed": seed,
        "inputs": input_meta,
        "sessions": {},
    }

    for sess_idx, idx_np in enumerate(splits, start=1):
        idx = torch.as_tensor(idx_np, dtype=torch.long)
        x_s = X.index_select(0, idx)
        y_s = y.index_select(0, idx)
        g_s = origin_gid.index_select(0, idx)
        out_path = output_dir / f"mixed_session{sess_idx}.pt"
        _save_split(out_path, x_s, y_s, sess_idx, g_s)
        manifest["sessions"][f"session{sess_idx}"] = {
            "file": str(out_path.resolve()),
            "num_samples": int(x_s.shape[0]),
            "label_hist": _hist_counts(y_s),
            "origin_group_hist": _hist_counts(g_s),
        }

    manifest_path = output_dir / "mixed_sessions_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mix daniela group_1/2/3 datasets and random-split into 3 pseudo sessions for LOSO."
    )
    parser.add_argument("--input-dir", type=str, default="data/.processed")
    parser.add_argument("--glob", type=str, default="concatenated_dataset-daniela-letters-group_*.pt")
    parser.add_argument("--output-dir", type=str, default="data/loso")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mix_and_split_sessions(args.input_dir, args.glob, args.output_dir, args.seed)
    output_dir = Path(args.output_dir)
    print(f"Wrote mixed session files to {output_dir}")
    print(f"Wrote manifest to {output_dir / 'mixed_sessions_manifest.json'}")


if __name__ == "__main__":
    main()
