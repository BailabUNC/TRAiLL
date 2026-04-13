import argparse
from pathlib import Path
import sys

import torch


def resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_default_dataset(root: Path) -> Path:
    preferred = root / "data" / ".processed" / "concatenated_dataset-krushna-letters.pt"
    if preferred.exists():
        return preferred

    candidates = sorted((root / "data" / ".processed").glob("concatenated_dataset-*.pt"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "No concatenated dataset found under data/.processed/. "
        "Pass --data-path explicitly."
    )


def print_dict_dataset_info(data: dict) -> None:
    print("Loaded dict-style dataset.")
    keys = sorted(data.keys())
    print(f"Keys: {keys}")

    features = data.get("features")
    labels = data.get("labels")

    if torch.is_tensor(features):
        print(f"features shape: {tuple(features.shape)}, dtype: {features.dtype}")
    else:
        print(f"features type: {type(features)}")

    if torch.is_tensor(labels):
        print(f"labels shape: {tuple(labels.shape)}, dtype: {labels.dtype}")
        unique = torch.unique(labels)
        print(f"unique labels ({unique.numel()}): {unique.tolist()[:30]}")
    else:
        print(f"labels type: {type(labels)}")


def print_object_dataset_info(data: object) -> None:
    print(f"Loaded object-style dataset: {type(data)}")
    if hasattr(data, "__len__"):
        print(f"len(dataset): {len(data)}")

    if hasattr(data, "instances") and len(data.instances) > 0:
        first = data.instances[0]
        print(f"instance keys: {list(first.keys())}")
        if "features" in first:
            print(f"first instance feature shape: {first['features'].shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect TRAiLL dataset .pt files.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to dataset .pt file. Defaults to a concatenated dataset in data/.processed.",
    )
    args = parser.parse_args()

    root = resolve_repo_root()
    # Keep import compatibility with project-local packages.
    sys.path.append(str(root))

    data_path = Path(args.data_path) if args.data_path else resolve_default_dataset(root)
    if not data_path.is_absolute():
        data_path = root / data_path

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {data_path}")

    print(f"Loading: {data_path}")
    data = torch.load(data_path, map_location="cpu", weights_only=False)

    if isinstance(data, dict):
        print_dict_dataset_info(data)
    else:
        print_object_dataset_info(data)


if __name__ == "__main__":
    main()
