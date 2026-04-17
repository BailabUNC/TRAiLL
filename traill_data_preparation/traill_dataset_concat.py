# traill_dataset_concat.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset

from traill_data_preparation.traill_dataset import TRAiLLDataset

def generate_pattern(person: str, pattern_type: str, group: int = None) -> str:
    """
    Generate regex patterns for matching specific filename formats.
    
    Parameters
    ----------
    person : str
        Name of the participant.
    pattern_type : str
        Type of pattern to generate. Must be one of "letters", "commands", or "augmented".
    
    Returns
    -------
    str
        Compiled regex pattern string
    """
    # Escape any special regex characters in person name
    person_escaped = re.escape(person)
    # Base pattern for both types
    base = f'dataset-{person_escaped}-'

    # If group is specified, append it to the base pattern
    if group is not None:
        group_suffix = f'-group_{group}'
    else:
        group_suffix = ''

    patterns = {
        'letters': base + r'(?P<letter>[A-Za-z])' + group_suffix + r'\.pt$',
        'letters_no_filter': base + r'(?P<letter>[A-Za-z])' + group_suffix + r'-no_filter\.pt$',
        'commands': base + r'(?P<command>open|fist|point|pinch|wave|trigger|grab|thumbs_up|swipe)' + group_suffix + r'\.pt$',
        'fingers': base + r'(?P<finger>index|middle|ring|pinky|thumb)' + group_suffix + r'\.pt$',
        'augmented': base + r'(?P<letter>[A-Za-z])_(?=.*(offset|rotate)).*' + group_suffix + r'\.pt$',
    }
    
    if pattern_type not in patterns:
        raise ValueError(f"Pattern type must be one of {list(patterns.keys())}")
        
    return patterns[pattern_type]

def load_datasets(data_dir: str, pattern: str) -> list:
    """
    Load all datasets from the specified directory that match the given regex pattern.

    Args:
        data_dir (str): Directory containing the dataset files.
        pattern (str): Regex pattern to match dataset filenames.

    Returns:
        list: List of loaded datasets.
    """
    datasets = []
    print(f"Using this pattern: {pattern}")
    for file in Path(data_dir).glob('*.pt'):
        # print(f'Checking file: {file}')
        if re.match(pattern, file.name):
            dataset = torch.load(file, weights_only=False)
            datasets.append(dataset)
            print(f'Loaded dataset from {file} with {len(dataset)} instances.')
    return datasets

def _dataset_to_tensors(dataset_obj) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a dataset object into tensors.

    Supported inputs:
    - dictionary with keys {"features", "labels"}
    - torch Dataset (e.g., TRAiLLDataset)
    """
    if isinstance(dataset_obj, dict):
        if "features" not in dataset_obj or "labels" not in dataset_obj:
            raise ValueError("Dictionary dataset must contain keys 'features' and 'labels'.")
        features = dataset_obj["features"].float()
        labels = dataset_obj["labels"].long()
        if features.ndim != 3 or labels.ndim != 1:
            raise ValueError(
                f"Dictionary dataset has invalid shapes: features={tuple(features.shape)}, labels={tuple(labels.shape)}"
            )
        if features.shape[0] != labels.shape[0]:
            raise ValueError("Dictionary dataset feature/label length mismatch.")
        return features, labels

    if isinstance(dataset_obj, Dataset):
        if len(dataset_obj) == 0:
            raise ValueError("Dataset is empty.")
        first_features, _ = dataset_obj[0]
        T, C = first_features.shape
        X = torch.empty((len(dataset_obj), T, C), dtype=torch.float)
        y = torch.empty((len(dataset_obj),), dtype=torch.long)
        for i, (features, label) in enumerate(dataset_obj):
            X[i] = features
            y[i] = label
        return X, y

    raise TypeError(f"Unsupported dataset type: {type(dataset_obj)}")


def generate_tensor(datasets: list[TRAiLLDataset]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a [N, T, C] feature tensor and a [N] label tensor from the datasets.
    """
    if not datasets:
        raise ValueError("No dataset files matched the given pattern.")

    feature_blocks = []
    label_blocks = []
    for dataset in datasets:
        x, y = _dataset_to_tensors(dataset)
        feature_blocks.append(x)
        label_blocks.append(y)

    X = torch.cat(feature_blocks, dim=0)
    y = torch.cat(label_blocks, dim=0)
    return X, y

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Concatenate TRAiLL datasets.')
    parser.add_argument('person', type=str, help='Name of the participant.')
    parser.add_argument('pattern_name', type=str, help='Name of the pattern to match dataset files.')
    parser.add_argument('--group', type=int, default=None, help='Group number for the dataset.')
    parser.add_argument('--data-dir', type=str, default='data/.processed', help='Directory containing dataset files.')
    args = parser.parse_args()

    pattern = generate_pattern(args.person, args.pattern_name, args.group)
    datasets = load_datasets(args.data_dir, pattern)
    features, labels = generate_tensor(datasets)

    # Save the concatenated dataset
    if args.group is not None:
        output_path = os.path.join(args.data_dir, f'concatenated_dataset-{args.person}-{args.pattern_name}-group_{args.group}.pt')
    else:
        output_path = os.path.join(args.data_dir, f'concatenated_dataset-{args.person}-{args.pattern_name}.pt')
    torch.save({'features': features, 'labels': labels}, output_path)
    print(f'Saved concatenated dataset to {output_path}.')