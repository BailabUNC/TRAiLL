# dataset_eval.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse

import torch
from traill.traill_dataset import TRAiLLDataset
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dataset = torch.load('data/processed/concatenated_dataset-krushna-letters.pt', weights_only=False)
    print(f'Loaded dataset with {len(dataset)} instances.')
    print(f'Shape of the feature tensor: {dataset.instances[0]["features"].shape}')