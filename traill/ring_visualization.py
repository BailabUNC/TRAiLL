# ring_visualization.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import pi
from cmap import Colormap

from traill.traill_dataset import TRAiLLDataset

def load_data(args):
    # load dataset from saved pt file
    data_path = os.path.join('data', 'processed', f'dataset-{args.person}-{args.test_name}.pt')
    dataset = torch.load(data_path, weights_only=False)
    print(f'Loaded dataset with {len(dataset)} instances.')
    return dataset

def draw_circle(dataset: TRAiLLDataset,
                cmap: Colormap,
                interval: float,
                inner_radius: float = 3.0,
                outer_radius: float = 4.0):
    """
    Plot the average waveform of each channel along a circle.
    
    Args:
      dataset: TRAiLLActionDataset yielding (features, label) pairs;
               features tensor of shape (target_length, num_channels).
      cmap:    a Matplotlib colormap (e.g. plt.cm.viridis).
      interval: float, radial offset per channel to avoid overlap.
    """
    features_list = []
    for features, _ in dataset:
        features_list.append(features.numpy())
    # trigger_indices = [inst['trigger_index'] for inst in dataset.instances]
    all_features = np.stack(features_list, axis=0)  # shape: (num_instances, target_length, num_channels)
    num_instances, target_length, num_channels = all_features.shape

    print(f'Visualizing {num_instances} instances...')
    
    avg_signal = np.mean(all_features, axis=0)  # shape: (target_length, num_channels)
    angles = np.deg2rad(90 - np.linspace(0, 360, target_length, endpoint=False))  # shape: (target_length,)

    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(8, 8))
    for ch in range(num_channels):
        radii = inner_radius + (outer_radius - inner_radius) * (avg_signal[:, ch] / np.max(avg_signal)) + interval * ch
        color = cmap(ch / num_channels)
        ax.plot(angles, radii, marker='o', mfc=color, ls='None')
    
    ax.set_ylim(0, 5)
    ax.set_xticks([])      # no angle labels
    ax.set_yticks([])      # no radius labels
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('person', type=str,
                        help='Name of the participant.')
    parser.add_argument('test_name', type=str,
                        help='Name of the data csv file to process.')
    parser.add_argument('--interval', type=float, default=0.1,
                        help='Radial offset per channel to avoid overlap.')
    args = parser.parse_args()
    dataset = load_data(args)
    draw_circle(dataset, Colormap('viridis'), args.interval)