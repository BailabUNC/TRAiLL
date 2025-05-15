# traill_dataset_augmentation.py

from __future__ import annotations
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from typing import Tuple
import random

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import *

GRID_H, GRID_W = 6, 8

def vec2array(x: torch.Tensor):
    """[T, C] -> [T, 1, H, W]"""
    return x.view(x.size(0), 1, GRID_H, GRID_W)

def array2vec(x: torch.Tensor):
    """[T, 1, H, W] -> [T, C]"""
    return x.view(x.size(0), -1)

def rand_affine(angle: float, shift: float, upsample: int) -> torch.Tensor:
    """
    Random 2-D rotation (±angle) + translation (±shift) matrix.
    Returned shape = [1, 2, 3] ready for `affine_grid`.
    **Coordinates are normalized to [-1, 1]**.
    """
    # random rotation in degrees
    rot = np.deg2rad(random.uniform(-angle, angle))
    c, s = np.cos(rot), np.sin(rot)

    # translate in high-res pixels
    tx_px = random.uniform(-shift, shift) * upsample
    ty_px = random.uniform(-shift, shift) * upsample

    # normalize to [-1, 1] for affine_grid
    tx = 2 * tx_px / (GRID_W * upsample)  # [-1, 1]
    ty = 2 * ty_px / (GRID_H * upsample)  # [-1, 1]

    theta = np.array(
        [[c, -s, tx],
         [s,  c, ty]],
        dtype=np.float32,
    )  # [2, 3]
    return torch.from_numpy(theta).unsqueeze(0)  # [1, 2, 3]

def augment_frame(
    frame: torch.Tensor,
    theta: torch.Tensor,
    upsample: int,
    noise_std: float | None,
    mirror: bool,
) -> torch.Tensor:
    """
    Single frame augmentation using a provided affine matrix.
    
    Parameters
    ----------
        frame: [1, H, W] tensor of features
        theta: [1, 2, 3] affine transformation matrix
        upsample: upsample factor (int)
        noise_std: std of Gaussian noise to add (float or None)
        mirror: random mirroring (bool)
    
    Returns
    -------
        lo: [1, H, W] tensor of augmented features
    """
    # Upsample
    hi = F.interpolate(frame.unsqueeze(0), scale_factor=upsample,
                       mode='bilinear', align_corners=False)  # [1, 1, H*upsample, W*upsample]
    
    # Random mirror
    if mirror and random.random() < 0.5:
        frame = frame.flip(-1)  # Flip along W axis

    # Apply the provided affine matrix
    grid = F.affine_grid(theta, hi.shape, align_corners=False)
    hi = F.grid_sample(hi, grid, padding_mode='border', align_corners=False)

    # Downsample back
    lo = F.interpolate(hi, size=(GRID_H, GRID_W), mode='area')

    # Add noise if noise_std is not None
    if noise_std is not None:
        lo += torch.randn_like(lo) * noise_std

    return lo.squeeze(0)


def augment_sample(
    sample: torch.Tensor,
    num_aug: int,
    upsample: int,
    shift: float,
    angle: float,
    mirror: bool,
    noise_std: float | None,
) -> torch.Tensor:
    """
    Augment a single sample (1, T, C) tensor using a single affine matrix for all frames.
    
    Parameters
    ----------
        sample: [1, T, C] tensor of features
        num_aug: number of augmentations to generate (int)
        upsample: upsample factor (int)
        shift: max translation (float)
        angle: max rotation (degrees)
        mirror: random mirroring (bool)
        noise_std: std of Gaussian noise to add (float or None)
    
    Returns
    -------
        aug_sample: [num_aug, T, C] tensor of augmented features
    """
    augmented = []
    for _ in range(num_aug):
        # Generate a single affine matrix for the entire sample
        theta = rand_affine(angle, shift, upsample).to(sample.device)

        # Convert sample to [T, 1, H, W] and apply augmentation
        frames = vec2array(sample)  # [T, 1, H, W]
        augmented_frames = [augment_frame(frame, theta, upsample, noise_std, mirror) for frame in frames]
        augmented_sample = array2vec(torch.stack(augmented_frames))  # [T, C]
        augmented.append(augmented_sample)

    return torch.stack(augmented)  # [num_aug, T, C]

def main(args):
    set_seed(args.seed)

    dataset = torch.load(args.data_pt, map_location='cpu', weights_only=False)
    features = dataset['features']  # [N, T, C]
    labels = dataset['labels']      # [N]
    print('Input features shape:', features.shape)
    print('Input labels shape:', labels.shape)

    if features.ndim != 3:
        raise ValueError('Input features must be 3D tensor (N, T, C)')
    
    N, T, C = features.shape
    aug_features = []
    aug_labels = []
    for i in tqdm(range(N), desc='Augmenting samples', unit='sample'):
        batch = augment_sample(
            features[i],
            num_aug=args.num_aug,
            upsample=args.upsample,
            shift=args.shift,
            angle=args.angle,
            mirror=args.mirror,
            noise_std=args.noise_std
        )
        aug_features.append(batch)                          # [num_aug, T, C]        
        aug_labels.append(labels[i].repeat(args.num_aug))   # [num_aug]
    
    out_features = torch.stack(aug_features)  # [N, num_aug, T, C]
    out_labels = torch.stack(aug_labels)      # [N, num_aug]

    print(f'Augmented features shape: {out_features.shape}')
    print(f'Augmented labels shape: {out_labels.shape}')
    
    torch.save({'features': out_features, 'labels': out_labels},
               os.path.join(args.out_dir, f'augmented_dataset_{args.num_aug}.pt'))
    print(f'Saved augmented dataset to {os.path.join(args.out_dir, f'augmented_dataset_{args.num_aug}.pt')}')

    # Visualization of augmented data
    print("Visualizing augmented data...")
    num_instances, num_aug, target_length, num_channels = out_features.shape

    # Select a random class (letter) to visualize
    random_class_idx = random.randint(0, num_instances - 1)
    print(f"Visualizing augmented samples for class index: {out_labels[random_class_idx][0]}")
    print(out_labels.shape)

    fig, axes = plt.subplots(nrows=6, ncols=8, figsize=(16, 12))
    colors = ['red', 'blue', 'green', 'orange', 'purple']  # Colors for the 5 samples

    for channel, ax in enumerate(axes.flat):
        # Plot the un-augmented data in black
        ax.plot(features[random_class_idx, :, channel].numpy(), c='black', alpha=0.8, label='Original')


        for aug_idx in range(min(num_aug, 5)):  # Show up to 5 augmented samples
            ax.plot(out_features[random_class_idx, aug_idx, :, channel].numpy(), c=colors[aug_idx], alpha=0.5)

        ax.set_xlim([0, target_length - 1])
        ax.set_ylim([-3, 3])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Channel {channel + 1}", fontsize=8)

    plt.suptitle(f"Augmented Samples for Class {out_labels[random_class_idx][0]}", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TRAiLL dataset augmentation.')
    parser.add_argument('data_pt', help='Path to saved (features, labels) tensor file')
    parser.add_argument('--out-dir', default='data/.augmented', help='Output directory for augmented dataset.')
    parser.add_argument('--num-aug', type=int, default=1000, help='Number of augmentations to generate.')
    parser.add_argument('--upsample', type=int, default=5, help='Upsample factor.')
    parser.add_argument('--shift', type=float, default=0.8, help='Maximum translation (grid cells).')
    parser.add_argument('--angle', type=float, default=5.0, help='Maximum rotation (degrees).')
    parser.add_argument('--noise-std', type=float, default=0.02, help='Standard deviation of Gaussian noise.')
    parser.add_argument('--no-mirror', action='store_false', dest='mirror', help='Disable random mirroring.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')

    args = parser.parse_args()
    main(args)
