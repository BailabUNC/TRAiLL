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
from cmap import Colormap

from utils import *

GRID_H, GRID_W = 6, 8

def vec2array(x: torch.Tensor):
    """[T, C] -> [T, 1, H, W]"""
    return x.view(x.size(0), 1, GRID_H, GRID_W)

def array2vec(x: torch.Tensor):
    """[T, 1, H, W] -> [T, C]"""
    return x.view(x.size(0), -1)

def rand_affine(angle: float, shift: float, upsample: int) -> Tuple[torch.Tensor, float, float, float]:
    """
    Random 2-D rotation (±angle) + translation (±shift) matrix.
    Returned shape = [1, 2, 3] ready for `affine_grid`.
    **Coordinates are normalized to [-1, 1]**.
    """
    # random rotation in degrees
    rot_deg = random.uniform(-angle, angle)
    rot = np.deg2rad(rot_deg)
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
    return torch.from_numpy(theta).unsqueeze(0), rot_deg, tx, ty


def augment_sample(
    sample: torch.Tensor,
    num_aug: int,
    upsample: int,
    shift: float,
    angle: float,
    mirror: bool,
    noise_std: float | None,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
        aug_params: [num_aug, 4] tensor of augmentation parameters
    """
    augmented = []
    aug_params = []
    T = sample.shape[0]
    for _ in range(num_aug):
        # Generate a single affine matrix for the entire sample
        theta, rot_deg, tx, ty = rand_affine(angle, shift, upsample)
        theta = theta.to(sample.device)
        
        mirrored = mirror and random.random() < 0.5

        # Convert sample to [T, 1, H, W] and apply augmentation
        frames = vec2array(sample)  # [T, 1, H, W]
        
        if mirrored:
            frames = frames.flip(-1)

        # Upsample
        hi = F.interpolate(frames, scale_factor=upsample, mode='bilinear', align_corners=False)

        # Apply affine transformation to all frames
        theta_batch = theta.repeat(T, 1, 1) # [T, 2, 3]
        grid = F.affine_grid(theta_batch, hi.shape, align_corners=False)
        hi = F.grid_sample(hi, grid, padding_mode='border', align_corners=False)

        # Downsample
        lo = F.interpolate(hi, size=(GRID_H, GRID_W), mode='area')

        # Add noise
        if noise_std is not None:
            lo += torch.randn_like(lo) * noise_std

        augmented_sample = array2vec(lo) # [T, C]
        augmented.append(augmented_sample)
        aug_params.append([tx, ty, rot_deg, float(mirrored)])

    return torch.stack(augmented), torch.tensor(aug_params, dtype=torch.float32)

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
    aug_params_all = []
    for i in tqdm(range(N), desc='Augmenting samples', unit='sample'):
        batch, batch_params = augment_sample(
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
        aug_params_all.append(batch_params)
    
    out_features = torch.stack(aug_features)  # [N, num_aug, T, C]
    out_labels = torch.stack(aug_labels)      # [N, num_aug]
    out_params = torch.stack(aug_params_all)  # [N, num_aug, 4]

    print(f'Augmented features shape: {out_features.shape}')
    print(f'Augmented labels shape: {out_labels.shape}')
    print(f'Augmented params shape: {out_params.shape}')
    
    if args.test_type is not None:
        file_basename = f'augmented_dataset_{args.test_type}_{args.num_aug}'
    else:
        file_basename = f'augmented_dataset_{args.num_aug}'
    
    save_path = os.path.join(args.out_dir, f'{file_basename}.pt')

    if not args.no_save:
        os.makedirs(args.out_dir, exist_ok=True)
        print(f'Saved augmented dataset to {save_path}')
        torch.save({
            'features': out_features,
            'labels': out_labels,
            'params': out_params
        }, save_path)
    else:
        print('Skipping save (--no-save specified).')

    # Visualization of augmented data as heatmaps
    print('Visualizing augmented data (heatmaps)...')
    num_instances, num_aug, target_length, num_channels = out_features.shape

    # Select a random sample index to visualize
    sample_idx = random.randint(0, num_instances - 1)
    sample_label = out_labels[sample_idx][0].item()
    print(f'Visualizing sample index: {sample_idx} (label={sample_label})')

    # Determine most significant timestamp (max sum of absolute channel values in original sample)
    orig_sample = features[sample_idx]  # [T, C]
    significance = orig_sample.abs().sum(dim=1)  # [T]
    ts_idx = int(significance.argmax().item())
    print(f'Most significant timestamp: {ts_idx}')

    # Prepare frames: original + first 3 augmented (or fewer if num_aug < 3)
    num_show_aug = min(3, num_aug)
    frames = []
    titles = ['Original']
    frames.append(orig_sample[ts_idx].view(GRID_H, GRID_W))
    for k in range(num_show_aug):
        frames.append(out_features[sample_idx, k, ts_idx].view(GRID_H, GRID_W))
        titles.append(f'Aug {k+1}')

    # Compute shared color scale
    stacked = torch.stack(frames)
    vmin = float(stacked.min().item())
    vmax = float(stacked.max().item())
    print(f'Color scale: vmin={vmin}, vmax={vmax}')

    fig, axes = plt.subplots(1, 1 + num_show_aug, figsize=(3.2 * (1 + num_show_aug), 4))
    if (1 + num_show_aug) == 1:
        axes = [axes]

    for ax, frame, title in zip(axes, frames, titles):
        im = ax.imshow(frame.numpy(), cmap=Colormap('cmocean:balance').to_mpl(), vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TRAiLL dataset augmentation.')
    parser.add_argument('data_pt', help='Path to saved (features, labels) tensor file')
    parser.add_argument('--test-type', type=str, default=None, help='Test type (e.g., "letter")')
    parser.add_argument('--out-dir', default='data/.augmented', help='Output directory for augmented dataset.')
    parser.add_argument('--num-aug', type=int, default=1000, help='Number of augmentations to generate.')
    parser.add_argument('--upsample', type=int, default=5, help='Upsample factor.')
    parser.add_argument('--shift', type=float, default=0.8, help='Maximum translation (grid cells).')
    parser.add_argument('--angle', type=float, default=5.0, help='Maximum rotation (degrees).')
    parser.add_argument('--noise-std', type=float, default=0.02, help='Standard deviation of Gaussian noise.')
    parser.add_argument('--no-mirror', action='store_false', dest='mirror', help='Disable random mirroring.')
    parser.add_argument('--no-save', action='store_true', help='Skip saving augmented dataset; only visualize.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')

    args = parser.parse_args()
    main(args)
