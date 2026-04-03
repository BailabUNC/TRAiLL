# ring_visualization.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from cmap import Colormap

def load_data(args):
    # load reduced dataset from saved pt file, shape (L, T, K)
    data_path = os.path.join('data', 'processed', f'concatenated_dataset-{args.person}-{args.test_name}.reduced-{args.k}.pt')
    data = torch.load(data_path, weights_only=False)
    print(f'Loaded dataset with shape {data["features"].shape}')
    return data

def draw_circle(features: np.ndarray,
                cmap: Colormap,
                interval: float,
                selected_class: int = 0,
                inner_radius: float = 3.0,
                outer_radius: float = 4.0):
    """
    Plot the average waveform of each reduced channel along a circle.
    
    Args:
      features: numpy array of shape (L, T, K=4)
      cmap: a Matplotlib colormap (e.g. plt.cm.viridis)
      interval: float, radial offset per channel to avoid overlap
    """
    L, T, K = features.shape
    print(f'Visualizing {L} classes with {K} reduced components...')
    
    selected_signal = features[selected_class, :, :]  # shape: (T, K)
    selected_signal = np.vstack([selected_signal, selected_signal[0:1, :]])
    
    angles = np.deg2rad(90 - np.linspace(0, 360, T+1, endpoint=True))

    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=(8, 8))
    for component in range(K):
        width = outer_radius - inner_radius
        normalized_signal = selected_signal[:, component] / np.max(selected_signal)
        radii = inner_radius + width * normalized_signal + interval * component
        color = cmap(component / K)
        ax.plot(angles, radii,
                color='gray', lw=1, alpha=0.7,
                marker='o', mfc=color, mec='None',
                label=f'Component {component+1}')
    
    ax.set_ylim(0, 5)
    ax.set_xticks([])      # no angle labels
    ax.set_yticks([])      # no radius labels
    ax.legend(bbox_to_anchor=(1.2, 1.0))
    plt.tight_layout()
    plt.show()

def draw_circles(features: np.ndarray,
                cmap: Colormap,
                interval: float,
                inner_radius: float = 3.0,
                outer_radius: float = 4.0):
    """
    Plot all classes in a grid of polar plots.
    
    Args:
      features: numpy array of shape (L, T, K=4)
      cmap: a Matplotlib colormap (e.g. plt.cm.viridis)
      interval: float, radial offset per channel to avoid overlap
    """
    L, T, K = features.shape
    print(f'Visualizing all {L} classes with {K} reduced components...')
    
    # Create a grid layout
    ncols = min(5, L)  # Max 5 plots per row
    nrows = (L - 1) // ncols + 1
    fig = plt.figure(figsize=(4*ncols, 4*nrows))
    
    for idx in range(L):
        selected_signal = features[idx]  # shape: (T, K)
        selected_signal = np.vstack([selected_signal, selected_signal[0:1, :]])
        angles = np.deg2rad(90 - np.linspace(0, 360, T+1, endpoint=True))
        
        ax = plt.subplot(nrows, ncols, idx+1, projection='polar')
        for component in range(K):
            width = outer_radius - inner_radius
            normalized_signal = selected_signal[:, component] / np.max(selected_signal)
            radii = inner_radius + width * normalized_signal + interval * component
            color = cmap(component / K)
            ax.plot(angles, radii,
                    color='gray', lw=1, alpha=0.7,
                    marker='o', mfc=color, mec='None',
                    label=f'Component {component+1}')
        
        ax.set_ylim(0, 5)
        ax.set_xticks([])
        ax.set_yticks([])
        # if idx == 0:  # Only show legend for first plot
        #     ax.legend(bbox_to_anchor=(1.2, 1.0))
        ax.set_title(f'Class {idx}')
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('person', type=str,
                        help='Name of the participant.')
    parser.add_argument('test_name', type=str,
                        help='Category to visualize (letters/commands).')
    parser.add_argument('-k', type=int, default=4,
                        help='Number of components in the reduced dataset.')
    parser.add_argument('--interval', type=float, default=0.1,
                        help='Radial offset per channel to avoid overlap.')
    parser.add_argument('-c', '--selected-class', type=int, default=0,
                        help='Class to visualize (0-indexed).')
    parser.add_argument('--show-all', action='store_true',
                        help='Show all classes in a grid layout.')
    args = parser.parse_args()
    
    data = load_data(args)
    if args.show_all:
        draw_circles(data['features'], Colormap('viridis'), args.interval,
                    outer_radius=4.0, inner_radius=3.0)
    else:
        draw_circle(data['features'], Colormap('viridis'), args.interval,
                    args.selected_class, outer_radius=4.0, inner_radius=3.0)