# channel_reduction.py

from __future__ import annotations

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pickle
import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from traill.traill_dataset import TRAiLLDataset

def _collapse_by_labels(
    features: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collapse the dataset by labels.

    Parameters
    ----------
    features : torch.Tensor
        Shape (N, T, C) raw window data.
    labels : torch.Tensor
        Shape (N,) integer class labels (0-based).

    Returns
    -------
    features : np.ndarray
        Shape (N, T, C) raw window data.
    labels : np.ndarray
        Shape (N,) integer class labels (0-based).
    """
    features_np = features.numpy()
    labels_np = labels.numpy()

    uniq = np.unique(labels_np)
    mean_windows = []
    for u in uniq:
        mean_windows.append(features_np[labels_np == u].mean(axis=0))
    return np.stack(mean_windows, axis=0), uniq

def _build_design_matrix(
    features: np.ndarray,
    labels: np.ndarray,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Flatten (N, T, C) to (N*T, C) and z-score by channel.

    Parameters
    ----------
    features : np.ndarray
        Shape (N, T, C) raw window data.
    labels : np.ndarray
        Shape (N,) integer class labels (0-based).
    scaler : sklearn.preprocessing.StandardScaler | None
        Existing scaler to reuse; if *None* a new one is fitted.

    Returns
    -------
    Xs : np.ndarray
        Z-scored design matrix, shape (N*T, C).
    y : np.ndarray
        Repeated labels, length N*T.
    scaler : StandardScaler
        The fitted scaler.
    """
    N, T, C = features.shape
    X = features.reshape(N * T, C)  # shape: (N*T, C)
    y = np.repeat(labels, T)  # shape: (N*T,)
    if scaler is None:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)  # shape: (N*T, C)
    else:
        Xs = scaler.transform(X)  # shape: (N*T, C)
    return Xs, y, scaler

def _train_lda(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int,
) -> LinearDiscriminantAnalysis:
    lda = LinearDiscriminantAnalysis(
        n_components=n_components,
        solver='eigen',
        shrinkage=0.5,
    )
    lda.fit(X, y)
    print("lda.scalings_ shape:", lda.scalings_.shape)
    return lda

def _cache_paths(data_path: Path, suffix: str) -> Path:
    """
    Return path for a cache file stting next to `data_path`.
    """
    return data_path.with_suffix(f'.{suffix}.pkl')

def reduce_channels(
    features: torch.Tensor,
    labels: torch.Tensor,
    n_components: int = 4,
    data_path: Optional[Path] = None,
    use_cache: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Project dataset onto *n_components* supervised meta channels.

    Parameters
    ----------
    features: torch.Tensor
        Shape (N, T, C) raw window data.
    labels: torch.Tensor
        Shape (N,) integer class labels (0-based).
    n_components: int, default=4
        Number of components to keep.
    data_path: Path | None, default=None
        Path to the dataset file. If provided, the cache will be saved next to it.
        If *None*, the cache will not be saved.
    use_cache: bool, default=True
        Whether to use the cache if it exists. If *False*, the cache will be ignored
        and a new one will be created.
    
    Returns
    -------
    reduced: np.ndarray
        Reduced dataset, shape (N, T, n_components) meta-channel windows.
    info: dict
        Contains the scaler ('scaler') and the LDA model ('lda').
    """

    # Cache handling
    scaler_path = None
    lda_path = None

    scaler = None
    lda = None

    if use_cache and data_path is not None:
        scaler_path = _cache_paths(data_path, 'scaler')
        lda_path = _cache_paths(data_path, 'lda')
        if scaler_path.exists() and lda_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            with open(lda_path, 'rb') as f:
                lda = pickle.load(f)
            print(f'Loaded cached scaler from {scaler_path}')
            print(f'Loaded cached LDA from {lda_path}')

    # Build design matrix and train LDA if not cached
    features_mean, label_ids = _collapse_by_labels(features, labels)    # shape: (L, T, C)
    L, T, C = features_mean.shape

    Xs, y, scaler = _build_design_matrix(
        features_mean,  # shape: (L, T, C)
        label_ids,      # shape: (L,)
        scaler,
    )

    if lda is None:
        lda = _train_lda(Xs, y, n_components)
        if scaler_path is not None and lda_path is not None:
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            with open(lda_path, 'wb') as f:
                pickle.dump(lda, f)
            print(f'Saved scaler to {scaler_path}')
            print(f'Saved LDA to {lda_path}')
    
    # Project each full window - keep time dimension
    W = lda.scalings_[:, :n_components]                         # (C, K)
    # apply channel-wise scaling to original data
    features_std = (features_mean - scaler.mean_) / scaler.scale_    # shape: (L, T, C)
    reduced = np.tensordot(features_std, W, axes=([2], [0]))    # shape: (L, T, K)
    
    print(f'Reduced features shape: {reduced.shape}')
    return reduced, {
        'scaler':   scaler,
        'lda':      lda,
        'labels':   label_ids      # optional mapping, e.g. [0,1,…,25]
    }

def show_component_heatmap(
    lda,
    component_idx: int,
    grid_shape: tuple[int, int] = (6, 8),      # rows × cols  (6 × 8 = 48 channels)
    channel_names: list[str] | None = None,    # optional labels for each chan
    cmap: str = "coolwarm",
):
    """
    Display a 6x8 heat-map of per-channel weights for a single LDA component.

    Parameters
    ----------
    lda : sklearn.discriminant_analysis.LinearDiscriminantAnalysis
        A *trained* LDA model; must contain `lda.scalings_` of shape (C, K).
    component_idx : int
        Which of the K meta-components to visualise (0-based).
    grid_shape : (int, int), default (6, 8)
        Layout of the electrodes / channels on the heat-map.
        The product must equal the number of input channels *C*.
    channel_names : list[str] | None
        Optional per-channel names (length == C).  If supplied they appear
        as x-axis tick labels; otherwise the channels are just numbered.
    cmap : str, default 'coolwarm'
        Matplotlib colour-map to use.
    """
    """
    Heatmap of the LDA channel weights.

    Parameters
    ----------
    lda : LinearDiscriminantAnalysis
        Fitted LDA model. Must have `lda.scalings_` attribute.
    chan_labels : list of str | None, default=None
        Channel labels. If *None*, the channels are numbered from 0 to C-1.
    """
    W = lda.scalings_  # shape: (C, K)
    C, K = W.shape

    weights = W[:, component_idx].reshape(grid_shape)  # shape: (rows, cols)


    fig, ax = plt.subplots(figsize=(grid_shape[1] * 1.2, grid_shape[0]))
    im = ax.imshow(weights, cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label='weight')
    # tick labels
    if channel_names is None:
        channel_names = [str(i) for i in range(C)]

    # ax.set_xticks(np.arange(grid_shape[1]))
    # ax.set_xticklabels(channel_names, rotation=90)
    # ax.set_yticks(np.arange(grid_shape[0]))
    # ax.set_yticklabels(np.arange(grid_shape[0]))

    plt.tight_layout()
    plt.show()

def show_grouping_scatter(
    reduced: np.ndarray,    # (L, T, K)
    labels: np.ndarray,     # (L,)
    per_time: bool = False, # False: one dot per class
):
    """
    Quick visual check of class clustering in the reduced LDA space.
    
    Parameters
    ----------
    reduced : shape (L, T, K)
        L = number of classes (labels) after collapsing windows
        T = time-points per window
        K = number of LDA components
    labels : the class indices in the same order as `reduced`.
    per_time :
        False -> one point per class = mean over the time dimension
        True  -> T points per class = every time-sample is plotted
    """
    # Reshape into (samples, K)
    if per_time:
        # LxT samples, keep time dimension
        X = reduced.reshape(-1, reduced.shape[-1])  # shape: (L*T, K)
        y = np.repeat(labels, reduced.shape[1])     # shape: (L*T,)
    else:
        # one mean vector per class
        X = reduced.mean(axis=1)                    # shape: (L, K)
        y = labels

    coords = X[:, :2]  # take first two components for 2D plot

    # plot the points
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.get_cmap('cool', len(np.unique(y)))
    
    for cls in np.unique(y):
        pts = coords[y == cls]
        ax.scatter(pts[:, 0], pts[:, 1],
                   s=25 if per_time else 200,
                   alpha=0.6 if per_time else 0.8,
                   label=f'{cls}',
                   color=cmap(cls))
        
        # 1-sigma ellipse
        if not per_time and pts.shape[0] >= 2:
            print(f'Class {cls} has {pts.shape[0]} points.')
            cov = np.cov(pts.T)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vx, vy = vecs[:, order]
            theta = np.degrees(np.arctan2(vy, vx))
            width, height = 2 * np.sqrt(vals[order])  # 1-sigma
            ell = Ellipse(pts.mean(axis=0), width, height,
                          angle=theta, edgecolor=cmap(cls),
                          facecolor='none', lw=1.2, alpha=0.9)
            ax.add_patch(ell)

        if not per_time:
            if cls >= 9:
                label_char = chr(ord('A') + (cls - 9))
                ax.text(pts[0, 0], pts[0, 1], label_char,
                        ha='center', va='center')
            else:
                ax.text(pts[0, 0], pts[0, 1], str(cls),
                        ha='center', va='center')
    
    # ax.set_xlim(-3, 1)
    # ax.set_ylim(-0.05, 0.1)
    ax.set_xlabel('LDA component 1')
    ax.set_ylabel('LDA component 2')
    # ax.legend(title='class', bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Channel LDA reduction script.')
    parser.add_argument('person', type=str, help='Name of the participant.')
    parser.add_argument('category', type=str, help='Name of category to process.')
    parser.add_argument('-k', type=int, default=4, help='Number of components to keep.')
    parser.add_argument('-c', '--show-component', type=int, default=0, help='LDA component to visualize.')
    parser.add_argument('--viz', choices=['scatter', 'heatmap'], default='scatter', help='Visualization type.')
    parser.add_argument('--per-time', action='store_true', help='Show individual time points in scatter plot instead of class means.')
    parser.add_argument('--use-cache', action='store_true', help='Use cached scaler and LDA if available.')
    args = parser.parse_args()

    data_path = Path('data') / 'processed' / f'concatenated_dataset-{args.person}-{args.category}.pt'
    dataset = torch.load(data_path, weights_only=False)
    features = dataset['features']  # shape: (N, T, C)
    labels = dataset['labels']      # shape: (N,)
    print(f'Loaded feature with shape {features.shape}, labels with shape {labels.shape}.')

    reduced, info = reduce_channels(features, labels, n_components=args.k, data_path=data_path, use_cache=args.use_cache)
    lda = info['lda']
    labels = info['labels']

    if args.viz == 'scatter':
        show_grouping_scatter(reduced, labels, per_time=args.per_time)
    else:
       show_component_heatmap(lda, component_idx=args.show_component)

    # Save the reduced dataset
    output_path = data_path.with_suffix(f'.reduced-{args.k}.pt')
    torch.save({
        'features': reduced,
        'labels': info['labels'],
        'scaler': info['scaler'],
        'lda': info['lda'],
    }, output_path)

if __name__ == '__main__':
    main()