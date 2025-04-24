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

from traill.traill_dataset import TRAiLLDataset

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
) -> Tuple[LinearDiscriminantAnalysis, np.ndarray]:
    """
    Train LDA and return the projection matrix.
    """
    lda = LinearDiscriminantAnalysis(
        n_components=n_components,
        solver='eigen',
        shrinkage='auto',
    )
    lda.fit(X, y)
    print("lda.coef_ shape:", lda.coef_.shape)
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
    Xs, y, scaler = _build_design_matrix(features, labels, scaler)
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
    W = lda.scalings_  # shape: (C, n_components)
    # apply channel-wise scaling to original data
    all_features_std = (features - scaler.mean_) / scaler.scale_  # shape: (N, T, C)
    reduced = np.tensordot(all_features_std, W, axes=([2], [0]))  # shape: (N, T, n_components)
    print(f'Reduced features shape: {reduced.shape}')
    return reduced, {'scaler': scaler, 'lda': lda}

def average_meta_waveform(reduced: np.ndarray) -> np.ndarray:
    """
    Compute the average waveform of each channel.

    Parameters
    ----------
    reduced : np.ndarray
        Shape (N, T, n_components) meta-channel windows.

    Returns
    -------
    avg_signal : np.ndarray
        Shape (T, n_components) average waveforms.
    """
    return np.mean(reduced, axis=0)  # shape: (T, n_components)

def cv_accuracy(X: np.ndarray, y: np.ndarray, cv: int = 5) -> float:
    """
    Compute the cross-validation accuracy of the LDA model.

    Parameters
    ----------
    X : np.ndarray
        Shape (N*T, n_components) design matrix.
    y : np.ndarray
        Shape (N*T,) integer class labels (0-based).
    n_splits : int, default=5
        Number of splits for cross-validation.

    Returns
    -------
    accuracy : float
        Cross-validation accuracy.
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    accuracies = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        lda = _train_lda(X_train, y_train, n_components=1)
        y_pred = lda.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    return np.mean(accuracies)

def main():
    """Quick test of the channel reduction."""
    parser = argparse.ArgumentParser(description='Channel LDA reduction script.')
    parser.add_argument('person', type=str, help='Name of the participant.')
    parser.add_argument('category', type=str, help='Name of category to process.')
    parser.add_argument('-k', type=int, default=4, help='Number of components to keep.')
    parser.add_argument('--use_cache', action='store_true', help='Use cached scaler and LDA if available.')
    args = parser.parse_args()

    data_path = Path('data') / 'processed' / f'concatenated_dataset-{args.person}-{args.category}.pt'
    dataset = torch.load(data_path, weights_only=False)
    features = dataset['features']  # shape: (N, T, C)
    labels = dataset['labels']  # shape: (N,)
    print(f'Loaded feature with shape {features.shape}.')

    reduced, info = reduce_channels(features, labels, n_components=args.k, data_path=data_path, use_cache=args.use_cache)
    X = reduced.mean(axis=1)  # shape: (N, n_components)
    y = np.array([int(l) for l in labels])
    accuracy = cv_accuracy(X, y, cv=5)
    print(f'Cross-validation accuracy: {accuracy:.2f}')

    print(X.shape)

if __name__ == '__main__':
    main()