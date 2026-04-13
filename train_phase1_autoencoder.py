"""Phase I (manuscript): train the 1D-CNN denoising autoencoder on stacked feature windows.

Reads precomputed tensors from ``data/.augmented/augmented_dataset_letters_group_1_10_std0.15.pt`` (default).
Writes weights to ``checkpoints/phase1/augmented_phase_1.pth``. Supports ``--dry-run`` and
optional inline recon previews during training. Run from repo root.
"""
import os
import random
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from path_config import ROOT, first_existing, PHASE1_DATA_DEFAULT

# Paths and training hyperparameters
PT_PATH      = first_existing(
    *PHASE1_DATA_DEFAULT,
)
SAVE_MODEL   = str(ROOT / "checkpoints" / "phase1" / "augmented_phase_1.pth")
BATCH_SIZE   = 32
LR           = 1e-3
EPOCHS       = 60
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES  = 3
VIS_CHANNELS = list(range(48))
train_losses = []
train_accs = []


class FeatureDataset(Dataset):
    """Windows shaped (N, 48, 512): channels x time."""

    def __init__(self, features_tensor):
        self.features = features_tensor

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, idx):
        x = self.features[idx]
        return x, x


class CNNAutoencoderStrided512(nn.Module):
    def __init__(self, in_channels=48, latent_channels=32):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv1d(in_channels, 128, 3, 2, 1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv1d(128, 64, 3, 2, 1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv1d(64, 32, 3, 2, 1), nn.ReLU())
        self.enc4 = nn.Sequential(nn.Conv1d(32, 16, 3, 2, 1), nn.ReLU())
        self.enc5 = nn.Sequential(nn.Conv1d(16, latent_channels, 3, 2, 1), nn.ReLU())

        # Decoder (skip connections)
        self.dec5 = nn.Sequential(
            nn.ConvTranspose1d(latent_channels, 16, 3, 2, 1, output_padding=1),
            nn.ReLU()
        )
        self.dec4 = nn.Sequential(
            nn.Conv1d(16+16, 32, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, 32, 3, 2, 1, output_padding=1), nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.Conv1d(32+32, 64, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(64, 64, 3, 2, 1, output_padding=1), nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.Conv1d(64+64, 128, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(128, 128, 3, 2, 1, output_padding=1), nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.Conv1d(128+128, 128, 3, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(128, in_channels, 3, 2, 1, output_padding=1),
            nn.Sigmoid()  # targets are min–max normalized to [0, 1]
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        z  = self.enc5(e4)
        d5 = self.dec5(z)
        d4 = self.dec4(torch.cat([d5, e4], dim=1))
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        return self.dec1(torch.cat([d2, e1], dim=1))


def main(dry_run: bool = False):
    Path(SAVE_MODEL).parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f"[dry-run] PT_PATH={PT_PATH}")
        print(f"[dry-run] SAVE_MODEL={SAVE_MODEL}")
        print("[dry-run] config loaded successfully.")
        return
    ckpt = torch.load(PT_PATH, map_location="cpu")
    feats = ckpt['features'].float()
    N, S, L, C = feats.shape
    feats = feats.reshape(N*S, L, C)
    feats = feats.permute(0, 2, 1)

    mins = feats.min(dim=2, keepdim=True).values
    maxs = feats.max(dim=2, keepdim=True).values
    feats = (feats - mins) / (maxs - mins + 1e-8)

    dataset = FeatureDataset(feats)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model     = CNNAutoencoderStrided512(in_channels=48, latent_channels=32).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.SmoothL1Loss()

    model.train()
    for epoch in range(1, EPOCHS+1):
        total_loss = 0.0
        total_acc = 0.0
        total_samples = 0
        for x, _ in loader:
            x = x.to(DEVICE)
            noise = torch.randn_like(x) * 0.2
            x_noisy = x + noise
            x_r = model(x_noisy)
            loss = criterion(x_r, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

            acc = ((x_r - x).abs() < 0.1).float().mean().item()
            total_acc += acc * x.size(0)
            total_samples += x.size(0)

        avg_loss = total_loss / len(dataset)
        avg_acc = total_acc / total_samples

        train_losses.append(avg_loss)
        train_accs.append(avg_acc)
        print(f"Epoch {epoch:02d}/{EPOCHS} — Smooth Loss: {avg_loss:.6f} — Acc: {avg_acc:.4f}")

    torch.save(model.state_dict(), SAVE_MODEL)
    np.save("train_losses.npy", np.array(train_losses))
    np.save("train_accs.npy", np.array(train_accs))
    print(f"\nModel saved to `{SAVE_MODEL}`")
    print("Loss/Acc curves saved as 'train_losses.npy', 'train_accs.npy'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
