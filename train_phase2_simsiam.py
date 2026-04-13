#!/usr/bin/env python3
# coding: utf-8
"""Phase II (manuscript): SimSiam-style training with frozen Phase-I encoder trunk.

Loads Phase-I CNN weights from ``checkpoints/phase1/``, multi-view sequences from
``data/.augmented/augmented_dataset_letters_group_1_50_no_translation.pt`` (default), and fits
only projection + prediction MLP heads. Saves ``proj_head.pth`` / ``pred_head.pth`` under
``checkpoints/phase2/simsiam/``. CLI overrides: batch size, epochs, LR, data paths. Run from repo root.
"""

import os
import random
import argparse
import multiprocessing
from pathlib import Path
from path_config import (
    ROOT,
    first_existing,
    PHASE1_AUTOENCODER_CKPT,
    PHASE2_SIMSIAM_DIR,
    PHASE2_DATA_DEFAULT,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

# Default paths
PT1_PATH   = first_existing(*PHASE1_AUTOENCODER_CKPT)
PT2_DATA   = first_existing(*PHASE2_DATA_DEFAULT)
SAVE_DIR   = str(ROOT / PHASE2_SIMSIAM_DIR)
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters (overridable via CLI below)
BATCH_SIZE   = 512
EPOCHS       = 100
LR           = 0.002
LATENT       = 64
PROJ_DIM     = 64
WEIGHT_DECAY = 1e-4


class CNNAutoencoderStrided512(nn.Module):
    """Encoder trunk only (matches Phase-I architecture up to enc5)."""

    def __init__(self, in_channels=48, latent_channels=LATENT):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv1d(in_channels, 128, 3, 2, 1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv1d(128,  64, 3, 2, 1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv1d(64,   32, 3, 2, 1), nn.ReLU())
        self.enc4 = nn.Sequential(nn.Conv1d(32,   16, 3, 2, 1), nn.ReLU())
        self.enc5 = nn.Sequential(nn.Conv1d(16, latent_channels, 3, 2, 1), nn.ReLU())

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        return self.enc5(x)


class Phase1Encoder(nn.Module):
    def __init__(self, ae: CNNAutoencoderStrided512):
        super().__init__()
        self.encoder = nn.Sequential(ae.enc1, ae.enc2, ae.enc3, ae.enc4, ae.enc5)

    def forward(self, x):
        return self.encoder(x)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim=LATENT, hidden_dim=128, out_dim=PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x):
        return self.net(x)


class PredictionHead(nn.Module):
    def __init__(self, in_dim=PROJ_DIM, hidden_dim=128, out_dim=PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def augment_1d(x):
    """Two-view augmentation: crop-resize, flip, roll, noise, channel drop, spectral mask."""
    C, L = x.shape
    crop = int(L * random.uniform(0.8, 1.0))
    start = random.randint(0, L-crop)
    x1 = x[:, start:start+crop]
    x1 = F.interpolate(x1.unsqueeze(0), size=L, mode='linear', align_corners=False).squeeze(0)
    if random.random() < 0.5:
        x1 = torch.flip(x1, dims=[1])
    shift = random.randint(-L//10, L//10)
    x1 = torch.roll(x1, shift, dims=1)
    x1 = x1 + torch.randn_like(x1)*0.01
    if random.random() < 0.2:
        drop_ch = random.sample(range(C), k=C//10)
        x1[drop_ch] = 0
    f = random.randint(0, L//5)
    f0 = random.randint(0, L - f)
    x1[:, f0:f0+f] = 0
    return x1.clamp(0, 1)


class SimSiam1DDataset(Dataset):
    def __init__(self, feats):
        self.data = feats

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx]
        return augment_1d(x), augment_1d(x)


def simsiam_loss(p1, z2, p2, z1):
    p1 = F.normalize(p1, dim=1)
    p2 = F.normalize(p2, dim=1)
    z1 = z1.detach()
    z2 = z2.detach()
    return -0.5 * (
        F.cosine_similarity(p1, z2, dim=1).mean() +
        F.cosine_similarity(p2, z1, dim=1).mean()
    )


def main():
    ae = CNNAutoencoderStrided512(in_channels=48, latent_channels=LATENT).to(DEVICE)
    ckpt = torch.load(PT1_PATH, map_location=DEVICE)
    # Drop enc5 weights from Phase-I checkpoint when latent width differs
    filtered = {k: v for k, v in ckpt.items() if not k.startswith('enc5')}
    ae.load_state_dict(filtered, strict=False)
    encoder = Phase1Encoder(ae).to(DEVICE)
    encoder.eval()
    for name, p in encoder.named_parameters():
        if any(name.startswith(f"encoder.{i}") for i in (0, 1, 2)):
            p.requires_grad = False

    proj_head = ProjectionHead().to(DEVICE)
    pred_head = PredictionHead().to(DEVICE)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(proj_head.parameters()) + list(pred_head.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    raw = torch.load(PT2_DATA, map_location='cpu')
    feats = raw['features']
    if feats.dim() == 4:
        B1, B2, T, C = feats.shape
        feats = feats.reshape(B1*B2, T, C)
    feats = feats.permute(0, 2, 1)

    ds = SimSiam1DDataset(feats)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)

    for epoch in range(1, EPOCHS+1):
        total_loss = 0.0
        for x1, x2 in loader:
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
            with torch.no_grad():
                h1 = encoder(x1).mean(dim=2)
                h2 = encoder(x2).mean(dim=2)
            z1 = proj_head(h1)
            z2 = proj_head(h2)
            p1 = pred_head(z1)
            p2 = pred_head(z2)

            loss = simsiam_loss(p1, z2, p2, z1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x1.size(0)

        scheduler.step()
        avg = total_loss / len(ds)
        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:03d}/{EPOCHS:03d} — Loss: {avg:.4f} — LR: {lr_now:.2e}")

    torch.save(proj_head.state_dict(), os.path.join(SAVE_DIR, "proj_head.pth"))
    torch.save(pred_head.state_dict(), os.path.join(SAVE_DIR, "pred_head.pth"))
    print("Phase-II SimSiam training finished; projection and prediction heads saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int,   default=512)
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--lr",         type=float, default=0.002)
    parser.add_argument("--latent",     type=int,   default=64)
    parser.add_argument("--proj-dim",   type=int,   default=64)
    parser.add_argument("--wd",         type=float, default=1e-4)
    parser.add_argument("--pt1-path",   type=str, default=PT1_PATH)
    parser.add_argument("--pt2-data",   type=str, default=PT2_DATA)
    parser.add_argument("--save-dir",   type=str, default=SAVE_DIR)
    args = parser.parse_args()
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    LATENT = args.latent
    PROJ_DIM = args.proj_dim
    WEIGHT_DECAY = args.wd
    PT1_PATH = args.pt1_path
    PT2_DATA = args.pt2_data
    SAVE_DIR = args.save_dir
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

    multiprocessing.freeze_support()
    main()
