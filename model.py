# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TRAiLLClassifier(nn.Module):
    def __init__(self, in_channel=48, num_classes=26, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channel, 64, kernel_size=5, padding='same'),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(128, 128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.MaxPool1d(kernel_size=2),  # halves T
        )
        self.rnn = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            bidirectional=False,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(128 * 2, num_classes)  # 2 GRU layers

    def forward(self, x):           # [B, T, C]
        x = x.permute(0, 2, 1)      # [B, C, T]
        x = self.conv(x)            # [B, 128, T/2]
        x = x.permute(0, 2, 1)      # [B, T/2, 128]  (GRU expects [B, T, C])
        _, hn = self.rnn(x)         # [B, T/2, 128]
        h_last = torch.cat((hn[-2], hn[-1]), dim=1) # [B, 128*2]
        x = self.fc(h_last)
        return x                   # [B, num_classes]