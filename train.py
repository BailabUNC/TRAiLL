# train.py

import os
import math
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from tqdm.auto import tqdm

from traill.traill_dataset import TRAiLLDataset
from model import TRAiLLClassifier
from utils import *

def get_dataloaders(path, batch_size, val_split, generator=None):
    """
    Load tensors from the concatenated .pt file and return train/val dataloaders.
    """
    dataset = torch.load(path, map_location='cpu', weights_only=False)
    features, labels = dataset['features'], dataset['labels']
    print('Input features shape:', dataset['features'].shape)

    labels -= 9  # label offset

    # sanity check dtypes
    features = features.float()
    labels = labels.long()
    dataset = TensorDataset(features, labels)

    val_len = int(math.ceil(len(dataset) * val_split))
    train_len = len(dataset) - val_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len], generator=generator)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

    print(f'Train set: {len(train_dataset)} samples')
    print(f'Validation set: {len(val_dataset)} samples')

    return train_dl, val_dl, dataset

def evaluate(model: nn.Module, dataloader: DataLoader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return correct / total if total else 0.0

def train(args):
    generator = set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_dl, val_dl, dataset = get_dataloaders(args.data_pt, args.batch, args.val_split, generator)

    sample_x, _ = dataset[0]
    num_classes = int(torch.unique(torch.stack([y for _, y in dataset])).numel())

    model = TRAiLLClassifier(in_channel=sample_x.shape[-1],
                             num_classes=num_classes,
                             dropout=args.dropout).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0
        run_correct = 0
        seen = 0

        # Create progress bar
        pbar = tqdm(train_dl, desc=f'Epoch {epoch}/{args.epochs}', leave=False)
        
        for batch in pbar:
            x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            seen += x.size(0)
            run_loss += loss.item() * x.size(0)
            run_correct += (logits.argmax(1) == y).sum().item()

            # Update progress bar description instead of using tqdm.write
            pbar.set_postfix({
                'loss': f'{run_loss/seen:.4f}',
                'acc': f'{run_correct/seen:.4f}'
            })

        scheduler.step()

        train_loss = run_loss / seen
        train_acc  = run_correct / seen
        val_acc    = evaluate(model, val_dl, device)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'best_model.pth'))

        print(f'Epoch {epoch:03d}/{args.epochs} | ',
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | ',
              f'Val Acc: {val_acc:.4f} | Best Val Acc: {best_acc:.4f}')
    
    print(f'\nTraining finished - best validation accuracy: {best_acc:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TRAiLL CNN-GRU classifier')
    parser.add_argument('data_pt',      help='Path to saved (features, labels) tensor file')
    parser.add_argument('--out_dir',    default='checkpoints')
    parser.add_argument('--epochs',     type=int,   default=20)
    parser.add_argument('--batch',      type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=2e-3)
    parser.add_argument('--dropout',    type=float, default=0.2)
    parser.add_argument('--val_split',  type=float, default=0.2,
                        help='Fraction for validation set')
    parser.add_argument('--tlen',       type=int,   default=128,
                        help='Resampled time-length')
    args = parser.parse_args()

    train(args)