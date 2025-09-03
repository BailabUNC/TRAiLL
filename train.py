# train.py

import os
import math
import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from tqdm.auto import tqdm
from prettytable import PrettyTable

from traill.traill_dataset import TRAiLLDataset
from model import TRAiLLClassifier2D
from utils import *

def get_dataloaders(path, batch_size, val_split, generator=None, label_offset=9):
    """
    Load tensors from the concatenated .pt file and return train/val dataloaders.
    """
    dataset = torch.load(path, map_location='cpu', weights_only=False)
    features, labels = dataset['features'], dataset['labels']
    print('Input features shape:', dataset['features'].shape)

    # Reshape the augmented dataset to merge the augmentation dimension
    N, A, T, C = features.shape
    features = features.view(N * A, T, C)  # [N * A, T, C]
    labels = labels.view(N * A)            # [N * A]
    print('Reshaped features shape:', features.shape)
    print('Reshaped labels shape:', labels.shape)

    print('Original unique labels:', torch.unique(labels))
    labels -= label_offset  # label offset
    print('Labels after offset:', torch.unique(labels))

    # Remap labels to be contiguous and 0-indexed
    unique_labels = torch.unique(labels)
    label_map = {label.item(): i for i, label in enumerate(unique_labels)}
    
    mapped_labels = torch.zeros_like(labels)
    for original_label, new_label in label_map.items():
        mapped_labels[labels == original_label] = new_label
    
    labels = mapped_labels
    print('Remapped labels:', torch.unique(labels))

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

def evaluate(model: nn.Module, dataloader: DataLoader, device, criterion=None):
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if criterion:
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    val_loss = val_loss / total if total else 0.0
    val_acc = correct / total if total else 0.0
    return val_acc, val_loss

def train(args):
    generator = set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_dl, val_dl, dataset = get_dataloaders(args.data_pt, args.batch, args.val_split, generator, args.label_offset)

    num_classes = int(torch.unique(torch.stack([y for _, y in dataset])).numel())

    model = TRAiLLClassifier2D(
        num_classes=num_classes,
        grid_shape=(6, 8),
        dropout=args.dropout
    ).to(device)
    
    table = PrettyTable(["Module", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
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

            # Update progress bar description
            pbar.set_postfix({
                'loss': f'{run_loss/seen:.4f}',
                'acc': f'{run_correct/seen:.4f}'
            })

        scheduler.step()

        train_loss = run_loss / seen
        train_acc = run_correct / seen
        val_acc, val_loss = evaluate(model, val_dl, device, criterion)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'best_model.pth'))

        print(f'Epoch {epoch:03d}/{args.epochs} | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | '
              f'Best Val Acc: {best_acc:.4f}')
    
    print(f'\nTraining finished - best validation accuracy: {best_acc:.4f}')
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

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
    parser.add_argument('--label-offset', type=int, default=9,
                        help='Offset to apply to labels (default: 9)')
    args = parser.parse_args()

    train(args)