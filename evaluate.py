import argparse, json, os, math
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# ── project imports ────────────────────────────────────────────────────
from traill.traill_dataset import TRAiLLDataset           # noqa: F401  (model relies on it)
from model import TRAiLLClassifier
from utils import set_seed
# ───────────────────────────────────────────────────────────────────────

def get_dataloaders(path, batch_size, val_split, generator=None):
    data = torch.load(path, map_location="cpu", weights_only=False)
    feats, labs = data["features"].float(), data["labels"].long()

    # merge augmentation dimension
    N, A, T, C = feats.shape
    feats = feats.view(N * A, T, C)
    labs  = labs.view(N * A) - 9   # same label offset as train.py

    ds = TensorDataset(feats, labs)
    val_len  = int(math.ceil(len(ds) * val_split))
    train_len = len(ds) - val_len
    _, val_ds = random_split(ds, [train_len, val_len], generator=generator)

    return DataLoader(val_ds, batch_size=batch_size, shuffle=False)

# ----------------------------------------------------------------------
def plot_curves(history_path):
    with open(history_path, "r") as f:
        hist = json.load(f)

    epochs = hist["epoch"]
    plt.figure(figsize=(7,4))
    plt.plot(epochs, hist["train_acc"], label="Train acc")
    plt.plot(epochs, hist["val_acc"],   label="Val acc")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.title("Learning curves")
    plt.legend()
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------
@torch.inference_mode()
def confmat(model, loader, device):
    model.eval()
    all_y, all_pred = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        all_pred.append(logits.argmax(1).cpu())
        all_y.append(y)
    y_true  = torch.cat(all_y).numpy()
    y_pred  = torch.cat(all_pred).numpy()

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Validation confusion matrix")
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = set_seed(42)

    # ----- Data loader identical to train.py split ---------------------
    val_loader = get_dataloaders(args.data_pt, args.batch, args.val_split, g)

    # ----- Re‑instantiate architecture then load weights --------------
    # peek one sample to get channel size
    sample_x, _ = next(iter(val_loader))
    in_ch = sample_x.shape[-1]
    num_classes = int(torch.unique(torch.tensor([y for _,y in val_loader.dataset])).numel())

    model = TRAiLLClassifier(in_channel=in_ch,
                             num_classes=num_classes,
                             dropout=args.dropout).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"✔ Loaded weights from {args.checkpoint}")

    # ----- 1. learning curves -----------------------------------------
    if args.history:
        print("Plotting learning curves ...")
        plot_curves(args.history)
    else:
        print("No --history supplied → skipping curve plot.")

    # ----- 2. confusion matrix ----------------------------------------
    print("Computing confusion matrix ...")
    confmat(model, val_loader, device)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate TRAiLL model")
    p.add_argument("checkpoint", help="Path to best_model.pth")
    p.add_argument("data_pt",    help="Same concatenated dataset used for training")
    p.add_argument("--history",  help="history.json file (to draw learning curves)")
    p.add_argument("--batch",    type=int,   default=128)
    p.add_argument("--val_split",type=float, default=0.2)
    p.add_argument("--dropout",  type=float, default=0.2)
    args = p.parse_args()
    main(args)
