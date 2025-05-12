import torch
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import TRAiLLClassifier2D

def evaluate_model(model, dataloader, device):
    """
    Evaluate the model on the given dataloader and compute metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)  # Get predicted class
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return accuracy, precision, recall, f1

def main(args):
    # Load the dataset
    data = torch.load(args.data_path, map_location='cpu')
    features, labels = data['features'], data['labels']  # [N, T, C], [N]
    dataset = torch.utils.data.TensorDataset(features, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TRAiLLClassifier2D(num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print(f"Loaded model checkpoint from {args.checkpoint}")

    # Evaluate the model
    accuracy, precision, recall, f1 = evaluate_model(model, dataloader, device)

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate TRAiLLClassifier2D on a dataset.")
    parser.add_argument('data_path', type=str, help="Path to the dataset (.pt file).")
    parser.add_argument('checkpoint', type=str, help="Path to the model checkpoint (.pth file).")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument('--num-classes', type=int, default=26, help="Number of classes in the dataset.")
    args = parser.parse_args()

    main(args)