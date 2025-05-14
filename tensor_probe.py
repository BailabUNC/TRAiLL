import torch
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Inspect a .pt file.")
parser.add_argument("path", type=str, help="Path to the .pt file to inspect.")
args = parser.parse_args()

# Load and inspect the .pt file
data = torch.load(args.path, map_location='cpu')

if isinstance(data, dict):
    for key, value in data.items():
        if torch.is_tensor(value):
            print(f"{key}: tensor shape = {tuple(value.shape)}, dtype = {value.dtype}")
        else:
            print(f"{key}: type = {type(value)}")
elif isinstance(data, list):
    for i, item in enumerate(data):
        if torch.is_tensor(item):
            print(f"[{i}]: tensor shape = {tuple(item.shape)}, dtype = {item.dtype}")
        else:
            print(f"[{i}]: type = {type(item)}")
elif torch.is_tensor(data):
    print(f"Top-level tensor: shape = {tuple(data.shape)}, dtype = {data.dtype}")
else:
    print(f"Top-level object type: {type(data)}")