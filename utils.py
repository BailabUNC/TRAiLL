# utils.py

import random

import numpy as np
import torch

def set_seed(seed: int = 114514) -> torch.Generator:
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # For all GPUs
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator