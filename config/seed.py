# config/seed.py

"""
Utility for setting global random seeds for reproducibility.

This ensures deterministic behavior across:
    * Python's built-in RNG
    * NumPy RNG
    * PyTorch CPU
    * PyTorch CUDA (if available)
"""

import random
import numpy as np
import torch

from config.vars import DEFAULT_SEED


def set_seed(seed: int = DEFAULT_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
