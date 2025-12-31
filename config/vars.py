import torch
from pathlib import Path


DEFAULT_SEED: int = 1337
DEFAULT_BATCH_SIZE: int = 32
NUM_WORKERS: int = 4

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

RESULTS_DIR: str = "results"
FIGURES_DIR: str = "figures"

Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
