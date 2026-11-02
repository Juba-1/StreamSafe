"""Device / precision used by every PyTorch model."""

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HALF = torch.float16 if DEVICE == "cuda" else torch.float32
