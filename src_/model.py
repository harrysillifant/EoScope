from __future__ import annotations
import torch.nn as nn
from .config import Config
import numpy as np

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x): return self.net(((x + 2*np.pi) % (4*np.pi)) - 2*np.pi)

def build_model(cfg: Config) -> nn.Module:
    return MLP(cfg.input_dim, cfg.hidden)
