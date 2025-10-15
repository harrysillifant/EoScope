from __future__ import annotations
import torch
import torch.nn as nn
from .config import Config
import numpy as np
import math

def _act(name: str):
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"unknown activation: {name}")

class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        layers = []
        in_dim = int(getattr(cfg, "input_dim", 1))
        for _ in range(cfg.depth):
            layers += [nn.Linear(in_dim, cfg.hidden), _act(cfg.activation)]
            in_dim = cfg.hidden
        layers += [nn.Linear(in_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x) 
        period = 2*math.pi
        T = x.new_tensor(period)
        halfT = 0.5 * T
        x_wrapped = torch.remainder(x + halfT, T) - halfT  # in (-T/2, T/2]
        return self.net(x_wrapped)

def build_model(cfg: Config) -> nn.Module:
    return MLP(cfg)
