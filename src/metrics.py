from __future__ import annotations
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from .linalg import top_hessian_eigenpair
from .ntk import compute_ntk_full, topk_eigs_from_ntk

@torch.no_grad()
def mse_full_range(model: nn.Module, X_vis_t: torch.Tensor, y_true_vis_t: torch.Tensor) -> float:
    y_pred = model(X_vis_t).squeeze(1)
    return float(torch.mean((y_pred - y_true_vis_t) ** 2).item())

def sharpness_top(model: nn.Module, criterion, X_train_t: torch.Tensor, y_train_t: torch.Tensor,
                  iters: int = 20, init_vec=None) -> Tuple[float, torch.Tensor]:
    lam, v = top_hessian_eigenpair(model, criterion, X_train_t, y_train_t, iters=iters, init_vec=init_vec)
    return lam, v

def ntk_topk_eigs(model: nn.Module, X_probe: torch.Tensor, k: int):
    K = compute_ntk_full(model, X_probe)
    w = topk_eigs_from_ntk(K, k)
    return w
