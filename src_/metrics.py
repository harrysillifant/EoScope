# src/metrics.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn

from .ntk import compute_ntk_per_layer, sample_ntk_probes_1d, sample_ntk_probes_2d
from .linalg import top_hessian_eigenpair

@torch.no_grad()
def mse_full_range(model: nn.Module, X_vis_t: torch.Tensor, y_true_vis_t: torch.Tensor) -> float:
    y_pred = model(X_vis_t).squeeze(1)
    return float(torch.mean((y_pred - y_true_vis_t) ** 2).item())

def sharpness_top(model: nn.Module, criterion, X_train_t: torch.Tensor, y_train_t: torch.Tensor,
                  iters: int = 20, init_vec=None) -> Tuple[float, torch.Tensor]:
    # requires grad
    lam, v = top_hessian_eigenpair(model, criterion, X_train_t, y_train_t, iters=iters, init_vec=init_vec)
    return lam, v

def ntk_spectrum_and_layer_norms(
    model: nn.Module,
    cfg,
    device: torch.device
) -> Tuple[np.ndarray, List[float]]:
    # Build probe set
    if cfg.input_dim == 1:
        if cfg.ntk_use_train_domain:
            probes = sample_ntk_probes_1d(cfg.x_min, cfg.x_max, cfg.ntk_probe_points, device)
        else:
            span = (cfg.x_max - cfg.x_min)
            lo = cfg.x_min - cfg.vis_pad_frac * span
            hi = cfg.x_max + cfg.vis_pad_frac * span
            probes = sample_ntk_probes_1d(lo, hi, cfg.ntk_probe_points, device)
    else:
        if cfg.ntk_use_train_domain:
            probes = sample_ntk_probes_2d(cfg.x1_min, cfg.x1_max, cfg.x2_min, cfg.x2_max, cfg.ntk_probe_points, device)
        else:
            s1 = (cfg.x1_max - cfg.x1_min); p1 = cfg.vis_pad_frac*s1
            s2 = (cfg.x2_max - cfg.x2_min); p2 = cfg.vis_pad_frac*s2
            probes = sample_ntk_probes_2d(cfg.x1_min-p1, cfg.x1_max+p1, cfg.x2_min-p2, cfg.x2_max+p2, cfg.ntk_probe_points, device)

    # per-layer kernels (CPU tensors)
    Ks = compute_ntk_per_layer(model, probes)  # list of (M,M)
    # per-layer Frobenius norms
    layer_norms = [float(torch.linalg.matrix_norm(K, ord='fro').item()) for K in Ks]
    # full NTK = sum over layers
    Ksum = sum(K.numpy() for K in Ks)
    # spectrum (descending)
    w = np.linalg.eigvalsh((Ksum + Ksum.T) * 0.5)
    w = np.sort(w)[::-1]
    return w, layer_norms
