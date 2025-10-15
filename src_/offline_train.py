# src/offline_train.py
from __future__ import annotations
from typing import Callable, List, Dict, Any
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from .config import Config
from .metrics import mse_full_range, sharpness_top, ntk_spectrum_and_layer_norms
from .fit_store import FitRecorder

CriterionFn = Callable[[nn.Module, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

def _gd_epoch(model: nn.Module, criterion: CriterionFn, X: torch.Tensor, y: torch.Tensor, lr: float) -> float:
    y_hat = model(X)
    loss = criterion(model, X, y_hat, y)
    model.zero_grad(set_to_none=True)
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.add_( -lr * p.grad )
    return float(loss.item())

def run_training_and_log_csv(
    cfg: Config,
    model: nn.Module,
    X_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    X_vis: np.ndarray,
    y_true_vis: np.ndarray,
    criterion: CriterionFn,
) -> pd.DataFrame:
    """
    Trains for cfg.epochs (full-batch GD).
    Every cfg.log_every_epochs:
        - logs: epoch, step, train_loss, gen_mse, sharpness, top-k NTK eigenvalues,
                (optional) per-layer NTK Frobenius norms.
    Saves CSV at cfg.metrics_csv and returns the DataFrame.
    """
    assert cfg.metrics_csv is not None, "cfg.metrics_csv must be set by runio."
    device = next(model.parameters()).device

    X_vis_t = torch.from_numpy(X_vis).to(device)
    y_true_vis_t = torch.from_numpy(y_true_vis).to(device)

    records: List[Dict[str, Any]] = []
    v_prev = None  # eigenvector warm-start for power iteration
    steps_per_epoch = 1  # full-batch; keep 'step' for downstream plots

    pbar = tqdm(range(1, cfg.epochs + 1), desc="Offline training")
    for epoch in pbar:
        # ---- One epoch of full-batch GD ----
        train_loss = _gd_epoch(model, criterion, X_train_t, y_train_t, cfg.lr)

        # ---- Periodic logging ----
        if epoch % cfg.log_every_epochs == 0 or epoch == 1 or epoch == cfg.epochs:
            # gen MSE over full vis range
            gen_mse = mse_full_range(model, X_vis_t, y_true_vis_t)

            # sharpness (top Hessian eigenvalue)
            try:
                lam, v_prev = sharpness_top(model, criterion, X_train_t, y_train_t,
                                            iters=cfg.power_iters, init_vec=v_prev)
            except Exception:
                lam = float("nan")

            # NTK spectrum (descending) + optional per-layer norms
            try:
                w, layer_norms = ntk_spectrum_and_layer_norms(model, cfg, device)
            except Exception:
                w, layer_norms = np.array([]), []

            row = {
                "epoch": epoch,
                "step": epoch * steps_per_epoch,
                "train_loss": train_loss,
                "gen_mse": gen_mse,
                "sharpness": lam,
            }

            # top-k eigenvalues
            k = min(cfg.ntk_topk, len(w))
            for i in range(k):
                row[f"eig_{i+1}"] = float(w[i])

            # per-layer NTK Frobenius norms (optional)
            if cfg.log_layer_ntk_norms and layer_norms:
                for i, fnorm in enumerate(layer_norms, start=1):
                    row[f"layer_frob_{i}"] = float(fnorm)

            records.append(row)
            # live progress
            pbar.set_postfix({"loss": f"{train_loss:.3e}", "gen": f"{gen_mse:.3e}", "lam": f"{lam:.3e}"})

    # prepare fit recorder
    fitrec = None
    if cfg.fit_store:
        fitrec = FitRecorder(cfg, X_vis, device=device,
                             total_epochs=cfg.epochs,
                             log_every_epochs=cfg.log_every_epochs)

    pbar = tqdm(range(1, cfg.epochs + 1), desc="Offline training")
    snap_every = cfg.log_every_epochs

    for epoch in pbar:
        train_loss = _gd_epoch(model, criterion, X_train_t, y_train_t, cfg.lr)

        # record snapshot + metrics on schedule
        if epoch % snap_every == 0 or epoch == 1 or epoch == cfg.epochs:
            # (A) function fit snapshot
            if fitrec is not None:
                fitrec.record(model)

            # (B) metrics you already log (gen MSE, sharpness, NTK eigs, etc.)
            gen_mse = mse_full_range(model, X_vis_t, y_true_vis_t)
            ...
            records.append(row)
            pbar.set_postfix({...})

    if fitrec is not None:
        fitrec.close()

    df = pd.DataFrame.from_records(records)
    df.to_csv(cfg.metrics_csv, index=False)
    print(f"[OK] metrics saved -> {cfg.metrics_csv}")
    return df
