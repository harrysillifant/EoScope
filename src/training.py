# src/offline_train.py
from __future__ import annotations
from typing import Callable, List, Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import os

from .config import Config
from .fit_store import FitRecorder
from .metrics import mse_full_range, ntk_topk_eigs
from .linalg import top_hessian_eigenpair, flatten_params, second_hessian_eigenvalue_deflated  # <-- NEW import

CriterionFn = Callable[[nn.Module, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

def _gd_epoch(model: nn.Module, criterion: CriterionFn, X: torch.Tensor, y: torch.Tensor, lr: float,
              grad_clip: float | None = 100.0) -> float:
    y_hat = model(X)
    loss = criterion(model, X, y_hat, y)
    model.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip is not None and grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.add_(-lr * p.grad)
    return float(loss.item())

def _build_probe(cfg: Config, device: torch.device) -> torch.Tensor:
    xs = np.linspace(cfg.x_min, cfg.x_max, cfg.ntk_probe_points, dtype=np.float32)[:, None]
    return torch.from_numpy(xs).to(device)

def _build_snapshot_schedule(epochs: int, every: int) -> list[int]:
    s = {1, epochs}
    s.update(range(every, epochs + 1, every))
    return sorted(s)

def run_training_and_log_csv(
    cfg: Config,
    model: nn.Module,
    X_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    X_vis: np.ndarray,
    y_true_vis: np.ndarray,
    criterion: CriterionFn,
) -> pd.DataFrame:
    assert cfg.metrics_csv and cfg.fits_path and cfg.fits_meta_path and cfg.fits_x_path and cfg.train_points_path
    device = next(model.parameters()).device

    # Save training points once
    np.save(cfg.train_points_path, {
        "X_train": X_train_t.detach().cpu().numpy(),
        "y_train": y_train_t.detach().cpu().numpy(),
    }, allow_pickle=True)

    # Schedule & fit recorder
    schedule = _build_snapshot_schedule(cfg.epochs, cfg.log_every_epochs)
    fitrec = FitRecorder(cfg, X_vis, schedule=schedule, device=device)

    # Constant tensors
    X_vis_t = torch.from_numpy(X_vis).to(device)
    y_true_vis_t = torch.from_numpy(y_true_vis).to(device)
    X_probe_t = _build_probe(cfg, device)

    records: List[Dict[str, Any]] = []

    # --- Projection bookkeeping (all on CPU to avoid device mismatch) ---
    theta0_cpu = None           # flattened params at first snapshot
    theta_prev_cpu = None       # flattened params at previous snapshot
    v_prev = None               # warm-start eigenvector (device vector)
    v2_prev = None 

    pbar = tqdm(range(1, cfg.epochs + 1), desc="Training (offline logging)")
    for epoch in pbar:
        train_loss = _gd_epoch(model, criterion, X_train_t, y_train_t, cfg.lr)

        if epoch in schedule:
            # (A) store function fit snapshot
            fitrec.record(model, epoch)

            # (B) metrics
            gen_mse = mse_full_range(model, X_vis_t, y_true_vis_t)

            # Top Hessian eigenpair (use your working implementation)
            try:
                lam, v_prev = top_hessian_eigenpair(
                    model, criterion, X_train_t, y_train_t,
                    iters=cfg.power_iters, init_vec=v_prev
                )
            except Exception:
                lam, v_prev = float("nan"), v_prev

            try:
                if isinstance(v_prev, torch.Tensor):
                    lam2, v2_prev = second_hessian_eigenvalue_deflated(
                        model, criterion, X_train_t, y_train_t,
                        v1=v_prev, iters=cfg.power_iters, init_vec=v2_prev
                    )
                else:
                    lam2 = float("nan")
            except Exception:
                lam2 = float("nan")

            # NTK eigenvalues (top-k)
            try:
                eigs = ntk_topk_eigs(model, X_probe_t, cfg.ntk_topk)
            except Exception:
                eigs = []
                
                
            # --- NEW: spectral norms (operator 2-norm) per Linear layer ---
            spec_vals = []
            try:
                import math
                for m in model.modules():
                    if isinstance(m, nn.Linear):
                        # Move to CPU for stable linalg; use float32 for speed/stability.
                        W = m.weight.detach().cpu().float()
                        # Use torch.linalg.norm(W, 2) (spectral norm). This uses power/SVD internally.
                        # For older PyTorch, fallback to svdvals.
                        try:
                            s = torch.linalg.matrix_norm(W, 2)  # spectral norm
                            spec = float(s.item())
                        except Exception:
                            svals = torch.linalg.svdvals(W)
                            spec = float(svals.max().item())
                        if not math.isfinite(spec):
                            spec = float("nan")
                        spec_vals.append(spec)
            except Exception:
                # keep empty -> handled below
                pass


            # --- NEW: Projections of parameter displacement onto v_max ---
            # Flatten current parameters (CPU)
            theta_cpu = flatten_params(model)  # (P,) detached CPU tensor (from your linalg.py)

            # Initialize references
            if theta0_cpu is None:
                theta0_cpu = theta_cpu.clone()
            if theta_prev_cpu is None:
                theta_prev_cpu = theta_cpu.clone()

            # v_max -> CPU (ensure same dtype)
            if isinstance(v_prev, torch.Tensor):
                v_cpu = v_prev.detach().cpu()
            else:
                # No eigenvector available (e.g., failure) -> zeros to keep shapes consistent
                v_cpu = torch.zeros_like(theta_cpu)

            # Step displacement (since last snapshot) and cumulative displacement (since first)
            delta_step = theta_cpu - theta_prev_cpu
            delta_cum  = theta_cpu - theta0_cpu

            # Projections (absolute values)
            try:
                proj_step = float(torch.abs(torch.dot(delta_step, v_cpu)).item())
            except Exception:
                proj_step = float("nan")
            try:
                proj_cum = float(torch.abs(torch.dot(delta_cum,  v_cpu)).item())
            except Exception:
                proj_cum = float("nan")

            # Advance snapshot reference
            theta_prev_cpu = theta_cpu.clone()

            # Compose row
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "gen_mse": gen_mse,
                "sharpness": lam,
                "sharpness_2": lam2,
                "proj_step": proj_step,   # <-- NEW
                "proj_cum": proj_cum,     # <-- NEW
            }
            for i, val in enumerate(eigs, start=1):
                row[f"eig_{i}"] = float(val)

            for i, val in enumerate(spec_vals, start=1):
                row[f"spec_{i}"] = float(val)

            records.append(row)
            pbar.set_postfix({
                "loss": f"{train_loss:.3e}" if np.isfinite(train_loss) else "NaN",
                "gen":  f"{gen_mse:.3e}"    if np.isfinite(gen_mse) else "NaN",
                "lam":  f"{lam:.3e}"        if np.isfinite(lam) else "NaN",
                "lam2": f"{lam2:.3e}"       if np.isfinite(lam2) else "NaN",
                "proj": f"{proj_step:.3e}"  if np.isfinite(proj_step) else "NaN",
            })

    fitrec.close()
    df = pd.DataFrame.from_records(records)
    df.to_csv(cfg.metrics_csv, index=False)
    print(f"[OK] metrics saved -> {cfg.metrics_csv}  (rows={len(df)})")
    return df
