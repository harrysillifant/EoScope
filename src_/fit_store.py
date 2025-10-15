# src/fit_store.py
from __future__ import annotations
import json, os
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

_DTYPE_MAP = {
    "float16": np.float16,
    "float32": np.float32,
}

class FitRecorder:
    """
    Memory-efficient writer of model predictions over training.
    Creates a disk-backed memmap of shape (T_snapshots, P_fit)   for 1D,
                                   or (T_snapshots, G, G)        for 2D.
    Where T_snapshots = number of logging points = ceil(epochs / log_every_epochs).
    """
    def __init__(self,
                 cfg,
                 X_vis: np.ndarray,
                 device: torch.device,
                 total_epochs: int,
                 log_every_epochs: int):
        self.cfg = cfg
        self.device = device
        self.input_dim = cfg.input_dim
        self.dtype = _DTYPE_MAP.get(cfg.fit_dtype, np.float16)

        # choose grid to store
        if self.input_dim == 1:
            P_all = X_vis.shape[0]
            P_keep = min(cfg.fit_points_1d, P_all)
            idx = np.linspace(0, P_all - 1, P_keep, dtype=int)
            self.X_fit = X_vis[idx]                      # (P_fit,1)
            self.shape_tail = (P_keep,)
        else:
            # make a coarse GxG grid across vis range
            G = cfg.fit_grid_2d
            x1 = np.linspace(X_vis[:,0].min(), X_vis[:,0].max(), G, dtype=np.float32)
            x2 = np.linspace(X_vis[:,1].min(), X_vis[:,1].max(), G, dtype=np.float32)
            X1, X2 = np.meshgrid(x1, x2, indexing="xy")
            self.X_fit = np.stack([X1.ravel(), X2.ravel()], axis=1)  # (G*G,2)
            self.shape_tail = (G, G)

        # number of time snapshots
        self.T = int(np.ceil(total_epochs / log_every_epochs))
        self.t = 0

        # save the grid now
        if cfg.fits_x_path:
            np.save(cfg.fits_x_path, self.X_fit.astype(np.float32))

        # create memmap
        full_shape = (self.T,) + self.shape_tail
        assert cfg.fits_path is not None, "cfg.fits_path must be set"
        self.mm = np.memmap(cfg.fits_path, mode="w+", dtype=self.dtype, shape=full_shape)

        # metadata
        meta = {
            "input_dim": int(self.input_dim),
            "dtype": cfg.fit_dtype,
            "shape": list(full_shape),
            "epochs": int(total_epochs),
            "log_every_epochs": int(log_every_epochs),
            "grid_file": os.path.basename(cfg.fits_x_path) if cfg.fits_x_path else None,
            "order": "C",
        }
        if cfg.fits_meta_path:
            with open(cfg.fits_meta_path, "w") as f:
                json.dump(meta, f, indent=2)

        # cache tensor version of X_fit
        self.X_fit_t = torch.from_numpy(self.X_fit).to(device)

    @torch.no_grad()
    def record(self, model: nn.Module):
        """Compute prediction on X_fit and append to memmap at row t."""
        model.eval()
        y = model(self.X_fit_t).squeeze(1)  # (P_fit,) or (G*G,)
        y_np = y.detach().cpu().numpy().astype(self.dtype, copy=False)
        if len(self.shape_tail) == 2:
            G1, G2 = self.shape_tail
            y_np = y_np.reshape(G1, G2)
        self.mm[self.t, ...] = y_np
        self.mm.flush()
        self.t += 1

    def close(self):
        del self.mm
