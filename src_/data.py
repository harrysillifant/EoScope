# sinfit/data.py
from __future__ import annotations
from typing import Tuple, Callable, Optional, Dict, Any
import numpy as np
import torch
from .config import Config
from .functions import FUNCTIONS

TargetFn = Callable[[np.ndarray], np.ndarray]

def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

def resolve_target_fn(cfg: Config, override_fn: Optional[TargetFn] = None) -> Tuple[TargetFn, Dict[str, Any]]:
    """
    Resolve the target function to use.
    Priority:
      1) override_fn if provided (callable)
      2) FUNCTIONS[cfg.function_name] with cfg.function_params
    Returns:
      (callable f(x), params_dict_used)
    """
    if override_fn is not None:
        return override_fn, (cfg.function_params or {})
    if cfg.function_name not in FUNCTIONS:
        raise ValueError(f"Unknown function_name '{cfg.function_name}'. Available: {list(FUNCTIONS.keys())}")
    return (lambda x: FUNCTIONS[cfg.function_name](x, **(cfg.function_params or {}))), (cfg.function_params or {})

def generate_data(cfg: Config, target_fn: TargetFn) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      X_train_t: (N, D) float32
      y_train_t: (N, 1) float32
    """
    N = cfg.n_samples

    # --- Sample inputs ---
    x = np.random.uniform(cfg.x_min, cfg.x_max, size=N).astype(np.float32)
    X = x[:, None]  # (N,1)

    # --- Target values (normalize to 1-D) ---
    y_clean = target_fn(X)                     # could be (N,), (N,1), or weird shapes
    y_clean = np.asarray(y_clean, dtype=np.float32).reshape(-1)  # => (N,)
    if y_clean.shape[0] != N:
        raise ValueError(f"target_fn returned {y_clean.shape}, expected N={N}")

    # --- Noise (1-D), then add ---
    noise = (cfg.noise_std * np.random.randn(N)).astype(np.float32)  # (N,)
    y = y_clean + noise                                              # (N,)

    # --- Tensors with canonical shapes ---
    X_train_t = torch.from_numpy(X.astype(np.float32))               # (N,D)
    y_train_t = torch.from_numpy(y.astype(np.float32)).unsqueeze(1)  # (N,1)
    return X_train_t, y_train_t