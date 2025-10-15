from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class Config:
    # ---- project/io ----
    project_name: str = "run"
    seed: int = 42
    run_dir: Optional[str] = None      # set at runtime
    metrics_csv: Optional[str] = None  # set at runtime
    fits_path: Optional[str] = None    # set at runtime (memmap file)
    fits_meta_path: Optional[str] = None
    fits_x_path: Optional[str] = None
    train_points_path: Optional[str] = None

    # ---- data (1D only) ----
    input_dim: int = 1
    # training interval
    x_min: float = -2.0
    x_max: float = 2.0
    n_samples: int = 512
    noise_std: float = 0.0
    # visualization (full range for gen error)
    n_plot_points: int = 2048
    vis_pad_frac: float = 0.5         # expand beyond train interval for X_vis

    # ---- target function ----
    function_name: str = "sin"        # "sin", "poly", "custom"
    # poly: a*x^3 + b*x^2 + c*x + d
    function_params: dict = None      # e.g., {'a':2,'b':1,'c':-1,'d':5}
    # if function_name == "custom", provide a python path "package.module:fn"
    # where fn: np.ndarray(N,1) -> np.ndarray(N,1) (float32)

    # ---- model ----
    hidden: int = 64
    depth: int = 2                    # number of hidden layers
    activation: str = "tanh"          # "tanh" or "relu"

    # ---- optimization ----
    lr: float = 0.1
    epochs: int = 2000
    log_every_epochs: int = 5

    # ---- loss selection ----
    loss_name: str = "mse"            # "mse", "l1", "custom"
    custom_loss: Optional[str] = None # "package.module:func"

    # ---- NTK logging ----
    ntk_probe_points: int = 64
    ntk_topk: int = 6

    # ---- sharpness (power iteration) ----
    power_iters: int = 20

    # ---- fit storage (1D) ----
    fit_store: bool = True
    fit_dtype: str = "float16"        # "float16" | "float32"
    fit_points_1d: int = 512          # subsampled from X_vis for storage
