# sinfit/config.py
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

@dataclass
class Config:
    # Data (training interval)
    n_samples: int = 500
    x_min: float = -2 * math.pi
    x_max: float =  2 * math.pi
    noise_std: float = 0.08
    seed: int = 0

    # ---- Target function selection ----
    # Choose by name from sinfit.functions.FUNCTIONS (e.g., "sin", "poly")
    function_name: str = "sin"
    # Optional kwargs for that function, saved to config.json
    function_params: Dict[str, Any] = None  # e.g., {"a":0.1,"b":0.0,"c":1.0,"d":0.0}

    # Model / training
    input_dim: int = 1  # only works with 1d
    hidden: int = 64
    lr: float = 1e-2
    total_steps: int = 5000
    steps_per_frame: int = 10
    criterion_name: str = "mse"

    # Curvature tracking
    sharpness_stride: int = 20
    power_iters: int = 20

    # Visualization grid
    n_plot_points: int = 1200
    x_vis_min: Optional[float] = None
    x_vis_max: Optional[float] = None
    y_vis_min: Optional[float] = None
    y_vis_max: Optional[float] = None
    vis_pad_frac: float = 0.5

    # Figure / animation
    dpi: int = 140
    figsize: Tuple[float, float] = (11.5, 7.2)
    fps_mp4: int = 20
    bitrate_mp4: int = 1800
    fps_gif: int = 15

    # ---- Run management ----
    runs_root: str = "runs"
    project_name: Optional[str] = None
    run_dir: Optional[str] = None
    out_mp4: Optional[str] = None
    out_gif: Optional[str] = None

    # Optional ffmpeg path
    ffmpeg_path: Optional[str] = None

    # Visualization padding (used for both dims if input_dim=2)
    vis_pad_frac: float = 0.5

    # ---- Loss selection (JSON friendly) ----
    # base loss: "mse" | "l1"
    criterion_name: str = "mse"
    # extra options for loss/regularization/PDE
    criterion_params: Dict[str, Any] = field(default_factory=lambda: {
        # L2 weight decay on parameters (0.0 = off)
        "l2_weight": 0.0,
        # PDE term selection & weight (set weight=0 to disable)
        # Currently supported: "poisson1d" (u_xx - g(x) = 0) or "helmholtz1d" (u_xx + k^2 u = g(x))
        "pde": None,           # e.g., "poisson1d", "helmholtz1d"
        "pde_weight": 0.0,     # lambda_PDE
        "helmholtz_k": 1.0,    # used when pde == "helmholtz1d"
        # number of collocation points sampled each step (0 => use x_batch)
        "n_collocation": 0,
        # collocation domain (defaults to training domain if None)
        "colloc_min": None,
        "colloc_max": None,
        # optional right-hand side g(x) as a string key (for demo) or use override callable
        "rhs": None            # e.g., "zero"
            })

    # --- NTK settings ---
    ntk_stride: int = 20            # compute NTK every N steps
    ntk_probe_points: int = 36      # number of probe inputs (keep small: 16-64)
    ntk_use_train_domain: bool = True  # sample probes from train box/interval (else from vis box)
    ntk_cmap: str = "coolwarm"      # colormap for heatmaps
    ntk_figsize: tuple[float, float] = (10.5, 6.5)  # figure size for NTK animation
    ntk_bilinear_upsample: int = 1  # >1 to visually upscale kernels in imshow
    ntk_out_mp4: str | None = None  # set by runio to "<run_dir>/ntk_animation.mp4"
    ntk_out_gif: str | None = None  # set by runio to "<run_dir>/ntk_animation.gif"


    # Offline logging / training
    epochs: int = 2000                 # number of full-batch epochs
    log_every_epochs: int = 5          # record metrics every k epochs

    # NTK logging
    ntk_probe_points: int = 32         # small! (e.g., 16, 32, 36)
    ntk_use_train_domain: bool = True  # sample probes from train interval/box
    ntk_topk: int = 6                  # number of eigenvalues to keep
    log_layer_ntk_norms: bool = True   # also log per-layer Frobenius norms

    # Output (will be set by your runio)
    metrics_csv: Optional[str] = None  # e.g., runs/…/metrics.csv
    offline_anim_mp4: Optional[str] = None
    offline_anim_gif: Optional[str] = None

    # offline logging frequency
    epochs: int = 2000
    log_every_epochs: int = 5

    # function fit storage
    fit_store: bool = True
    fit_dtype: str = "float16"   # "float16" or "float32"
    fit_points_1d: int = 512     # subsampled along X_vis for 1D
    fit_grid_2d: int = 64        # GxG grid for 2D
    fits_path: Optional[str] = None       # set by runio e.g. runs/.../fits.dat
    fits_meta_path: Optional[str] = None  # runs/.../fits_meta.json
    fits_x_path: Optional[str] = None     # runs/.../fits_grid.npy