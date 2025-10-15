# sinfit/main.py
from __future__ import annotations
import numpy as np
import torch

from .config import Config
from .data import set_seeds, generate_data, resolve_target_fn, TargetFn
from .model import build_model
from .viz import make_anim, save_with_progress
from .runio import prepare_run_dir
from .losses import get_composite_loss
from .viz_ntk import make_anim_ntk, save_ntk_animation

def _make_vis_grid(cfg: Config, target_fn):
    span = cfg.x_max - cfg.x_min
    pad = cfg.vis_pad_frac * span
    x0 = cfg.x_min - pad if cfg.x_vis_min is None else cfg.x_vis_min
    x1 = cfg.x_max + pad if cfg.x_vis_max is None else cfg.x_vis_max
    x_vis = np.linspace(x0, x1, cfg.n_plot_points, dtype=np.float32)
    X_vis = x_vis[:, None]                 # (P,1)

    y_true_vis = target_fn(X_vis).astype(np.float32)        # (P,)
    return X_vis, y_true_vis


def main(cfg: Config | None = None, target_fn_override: TargetFn | None = None) -> None:
    """
    If you pass a custom callable in target_fn_override(x: np.ndarray)->np.ndarray,
    it will override cfg.function_name/params.
    """
    cfg = cfg or Config()
    set_seeds(cfg.seed)

    # Prepare run dir & save config
    run_dir = prepare_run_dir(cfg)
    print(f"[INFO] Run directory: {run_dir}")

    # Resolve target function
    target_fn, used_params = resolve_target_fn(cfg, override_fn=target_fn_override)
    print(f"[INFO] Target function: {cfg.function_name if target_fn_override is None else 'override callable'} "
          f"params={used_params}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Training data on device
    x_train_t, y_train_t = generate_data(cfg, target_fn)
    x_train_t, y_train_t = x_train_t.to(device), y_train_t.to(device)

    # Visualization grid (wider domain) using the same function
    x_vis, y_true_vis = _make_vis_grid(cfg, target_fn)

    # Model
    model = build_model(cfg).to(device)
    criterion = get_composite_loss(cfg.criterion_name, cfg.criterion_params, device=device)

    # Animation
    anim, _ = make_anim(cfg, model, x_train_t, y_train_t, x_vis, y_true_vis, criterion)

    # # Save
    save_with_progress(anim, cfg)

    # 2) NTK + loss animation
    #   (Reuses the same model and training tensors; continues training while drawing NTK heatmaps)
    # model = build_model(cfg).to(device)
    # anim_ntk, _ = make_anim_ntk(cfg, model, x_train_t, y_train_t, criterion)
    # save_ntk_animation(anim_ntk, cfg)


# src/main.py  (add an offline path)
from .offline_train import run_training_and_log_csv
from .offline_anim import make_offline_anim_from_csv

def main_offline(cfg: Config, target_fn_override=None, loss_fn=None):
    cfg = cfg or Config()
    set_seeds(cfg.seed)

    # Prepare run dir & save config
    run_dir = prepare_run_dir(cfg)
    print(f"[INFO] Run directory: {run_dir}")

    # Resolve target function
    target_fn, used_params = resolve_target_fn(cfg, override_fn=target_fn_override)
    print(f"[INFO] Target function: {cfg.function_name if target_fn_override is None else 'override callable'} "
          f"params={used_params}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Training data on device
    x_train_t, y_train_t = generate_data(cfg, target_fn)
    x_train_t, y_train_t = x_train_t.to(device), y_train_t.to(device)

    # Visualization grid (wider domain) using the same function
    x_vis, y_true_vis = _make_vis_grid(cfg, target_fn)
    # ... same prep as your main(): seeds, run_dir, target_fn, device, data, model init, criterion ...
    # Build model (single copy is enough)

    criterion = get_composite_loss(cfg.criterion_name, cfg.criterion_params, device=device)
    model = build_model(cfg).to(device)

    # Train & log to CSV
    df = run_training_and_log_csv(cfg, model, x_train_t, y_train_t, x_vis, y_true_vis, criterion)

    # Make an animation from the CSV (optional)
    make_offline_anim_from_csv(
        cfg.metrics_csv,
        out_mp4=cfg.offline_anim_mp4,
        out_gif=cfg.offline_anim_gif,
        fps=cfg.fps_mp4,
        bitrate=cfg.bitrate_mp4,
        k_eigs=cfg.ntk_topk,
        log_eigs=True,
    )
