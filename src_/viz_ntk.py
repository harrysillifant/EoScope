# sinfit/viz_ntk.py
from __future__ import annotations
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from tqdm import tqdm

from .config import Config
from .training import train_k_steps
from .ntk import compute_ntk_per_layer, sample_ntk_probes_1d, sample_ntk_probes_2d, layer_param_groups

def init_ntk_figure(cfg: Config, n_layers: int):
    """
    Create a figure with a row of NTK heatmaps (one per layer) and a loss plot.
    Layout: NTKs on the left (n_layers vertically) + loss on the right.
    """
    # dynamic gridspec: n_layers rows, 2 columns (NTK + loss spanning all rows)
    height = cfg.ntk_figsize[1]
    fig = plt.figure(figsize=cfg.ntk_figsize, dpi=cfg.dpi, constrained_layout=True)
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(n_layers, 2, figure=fig, width_ratios=[3.0, 1.4])

    ax_ntks = [fig.add_subplot(gs[i, 0]) for i in range(n_layers)]
    ax_loss = fig.add_subplot(gs[:, 1])

    im_handles = []
    for i, ax in enumerate(ax_ntks):
        im = ax.imshow(np.zeros((2,2)), cmap=cfg.ntk_cmap, origin="lower", aspect="auto", interpolation="nearest")
        ax.set_title(f"Layer {i+1} NTK")
        ax.set_xlabel("j"); ax.set_ylabel("i")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        im_handles.append(im)

    (loss_line,) = ax_loss.plot([], [], linewidth=2, label="Training loss")
    ax_loss.set_title("Training Loss")
    ax_loss.set_xlabel("step"); ax_loss.set_ylabel("loss")
    ax_loss.grid(True, alpha=0.3); ax_loss.legend(loc="upper right")

    return fig, ax_ntks, ax_loss, im_handles, loss_line

def make_anim_ntk(
    cfg: Config,
    model: nn.Module,
    X_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    criterion,  # composite loss (callable: criterion(model, X, y_hat, y))
) -> Tuple[FuncAnimation, List[float]]:
    """
    Animate per-layer NTK heatmaps alongside the training loss.
    Training continues during the animation: each frame runs `steps_per_frame` GD steps.

    Returns:
        (anim, loss_history)
    """
    device = next(model.parameters()).device
    model.eval()  # eval mode (no dropout/BN noise). Autograd remains enabled.

    # ---- Build a fixed probe set for NTK computation ----
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
            probes = sample_ntk_probes_2d(cfg.x1_min, cfg.x1_max, cfg.x2_min, cfg.x2_max,
                                          cfg.ntk_probe_points, device)
        else:
            span1 = (cfg.x1_max - cfg.x1_min); pad1 = cfg.vis_pad_frac * span1
            span2 = (cfg.x2_max - cfg.x2_min); pad2 = cfg.vis_pad_frac * span2
            probes = sample_ntk_probes_2d(cfg.x1_min - pad1, cfg.x1_max + pad1,
                                          cfg.x2_min - pad2, cfg.x2_max + pad2,
                                          cfg.ntk_probe_points, device)

    # ---- Initial NTKs (no torch.no_grad() here; we need autograd) ----
    Ks0 = compute_ntk_per_layer(model, probes)          # list of (M,M) on CPU
    n_layers = len(Ks0)

    # ---- Figure & artists ----
    fig, ax_ntks, ax_loss, im_handles, loss_line = init_ntk_figure(cfg, n_layers)

    # Initialize heatmaps with real shapes & symmetric color limits
    vmax0 = max(K.abs().max().item() for K in Ks0) if Ks0 else 1.0
    for im, K in zip(im_handles, Ks0):
        K_np = K.numpy()
        im.set_data(K_np)
        im.set_clim(-vmax0, vmax0)

    # ---- Histories ----
    loss_history: List[float] = []

    total_frames = max(1, cfg.total_steps // max(1, cfg.steps_per_frame))
    state = {"step": 0, "ntk_idx": 0}

    def init():
        loss_line.set_data([], [])
        # Return all heatmap images + the loss line
        return tuple([loss_line] + im_handles)

    def update(_frame_idx: int):
        # ---- Train a chunk of steps (full-batch GD) ----
        # `train_k_steps` appends one loss per STEP, so loss_history grows by steps_per_frame
        _ = train_k_steps(model, criterion, X_train_t, y_train_t,
                          cfg.lr, cfg.steps_per_frame, loss_history, deltas:=[])
        state["step"] += cfg.steps_per_frame

        # ---- Loss curve: x matches number of loss points ----
        if loss_history:
            xs_loss = np.arange(1, len(loss_history) + 1)  # 1..#loss_points
            loss_line.set_data(xs_loss, loss_history)

            ymin, ymax = float(np.min(loss_history)), float(np.max(loss_history))
            pad = 0.05 * (ymax - ymin + 1e-8)
            ax_loss.set_ylim(max(0.0, ymin - pad), ymax + pad)
            ax_loss.set_xlim(0, cfg.total_steps)

        # ---- Recompute NTKs on stride and update images ----
        if state["step"] % cfg.ntk_stride == 0:
            Ks = compute_ntk_per_layer(model, probes)  # list of (M,M) on CPU
            vmax = max(K.abs().max().item() for K in Ks) if Ks else 1.0
            for im, K in zip(im_handles, Ks):
                im.set_data(K.numpy())
                im.set_clim(-vmax, vmax)
            state["ntk_idx"] += 1

        return tuple([loss_line] + im_handles)

    # Blitting can be flaky with multiple imshow+colorbars; disable if you see glitches.
    use_blit = False

    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=total_frames, interval=80, blit=use_blit
    )
    return anim, loss_history

def save_ntk_animation(anim: FuncAnimation, cfg: Config) -> None:
    assert cfg.ntk_out_mp4 and cfg.ntk_out_gif and cfg.run_dir
    total_frames = anim.save_count if hasattr(anim, "save_count") else 100
    pbar = tqdm(total=total_frames, desc="Rendering NTK frames")

    def progress_callback(frame_number: int, total: int):
        pbar.update(1)
        if frame_number == total - 1:
            pbar.close()

    try:
        writer = FFMpegWriter(fps=cfg.fps_mp4, bitrate=cfg.bitrate_mp4)
        anim.save(cfg.ntk_out_mp4, writer=writer, progress_callback=progress_callback)
        print(f"[OK] Saved {cfg.ntk_out_mp4}")
    except Exception as e:
        print(f"[WARN] FFMpeg failed for NTK ({e}); falling back to GIF...")
        writer = PillowWriter(fps=cfg.fps_gif)
        anim.save(cfg.ntk_out_gif, writer=writer, progress_callback=progress_callback)
        print(f"[OK] Saved {cfg.ntk_out_gif}")
