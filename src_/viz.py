# sinfit/viz.py
from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401; needed to enable 3D
from tqdm import tqdm

from .config import Config
from .training import train_k_steps
from .linalg import top_hessian_eigenpair


def init_figure(
    cfg: Config,
    X_vis: np.ndarray,
    y_true_vis: np.ndarray,
    X_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes, plt.Axes, plt.Axes], Dict[str, object]]:
    """
    Build a 2x2 dashboard.

    Top-left:
        - If cfg.input_dim == 1: learned function (line), true fn (dashed),
          training samples (scatter), shaded training interval.
        - If cfg.input_dim == 2: 3D surface of the true function (translucent)
          + a placeholder surface for the learned function + training points (3D scatter)
          + a dashed rectangle on the x1-x2 plane showing the training box.

    Top-right:
        Training loss (solid) and generalization error (dotted), same y-axis.

    Bottom-left:
        Sharpness (top Hessian eigenvalue) with a dashed threshold at 2/η.

    Bottom-right:
        Projection magnitude |Δθ ⋅ v_max|.

    Returns:
        fig, (ax_fun, ax_loss, ax_sharp, ax_proj), handles
        where handles is a dict with artist references you should update in make_anim.
    """
    # Create 2x2 layout, but we'll replace the [0,0] with a 3D axis when input_dim==2
    fig, axes = plt.subplots(2, 2, figsize=cfg.figsize, dpi=cfg.dpi)
    ax_loss, ax_sharp, ax_proj = axes[0, 1], axes[1, 0], axes[1, 1]

    handles: Dict[str, object] = {}

    # ---------- Function panel ----------
    if cfg.input_dim == 1:
        ax_fun = axes[0, 0]

        # True curve across the visualization domain
        ax_fun.plot(
            X_vis[:, 0], y_true_vis,
            linestyle="--", linewidth=2, label="True f(x)"
        )

        # Learned curve placeholder
        line_pred, = ax_fun.plot([], [], linewidth=2, label="Learned f̂(x)")
        handles["line_pred"] = line_pred

        # Shade training interval
        ax_fun.axvspan(cfg.x_min, cfg.x_max, alpha=0.08, label="Training interval")
        ax_fun.axvline(cfg.x_min, linestyle="--", linewidth=1)
        ax_fun.axvline(cfg.x_max, linestyle="--", linewidth=1)

        # Training samples (ensure CPU numpy for Matplotlib)
        ax_fun.scatter(
            X_train_t[:, 0].detach().cpu().numpy(),
            y_train_t.detach().cpu().numpy().squeeze(1),
            s=12, alpha=0.85, label="Train samples"
        )

        ax_fun.set_xlim(X_vis[:, 0].min(), X_vis[:, 0].max())
        ax_fun.set_ylim(
            min(y_true_vis.min(), -1.7),
            max(y_true_vis.max(), 1.7)
        )
        ax_fun.set_title("Function fit (shaded = training interval)")
        ax_fun.set_xlabel("x")
        ax_fun.set_ylabel("y")
        ax_fun.legend(loc="lower right")

        info_text = ax_fun.text(0.02, 0.95, "", transform=ax_fun.transAxes, va="top")
        handles["info_text"] = info_text

    else:
        # Replace [0,0] axis by a 3D axis
        fig.delaxes(axes[0, 0])
        ax_fun = fig.add_subplot(2, 2, 1, projection="3d")

        # Rebuild a grid (m x m) from flattened X_vis for surf plotting
        m = int(np.sqrt(len(X_vis)))
        assert m * m == len(X_vis), (
            "For 3D surface, n_plot_points should be a perfect square (e.g. 2500, 3600)."
        )
        X1 = X_vis[:, 0].reshape(m, m)
        X2 = X_vis[:, 1].reshape(m, m)
        Y_true = y_true_vis.reshape(m, m)

        # True surface (translucent)
        ax_fun.plot_surface(
            X1, X2, Y_true,
            alpha=0.25, linewidth=0, antialiased=True
        )

        # Placeholder learned surface (we'll replace Z each frame)
        surf_pred = ax_fun.plot_surface(
            X1, X2, np.zeros_like(Y_true),
            alpha=0.60, linewidth=0, antialiased=True
        )
        handles["surf_pred"] = surf_pred
        handles["surf_grid"] = (X1, X2)  # reuse in updates

        # Training points as 3D scatter
        Xt = X_train_t.detach().cpu().numpy()
        Yt = y_train_t.detach().cpu().numpy().squeeze(1)
        ax_fun.scatter(Xt[:, 0], Xt[:, 1], Yt, s=8, alpha=0.85, label="Train samples")

        # Draw dashed training box on the x1-x2 plane at z = min true
        z0 = float(np.nanmin(Y_true))
        x1min, x1max, x2min, x2max = cfg.x1_min, cfg.x1_max, cfg.x2_min, cfg.x2_max
        ax_fun.plot([x1min, x1max, x1max, x1min, x1min],
                    [x2min, x2min, x2max, x2max, x2min],
                    [z0] * 5, "k--", linewidth=1)

        ax_fun.set_title("Function fit (3D). Dashed = training box")
        ax_fun.set_xlabel("x1")
        ax_fun.set_ylabel("x2")
        ax_fun.set_zlabel("y")

        info_text = ax_fun.text2D(0.02, 0.95, "", transform=ax_fun.transAxes, va="top")
        handles["info_text"] = info_text

    # ---------- Loss + Generalization panel ----------
    loss_line, = ax_loss.plot([], [], linewidth=2, label="Training loss (MSE)")
    gen_line,  = ax_loss.plot([], [], linewidth=2, linestyle=":", label="Gen error (MSE, OOD)")
    ax_loss.set_xlim(0, cfg.total_steps)
    ax_loss.set_ylim(0.0, 1.5)
    ax_loss.set_title("Loss & Generalization Error")
    ax_loss.set_xlabel("step")
    ax_loss.set_ylabel("MSE")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(loc="upper right")
    handles["loss_line"] = loss_line
    handles["gen_line"] = gen_line

    # ---------- Sharpness with threshold 2/η ----------
    sharp_line, = ax_sharp.plot([], [], linewidth=2, label="λ_max")
    threshold = 2.0 / cfg.lr
    ax_sharp.axhline(
        y=threshold, linestyle="--", linewidth=1.5,
        label=f"2/η = {threshold:.2f}"
    )
    ax_sharp.set_xlim(0, max(1, cfg.total_steps // max(1, cfg.sharpness_stride)))
    ax_sharp.set_ylim(0.0, max(5.0, threshold * 1.2))
    ax_sharp.set_title("Top Hessian eigenvalue (sharpness)")
    ax_sharp.set_xlabel(f"check idx (every {cfg.sharpness_stride} steps)")
    ax_sharp.set_ylabel("λ_max")
    ax_sharp.grid(True, alpha=0.3)
    ax_sharp.legend(loc="upper right")
    handles["sharp_line"] = sharp_line

    # ---------- Projection magnitude ----------
    proj_line, = ax_proj.plot([], [], linewidth=2)
    ax_proj.set_xlim(0, max(1, cfg.total_steps // max(1, cfg.sharpness_stride)))
    ax_proj.set_ylim(0.0, 1.0)
    ax_proj.set_title("|Δθ · v_max| (projection magnitude)")
    ax_proj.set_xlabel(f"check idx (every {cfg.sharpness_stride} steps)")
    ax_proj.set_ylabel("magnitude")
    ax_proj.grid(True, alpha=0.3)
    handles["proj_line"] = proj_line

    return fig, (ax_fun, ax_loss, ax_sharp, ax_proj), handles


# def init_figure(cfg: Config, x_vis: np.ndarray, y_true_vis: np.ndarray,
#                 x_train_t: torch.Tensor, y_train_t: torch.Tensor):
#     """
#     2x2 dashboard:
#       (0,0) function fit (over visualization range; shaded training interval)
#       (0,1) loss panel: training loss + out-of-range generalization error (MSE)
#       (1,0) sharpness (top Hessian eigenvalue) with threshold line
#       (1,1) |Δθ · v_max|
#     """
#     fig, axes = plt.subplots(2, 2, figsize=cfg.figsize, dpi=cfg.dpi)
#     ax_fun, ax_loss = axes[0,0], axes[0,1]
#     ax_sharp, ax_proj = axes[1,0], axes[1,1]

#     # --- Function fit (top-left) ---
#     (line_pred,) = ax_fun.plot([], [], linewidth=2, label="Learned function")
#     ax_fun.plot(x_vis, y_true_vis, linestyle="--", linewidth=2, label="True function")

#     ax_fun.axvspan(cfg.x_min, cfg.x_max, alpha=0.08, color=None, label="Training range")
#     ax_fun.axvline(cfg.x_min, linestyle="--", linewidth=1)
#     ax_fun.axvline(cfg.x_max, linestyle="--", linewidth=1)

#     ax_fun.scatter(
#         x_train_t.squeeze(1).detach().cpu().numpy(),
#         y_train_t.squeeze(1).detach().cpu().numpy(),
#         s=12, alpha=0.85, label="Training samples"
#     )
#     info_text = ax_fun.text(0.02, 0.95, "", transform=ax_fun.transAxes, va="top")
#     ax_fun.set_xlim(x_vis.min(), x_vis.max()); ax_fun.set_ylim(-1.7, 1.7)
#     ax_fun.set_title("Function fit (shaded = training interval)")
#     ax_fun.set_xlabel("x"); ax_fun.set_ylabel("y")
#     ax_fun.legend(loc="lower right")

#     # --- Loss panel (top-right): training loss + gen error ---
#     (loss_line,) = ax_loss.plot([], [], linewidth=2, label="Training loss (MSE)")
#     (gen_line,)  = ax_loss.plot([], [], linewidth=2, linestyle=":", label="Gen error (out-of-range MSE)")
#     ax_loss.set_xlim(0, cfg.total_steps)
#     ax_loss.set_ylim(0.0, 1.5)  # will expand dynamically
#     ax_loss.set_title("Loss & Generalization Error")
#     ax_loss.set_xlabel("step"); ax_loss.set_ylabel("MSE")
#     ax_loss.grid(True, alpha=0.3)
#     ax_loss.legend(loc="upper right")

#     # --- Sharpness (bottom-left) with threshold 2/eta ---
#     (sharp_line,) = ax_sharp.plot([], [], linewidth=2, label="λ_max")
#     threshold = 2.0 / cfg.lr
#     ax_sharp.axhline(y=threshold, linestyle="--", linewidth=1.5, label=f"2/η = {threshold:.2f}")
#     ax_sharp.set_xlim(0, cfg.total_steps // cfg.sharpness_stride)
#     ax_sharp.set_ylim(0.0, max(5.0, threshold * 1.2))
#     ax_sharp.set_title("Top Hessian eigenvalue (sharpness)")
#     ax_sharp.set_xlabel(f"check idx (every {cfg.sharpness_stride} steps)")
#     ax_sharp.set_ylabel("λ_max")
#     ax_sharp.grid(True, alpha=0.3)
#     ax_sharp.legend(loc="upper right")

#     # --- Projection magnitude (bottom-right) ---
#     (proj_line,) = ax_proj.plot([], [], linewidth=2)
#     ax_proj.set_xlim(0, cfg.total_steps // cfg.sharpness_stride)
#     ax_proj.set_ylim(0.0, 1.0)
#     ax_proj.set_title("|Δθ · v_max| (projection magnitude)")
#     ax_proj.set_xlabel(f"check idx (every {cfg.sharpness_stride} steps)")
#     ax_proj.set_ylabel("magnitude")
#     ax_proj.grid(True, alpha=0.3)

#     # Return both lines from loss panel so we can update them later
#     return fig, (ax_fun, ax_loss, ax_sharp, ax_proj), (line_pred, loss_line, gen_line, sharp_line, proj_line), info_text

def make_anim(
    cfg: Config,
    model: nn.Module,
    X_train_t: torch.Tensor,
    y_train_t: torch.Tensor,
    X_vis: np.ndarray,
    y_true_vis: np.ndarray,
    criterion: nn.Module | callable,
) -> Tuple[FuncAnimation, List[float]]:
    """
    Assemble the animation. Each frame:
      - Runs `cfg.steps_per_frame` GD steps on the full training set.
      - Updates function panel (1D line or 3D surface).
      - Updates loss panel with training loss and generalization MSE over the full visualization range.
      - Every `cfg.sharpness_stride` steps: updates sharpness and |Δθ·v_max|.

    Returns:
      (animation, loss_history)
    """
    device = next(model.parameters()).device

    # Build figure and grab artists
    fig, (ax_fun, ax_loss, ax_sharp, ax_proj), H = init_figure(cfg, X_vis, y_true_vis, X_train_t, y_train_t)

    # Tensors for visualization & truth
    X_vis_t = torch.from_numpy(X_vis).to(device)
    y_true_vis_t = torch.from_numpy(y_true_vis).to(device)

    # Histories
    loss_history: List[float] = []
    gen_history:  List[float] = []
    sharp_history: List[float] = []
    proj_history:  List[float] = []
    deltas: List[torch.Tensor] = []

    total_frames = max(1, cfg.total_steps // max(1, cfg.steps_per_frame))
    state = {"step": 0, "check_idx": 0, "last_v": None}

    # For 3D, cache grid for fast surface updates
    if cfg.input_dim == 2:
        m = int(np.sqrt(len(X_vis)))
        X1, X2 = H["surf_grid"]  # set in init_figure
        assert X1.shape == (m, m) and X2.shape == (m, m)

    def init():
        # Clear dynamic artists (loss/gen/sharp/proj lines already empty)
        info_text = H["info_text"]
        if cfg.input_dim == 1:
            H["line_pred"].set_data([], [])
        # (3D surface will be redrawn in update)
        info_text.set_text("")
        return tuple(a for a in [
            H.get("line_pred"),
            H["loss_line"], H["gen_line"],
            H["sharp_line"], H["proj_line"],
            H["info_text"]
        ] if a is not None)

    def maybe_update_curvature_and_projection():
        if state["step"] == 0 or (state["step"] % cfg.sharpness_stride != 0):
            return

        lam, v = top_hessian_eigenpair(
            model, criterion, X_train_t, y_train_t,
            iters=cfg.power_iters, init_vec=state["last_v"]
        )
        state["last_v"] = v
        sharp_history.append(lam)

        if len(deltas) > 0:
            d = deltas[-1].to(v.device)
            proj = float(torch.abs((d * v).sum()).item())
        else:
            proj = 0.0
        proj_history.append(proj)

        state["check_idx"] += 1

        # Update sharpness / projection plots
        xs_chk = np.arange(1, len(sharp_history) + 1)
        H["sharp_line"].set_data(xs_chk, sharp_history)
        H["proj_line"].set_data(xs_chk, proj_history)

        # Dynamic y-limits
        thr = 2.0 / cfg.lr
        if len(sharp_history) > 0 and np.isfinite(sharp_history).all():
            ax_sharp.set_ylim(0.0, max(max(sharp_history), thr) * 1.1)
        if len(proj_history) > 0 and np.isfinite(proj_history).all():
            ax_proj.set_ylim(0.0, max(proj_history) * 1.2 + 1e-6)

    def update(_frame_idx: int):
        # ---- Train a chunk ----
        last_loss = train_k_steps(
            model, criterion, X_train_t, y_train_t,
            cfg.lr, cfg.steps_per_frame, loss_history, deltas
        )
        state["step"] += cfg.steps_per_frame

        # ---- Model prediction on visualization grid ----
        with torch.no_grad():
            y_pred_vis = model(X_vis_t).squeeze(1)   # (P,)

        # ---- Generalization error over the full visualization range ----
        # MSE between true (noise-free) function and current prediction on X_vis
        gen_mse = torch.mean((y_pred_vis - y_true_vis_t) ** 2).item()
        gen_history.append(gen_mse)

        # ---- Update function panel ----
        if cfg.input_dim == 1:
            H["line_pred"].set_data(
                X_vis[:, 0],
                y_pred_vis.detach().cpu().numpy()
            )
            H["info_text"].set_text(
                f"Step: {state['step']} / {cfg.total_steps}\n"
                f"Train loss: {last_loss:.6f}\n"
                f"Gen MSE (full range): {gen_mse:.6f}"
            )
        else:
            # Replace the learned surface Z values safely
            Yp = y_pred_vis.detach().cpu().numpy().reshape(m, m)
            X1, X2 = H["surf_grid"]

            sp = H.get("surf_pred", None)
            if sp is not None:
                try:
                    sp.remove()
                except Exception:
                    pass

            H["surf_pred"] = ax_fun.plot_surface(
                X1, X2, Yp,
                alpha=0.60, linewidth=0, antialiased=True
            )
            H["info_text"].set_text(
                f"Step: {state['step']} / {cfg.total_steps}\n"
                f"Train loss: {last_loss:.6f}\n"
                f"Gen MSE (full range): {gen_mse:.6f}"
            )

        # ---- Update loss & generalization curves ----
        xs_steps = np.arange(1, len(loss_history) + 1)
        H["loss_line"].set_data(xs_steps, loss_history)

        gen_xs = np.arange(1, len(gen_history) + 1) * cfg.steps_per_frame
        H["gen_line"].set_data(gen_xs, gen_history)

        # Dynamic y-limit combining both (ignore NaNs)
        combined = [v for v in loss_history if np.isfinite(v)] + \
                   [v for v in gen_history if np.isfinite(v)]
        if combined:
            ymin, ymax = float(np.min(combined)), float(np.max(combined))
            pad = 0.05 * (ymax - ymin + 1e-8)
            ax_loss.set_ylim(max(0.0, ymin - pad), ymax + pad)

        # ---- Curvature & projection (intermittent) ----
        maybe_update_curvature_and_projection()

        # Return artists for blitting
        to_return = [
            H.get("line_pred"),
            H["loss_line"], H["gen_line"],
            H["sharp_line"], H["proj_line"],
            H["info_text"],
        ]
        return tuple(a for a in to_return if a is not None)

    # Blitting works well for 2D line plots; for 3D surfaces it's safer to disable
    use_blit = (cfg.input_dim == 1)

    anim = FuncAnimation(
        fig, update, init_func=init,
        frames=total_frames, interval=60, blit=use_blit
    )
    return anim, loss_history

# def make_anim(
#     cfg: Config,
#     model: nn.Module,
#     x_train_t: torch.Tensor,
#     y_train_t: torch.Tensor,
#     x_vis: np.ndarray,
#     y_true_vis: np.ndarray,
#     criterion: nn.Module,   # <--- added
# ) -> Tuple[FuncAnimation, List[float]]:
#     """
#     - Trains in chunks of cfg.steps_per_frame per frame.
#     - Updates function fit & loss every frame.
#     - Tracks sharpness and |Δθ·v_max| at cfg.sharpness_stride.
#     - Computes "generalization error" = MSE on x outside [x_min, x_max].
#     """
#     loss_history: List[float] = []
#     gen_history:  List[float] = []   # <--- new: out-of-range MSE
#     deltas: List[torch.Tensor] = []
#     sharpness: List[float] = []
#     projections: List[float] = []

#     fig, axes, lines, info_text = init_figure(cfg, x_vis, y_true_vis, x_train_t, y_train_t)
#     ax_fun, ax_loss, ax_sharp, ax_proj = axes
#     line_pred, loss_line, gen_line, sharp_line, proj_line = lines

#     device = next(model.parameters()).device
#     x_vis_t = torch.from_numpy(x_vis).unsqueeze(1).to(device)
#     y_true_vis_t = torch.from_numpy(y_true_vis).to(device)

#     # Build out-of-range mask once (numpy -> indices)
#     out_mask_np = (x_vis < cfg.x_min) | (x_vis > cfg.x_max)
#     out_idx = np.where(out_mask_np)[0]
#     # If nothing is outside (e.g., vis==train), avoid errors
#     has_outside = out_idx.size > 0
#     if has_outside:
#         out_idx_t = torch.from_numpy(out_idx).to(device)

#     total_frames = cfg.total_steps // cfg.steps_per_frame
#     state = {"step": 0, "check_idx": 0, "last_v": None}

#     def init():
#         line_pred.set_data([], [])
#         loss_line.set_data([], [])
#         gen_line.set_data([], [])
#         sharp_line.set_data([], [])
#         proj_line.set_data([], [])
#         info_text.set_text("")
#         return line_pred, loss_line, gen_line, sharp_line, proj_line, info_text

#     def maybe_update_curvature_and_projection():
#         if state["step"] % cfg.sharpness_stride != 0 or state["step"] == 0:
#             return
#         lam, v = top_hessian_eigenpair(
#             model, criterion, x_train_t, y_train_t,
#             iters=cfg.power_iters, init_vec=state["last_v"]
#         )
#         state["last_v"] = v
#         sharpness.append(lam)

#         if len(deltas) > 0:
#             d = deltas[-1].to(v.device)
#             projections.append(float(torch.abs((d * v).sum()).item()))
#         else:
#             projections.append(0.0)

#         state["check_idx"] += 1

#         ax_sharp.set_ylim(0.0, max(sharpness + [2.0 / cfg.lr]) * 1.1)
#         ax_proj.set_ylim(0.0, max(projections) * 1.2 + 1e-6)

#         xs_chk = np.arange(1, len(sharpness) + 1)
#         sharp_line.set_data(xs_chk, sharpness)
#         proj_line.set_data(xs_chk, projections)

#     def update(_frame_idx: int):
#         # Train a chunk
#         last_loss = train_k_steps(
#             model, criterion, x_train_t, y_train_t,
#             cfg.lr, cfg.steps_per_frame, loss_history, deltas
#         )
#         state["step"] += cfg.steps_per_frame

#         # Function fit on x_vis
#         with torch.no_grad():
#             y_pred_vis = model(x_vis_t).squeeze(1)

#         # Generalization error (MSE) on out-of-range region only
#         if has_outside:
#             with torch.no_grad():
#                 y_pred_out = y_pred_vis.index_select(0, out_idx_t)
#                 y_true_out = y_true_vis_t.index_select(0, out_idx_t)
#                 gen_mse = torch.mean((y_pred_out - y_true_out) ** 2).item()
#         else:
#             gen_mse = float("nan")  # if no outside region, record NaN

#         gen_history.append(gen_mse)

#         # --- Update top-left plot ---
#         line_pred.set_data(x_vis, y_pred_vis.detach().cpu().numpy())
#         info_text.set_text(f"Step: {state['step']} / {cfg.total_steps}\n"
#                            f"Train loss: {last_loss:.6f}\n"
#                            f"Gen error: {gen_mse:.6f}")

#         # --- Update loss + gen curves (top-right) ---
#         xs = np.arange(1, len(loss_history) + 1)
#         loss_line.set_data(xs, loss_history)

#         # gen_history is collected per frame as well; match x-axis to steps
#         gen_xs = np.arange(1, len(gen_history) + 1) * cfg.steps_per_frame
#         gen_line.set_data(gen_xs, gen_history)

#         # Dynamic y-limits considering BOTH curves (ignore NaNs)
#         combined = [v for v in loss_history if np.isfinite(v)] + \
#                    [v for v in gen_history if np.isfinite(v)]
#         if len(combined) > 0:
#             ymin, ymax = float(np.min(combined)), float(np.max(combined))
#             pad = 0.05 * (ymax - ymin + 1e-8)
#             ax_loss.set_ylim(max(0.0, ymin - pad), ymax + pad)

#         # --- Bottom row: curvature + projection ---
#         maybe_update_curvature_and_projection()

#         return line_pred, loss_line, gen_line, sharp_line, proj_line, info_text

#     anim = FuncAnimation(fig, update, init_func=init,
#                          frames=total_frames, interval=60, blit=True)
#     return anim, loss_history

def save_with_progress(anim: FuncAnimation, cfg: Config) -> None:
    if cfg.ffmpeg_path:
        import matplotlib as mpl
        mpl.rcParams["animation.ffmpeg_path"] = cfg.ffmpeg_path
    total_frames = anim.save_count if hasattr(anim, "save_count") else 500
    pbar = tqdm(total=total_frames, desc="Rendering frames")
    def progress_callback(frame_number: int, total: int):
        pbar.update(1)
        if frame_number == total - 1:
            pbar.close()
    try:
        writer = FFMpegWriter(fps=cfg.fps_mp4, bitrate=cfg.bitrate_mp4)
        anim.save(cfg.out_mp4, writer=writer, progress_callback=progress_callback)
        print(f"[OK] Saved {cfg.out_mp4}")
    except Exception as e:
        print(f"[WARN] FFMpeg failed ({e}); falling back to GIF...")
        writer = PillowWriter(fps=cfg.fps_gif)
        anim.save(cfg.out_gif, writer=writer, progress_callback=progress_callback)
        print(f"[OK] Saved {cfg.out_gif}")
