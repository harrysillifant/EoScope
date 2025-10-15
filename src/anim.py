from __future__ import annotations
from .config import Config
from typing import Callable, Optional, List
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

def _load_run(run_dir: str):
    meta = json.load(open(os.path.join(run_dir, "fits_meta.json"), "r"))
    grid_file  = os.path.join(run_dir, meta["grid_file"])       if meta.get("grid_file") else None
    train_file = os.path.join(run_dir, meta["train_points_file"]) if meta.get("train_points_file") else None
    fits_path  = os.path.join(run_dir, "fits.dat")
    shape = tuple(meta["shape"])
    fits  = np.memmap(fits_path, mode="r", dtype=meta["dtype"], shape=shape)
    X_fit = np.load(grid_file) if grid_file else None
    train = np.load(train_file, allow_pickle=True).item() if train_file else None
    df    = pd.read_csv(os.path.join(run_dir, "metrics.csv"))
    schedule: List[int] = meta.get("schedule", list(df["epoch"].values))
    return X_fit, fits, train, df, meta, schedule

def make_anim(
    run_dir: str,
    true_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    out_mp4: Optional[str] = None,
    out_gif: Optional[str] = None,
    fps: int = 30,
    bitrate: int = 2400,
    eig_log10: bool = True,
    cfg: Optional[Config] = None,
):
    """
    Build a 2x3 animation from logs (no autograd needed).

    Layout:
      Row 1: (0,0) Function fit   | (0,1) Train loss & Gen MSE | (0,2) NTK eigvals
      Row 2: (1,0) Sharpness (λ_max) + dashed 2/η line         | (1,1) Projections | (1,2) spacer
    """
    X_fit, fits, train, df, meta, schedule = _load_run(run_dir)
    assert X_fit is not None and fits is not None
    T, P = fits.shape

    # figure & axes via GridSpec (2 rows x 3 cols)
    fig = plt.figure(figsize=(16, 12), dpi=120, constrained_layout=True)
    gs = GridSpec(3, 3, figure=fig, width_ratios=[2.2, 1.6, 1.6])

    ax_fit   = fig.add_subplot(gs[0, 0])
    ax_loss  = fig.add_subplot(gs[0, 1])
    ax_eigs  = fig.add_subplot(gs[0, 2])
    ax_sharp = fig.add_subplot(gs[1, 0])
    ax_proj  = fig.add_subplot(gs[1, 1])
    ax_spec  = fig.add_subplot(gs[1, 2]) 
    # gs[1,2] left unused intentionally (spacer)

    # --- (0,0) function fit panel ---
    if train is not None:
        ax_fit.scatter(train["X_train"][:,0], train["y_train"][:,0], s=12, c="k", alpha=0.75, label="train pts")
    (line_pred,) = ax_fit.plot([], [], lw=2, label="f̂(x)")
    if true_fn is not None:
        y_true = true_fn(X_fit).astype(np.float32)
        ax_fit.plot(X_fit[:,0], y_true[:,0], ls="--", lw=2, label="true f(x)")
    ax_fit.set_title("Function fit over time"); ax_fit.set_xlabel("x"); ax_fit.set_ylabel("y")
    ax_fit.grid(True, alpha=0.3); ax_fit.legend(loc="best")

    # --- (0,1) loss & generalization ---
    (loss_line,) = ax_loss.plot([], [], lw=2, label="train loss")
    (gen_line,)  = ax_loss.plot([], [], lw=2, ls=":", label="gen MSE (full range)")
    ax_loss.set_title("Loss & Generalization"); ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("MSE")
    ax_loss.grid(True, alpha=0.3); ax_loss.legend(loc="best")

    # --- (0,2) eigenvalues ---
    eig_cols = [c for c in df.columns if c.startswith("eig_")]
    eig_cols.sort(key=lambda s: int(s.split("_")[1]))
    eig_lines = []
    for c in eig_cols:
        (l,) = ax_eigs.plot([], [], lw=2, label=c)
        eig_lines.append(l)
    ax_eigs.set_title("Top-k NTK eigenvalues" + (" (log10)" if eig_log10 else ""))
    ax_eigs.set_xlabel("epoch"); ax_eigs.set_ylabel("log10(λ)" if eig_log10 else "λ")
    ax_eigs.grid(True, alpha=0.3)
    if eig_cols:
        ax_eigs.legend(loc="best", ncol=min(3, len(eig_cols)))

    # --- (1,0) sharpness with 2/η threshold line ---
    (sharp_line,) = ax_sharp.plot([], [], lw=2, label="λ_max")
    (sharp2_line,) = ax_sharp.plot([], [], lw=2, ls="--", label="λ_2")
    lr = cfg.lr
    if lr is not None and np.isfinite(lr) and lr > 0:
        thr = 2.0 / lr
        ax_sharp.axhline(thr, ls="--", lw=1.5, color="tab:red", label=f"2/η = {thr:.3g}")
    ax_sharp.set_title("Sharpness (λ_max)"); ax_sharp.set_xlabel("epoch"); ax_sharp.set_ylabel("λ_max")
    ax_sharp.grid(True, alpha=0.3)
    ax_sharp.legend(loc="best")

    # --- (1,1) projections (separate plot) ---
    (proj_step_line,) = ax_proj.plot([], [], lw=2, ls="-",  color="tab:orange", label="|Δθ · v| (step)")
    (proj_cum_line,)  = ax_proj.plot([], [], lw=2, ls="--", color="tab:orange", alpha=0.9, label="|Δθ · v| (cum)")
    ax_proj.set_title("Projection along top Hessian eigenvector"); ax_proj.set_xlabel("epoch"); ax_proj.set_ylabel("magnitude")
    ax_proj.grid(True, alpha=0.3)
    ax_proj.legend(loc="best")

    # --- (1,2) spectral norms per layer ---
    spec_cols = [c for c in df.columns if c.startswith("spec_")]
    spec_cols.sort(key=lambda s: int(s.split("_")[1]))
    spec_lines = []
    for c in spec_cols:
        (l,) = ax_spec.plot([], [], lw=2, label=c)  # one line per layer
        spec_lines.append(l)
    ax_spec.set_title("Layer spectral norms ‖W‖₂")
    ax_spec.set_xlabel("epoch"); ax_spec.set_ylabel("spectral norm")
    ax_spec.grid(True, alpha=0.3)
    if spec_cols:
        ax_spec.legend(loc="best", ncol=min(3, len(spec_cols)))

    # Align df rows to schedule
    df_sched = pd.DataFrame({"epoch": schedule})
    dfm = df_sched.merge(df, on="epoch", how="left").fillna(method="ffill")

    epochs = dfm["epoch"].values.astype(int)
    loss   = dfm["train_loss"].values
    gen    = dfm["gen_mse"].values
    sharp  = dfm["sharpness"].values
    sharp2  = dfm["sharpness_2"].values if "sharpness_2" in dfm.columns else np.full_like(sharp, np.nan)
    proj_step = dfm["proj_step"].values if "proj_step" in dfm.columns else np.full_like(sharp, np.nan)
    proj_cum  = dfm["proj_cum"].values  if "proj_cum"  in dfm.columns else np.full_like(sharp, np.nan)
    eig_mat = dfm[eig_cols].values if eig_cols else np.zeros((len(dfm), 0))
    spec_mat = dfm[spec_cols].values if spec_cols else np.zeros((len(dfm), 0))

    # function panel y-limits from diverse samples
    y_samples = [np.array(fits[0]), np.array(fits[min(T-1, T//2)]), np.array(fits[T-1])]
    if train is not None: y_samples.append(train["y_train"][:,0])
    if true_fn is not None: y_samples.append(y_true[:,0])
    y_all = np.concatenate([ys.ravel() for ys in y_samples if ys is not None])
    y0, y1 = float(np.nanmin(y_all)), float(np.nanmax(y_all))
    pad = 0.08 * (y1 - y0 + 1e-12)
    ax_fit.set_xlim(float(X_fit[:,0].min()), float(X_fit[:,0].max()))
    ax_fit.set_ylim(y0 - pad, y1 + pad)

    def init():
        line_pred.set_data([], [])
        loss_line.set_data([], [])
        gen_line.set_data([], [])
        sharp_line.set_data([], [])
        sharp2_line.set_data([], [])
        proj_step_line.set_data([], [])
        proj_cum_line.set_data([], [])
        for l in eig_lines:
            l.set_data([], [])
        for l in spec_lines:
            l.set_data([], [])
        return (line_pred, loss_line, gen_line, sharp_line, sharp2_line, proj_step_line, proj_cum_line, *eig_lines, *spec_lines)

    def update(t: int):
        ep = epochs[:t+1]

        # function fit
        y_pred = np.array(fits[t])
        line_pred.set_data(X_fit[:,0], y_pred)

        # loss/gen
        loss_line.set_data(ep, loss[:t+1])
        gen_line.set_data(ep, gen[:t+1])
        vals = np.concatenate([loss[:t+1], gen[:t+1]])
        vals = vals[np.isfinite(vals)]
        if vals.size:
            lo, hi = float(vals.min()), float(vals.max())
            pad = 0.08 * (hi - lo + 1e-12)
            ax_loss.set_xlim(epochs[0], epochs[-1])
            ax_loss.set_ylim(max(0.0, lo - pad), hi + pad)

        # sharpness
        sharp_line.set_data(ep,  sharp[:t+1])
        sharp2_line.set_data(ep, sharp2[:t+1])
        svals = np.concatenate([sharp[:t+1], sharp2[:t+1]])
        svals = svals[np.isfinite(svals)]
        if svals.size:
            lo, hi = float(np.nanmin(svals)), float(np.nanmax(svals))
            pad = 0.08 * (hi - lo + 1e-12)
            ax_sharp.set_xlim(epochs[0], epochs[-1])
            ax_sharp.set_ylim(max(0.0, lo - pad), hi + pad)

        # projections (separate panel)
        proj_step_line.set_data(ep, proj_step[:t+1])
        proj_cum_line.set_data(ep,  proj_cum[:t+1])
        pvals = np.concatenate([proj_step[:t+1], proj_cum[:t+1]])
        pvals = pvals[np.isfinite(pvals)]
        if pvals.size:
            lo, hi = float(np.nanmin(pvals)), float(np.nanmax(pvals))
            pad = 0.08 * (hi - lo + 1e-12)
            ax_proj.set_xlim(epochs[0], epochs[-1])
            ax_proj.set_ylim(max(0.0, lo - pad), hi + pad)

        # eigenvalues
        if eig_cols:
            for j, l in enumerate(eig_lines):
                y = eig_mat[:t+1, j]
                if eig_log10:
                    y = np.log10(np.clip(y, 1e-12, None))
                l.set_data(ep, y)
            Y = np.concatenate([
                (np.log10(np.clip(eig_mat[:t+1, j], 1e-12, None)) if eig_log10 else eig_mat[:t+1, j])
                for j in range(eig_mat.shape[1])
            ]) if eig_mat.size else np.array([])
            if Y.size:
                lo, hi = float(np.nanmin(Y)), float(np.nanmax(Y))
                pad = 0.08 * (hi - lo + 1e-12)
                ax_eigs.set_xlim(epochs[0], epochs[-1])
                ax_eigs.set_ylim(lo - pad, hi + pad)

        # spectral norms
        if spec_cols:
            for j, l in enumerate(spec_lines):
                y = spec_mat[:t+1, j]
                l.set_data(ep, y)
            Y = spec_mat[:t+1, :].ravel() if spec_mat.size else np.array([])
            Y = Y[np.isfinite(Y)]
            if Y.size:
                lo, hi = float(np.nanmin(Y)), float(np.nanmax(Y))
                pad = 0.08 * (hi - lo + 1e-12)
                ax_spec.set_xlim(epochs[0], epochs[-1])
                ax_spec.set_ylim(max(0.0, lo - pad), hi + pad)

        return (line_pred, loss_line, gen_line, sharp_line, sharp2_line, proj_step_line, proj_cum_line, *eig_lines, *spec_lines)

    anim = FuncAnimation(fig, update, init_func=init, frames=T, interval=60, blit=False)

    if out_mp4 or out_gif:
        total = T
        pbar = tqdm(total=total, desc="Rendering animation")
        def cb(frame_number: int, total: int):
            pbar.update(1)
            if frame_number == total - 1:
                pbar.close()
        try:
            if out_mp4:
                anim.save(out_mp4, writer=FFMpegWriter(fps=fps, bitrate=bitrate), progress_callback=cb)
                print(f"[OK] saved {out_mp4}")
            elif out_gif:
                anim.save(out_gif, writer=PillowWriter(fps=fps), progress_callback=cb)
                print(f"[OK] saved {out_gif}")
        except Exception as e:
            print(f"[WARN] FFMpeg failed ({e}); falling back to GIF...")
            if out_gif:
                anim.save(out_gif, writer=PillowWriter(fps=fps), progress_callback=cb)
                print(f"[OK] saved {out_gif}")

    return anim
