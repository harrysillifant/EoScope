# src/offline_anim.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from typing import List, Tuple

def make_offline_anim_from_csv(
    csv_path: str,
    out_mp4: str | None = None,
    out_gif: str | None = None,
    fps: int = 30,
    bitrate: int = 2000,
    k_eigs: int | None = None,  # None -> infer from columns
    log_eigs: bool = True,
):
    df = pd.read_csv(csv_path)
    epochs = df["epoch"].values
    loss = df["train_loss"].values
    gen  = df["gen_mse"].values
    # figure out which eigen columns exist
    eig_cols = [c for c in df.columns if c.startswith("eig_")]
    eig_cols.sort(key=lambda s: int(s.split("_")[1]))
    if k_eigs is not None:
        eig_cols = eig_cols[:k_eigs]
    E = df[eig_cols].values  # (T, K)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), dpi=120, constrained_layout=True)

    # top: loss + gen
    (loss_line,) = ax1.plot([], [], lw=2, label="Train loss")
    (gen_line,)  = ax1.plot([], [], lw=2, ls=":", label="Gen MSE (full range)")
    ax1.set_title("Loss & Generalization over epochs")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("MSE")
    ax1.grid(True, alpha=0.3); ax1.legend(loc="upper right")

    # bottom: eigenvalue trajectories
    eig_lines = []
    for j, col in enumerate(eig_cols, start=1):
        (l,) = ax2.plot([], [], lw=2, label=col)
        eig_lines.append(l)
    ax2.set_title("Top-k NTK eigenvalues over epochs" + (" (log)" if log_eigs else ""))
    ax2.set_xlabel("epoch"); ax2.set_ylabel("eigenvalue")
    ax2.grid(True, alpha=0.3); ax2.legend(loc="upper right", ncol=min(3, len(eig_cols)))

    T = len(df)

    def init():
        loss_line.set_data([], [])
        gen_line.set_data([], [])
        for l in eig_lines:
            l.set_data([], [])
        return (loss_line, gen_line, *eig_lines)

    def update(t: int):
        ep = epochs[:t+1]
        loss_line.set_data(ep, loss[:t+1])
        gen_line.set_data(ep, gen[:t+1])

        # y-lims for top panel
        vals = np.concatenate([loss[:t+1], gen[:t+1]])
        vals = vals[np.isfinite(vals)]
        if vals.size:
            y0, y1 = float(vals.min()), float(vals.max())
            pad = 0.05 * (y1 - y0 + 1e-12)
            ax1.set_xlim(epochs[0], epochs[-1])
            ax1.set_ylim(max(0.0, y0 - pad), y1 + pad)

        # bottom: eigen trajectories
        for j, l in enumerate(eig_lines):
            y = E[:t+1, j]
            if log_eigs:
                y = np.log10(np.clip(y, 1e-12, None))
            l.set_data(ep, y)

        # y-lims for eigs
        Y = []
        for j in range(len(eig_lines)):
            y = E[:t+1, j]
            if log_eigs:
                y = np.log10(np.clip(y, 1e-12, None))
            Y.append(y)
        if Y:
            Ycat = np.concatenate(Y)
            y0, y1 = float(np.nanmin(Ycat)), float(np.nanmax(Ycat))
            pad = 0.05 * (y1 - y0 + 1e-12)
            ax2.set_xlim(epochs[0], epochs[-1])
            ax2.set_ylim(y0 - pad, y1 + pad)

        return (loss_line, gen_line, *eig_lines)

    anim = FuncAnimation(fig, update, init_func=init, frames=T, interval=60, blit=False)

    if out_mp4 or out_gif:
        try:
            if out_mp4:
                anim.save(out_mp4, writer=FFMpegWriter(fps=fps, bitrate=bitrate))
                print(f"[OK] saved {out_mp4}")
            elif out_gif:
                anim.save(out_gif, writer=PillowWriter(fps=fps))
                print(f"[OK] saved {out_gif}")
        except Exception as e:
            print(f"[WARN] FFMpeg failed ({e}); falling back to GIF...")
            if out_gif:
                anim.save(out_gif, writer=PillowWriter(fps=fps))
                print(f"[OK] saved {out_gif}")
    return anim
