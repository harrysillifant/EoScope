from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_run(run_dir: str):
    meta = json.load(open(os.path.join(run_dir, "fits_meta.json"), "r"))
    X_fit = np.load(os.path.join(run_dir, meta["grid_file"]))
    fits = np.memmap(os.path.join(run_dir, "fits.dat"),
                     mode="r", dtype=meta["dtype"], shape=tuple(meta["shape"]))
    train = np.load(os.path.join(run_dir, meta["train_points_file"]), allow_pickle=True).item()
    df = pd.read_csv(os.path.join(run_dir, "metrics.csv"))
    return X_fit, fits, train, df, meta

def plot_summary(run_dir: str, true_fn=None, save_path: str | None = None):
    X_fit, fits, train, df, meta = load_run(run_dir)

    # figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=120)
    ax_fit, ax_loss = axes[0,0], axes[0,1]
    ax_sharp, ax_eigs = axes[1,0], axes[1,1]

    # (0,0) Function fit: show first & last snapshot, with train points
    ax_fit.scatter(train["X_train"][:,0], train["y_train"][:,0], s=12, alpha=0.7, label="train pts")
    (line_first,) = ax_fit.plot(X_fit[:,0], fits[0], lw=2, label="f̂ (early)")
    (line_last,)  = ax_fit.plot(X_fit[:,0], fits[-1], lw=2, label="f̂ (final)")
    if true_fn is not None:
        y_true = true_fn(X_fit).astype(np.float32)
        ax_fit.plot(X_fit[:,0], y_true[:,0], ls="--", lw=2, label="true f")
    ax_fit.set_title("Function fit (first vs final snapshot)")
    ax_fit.set_xlabel("x"); ax_fit.set_ylabel("y")
    ax_fit.grid(True, alpha=0.3); ax_fit.legend(loc="best")

    # (0,1) Loss & generalization error
    ax_loss.plot(df["epoch"], df["train_loss"], lw=2, label="train loss")
    ax_loss.plot(df["epoch"], df["gen_mse"],  lw=2, ls=":", label="gen MSE (full range)")
    ax_loss.set_title("Loss & Generalization")
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("MSE")
    ax_loss.grid(True, alpha=0.3); ax_loss.legend(loc="best")

    # (1,0) Sharpness
    ax_sharp.plot(df["epoch"], df["sharpness"], lw=2)
    ax_sharp.set_title("Sharpness (top Hessian eigenvalue)")
    ax_sharp.set_xlabel("epoch"); ax_sharp.set_ylabel("λ_max")
    ax_sharp.grid(True, alpha=0.3)

    # (1,1) Top-k NTK eigenvalues (log10)
    eig_cols = [c for c in df.columns if c.startswith("eig_")]
    eig_cols.sort(key=lambda s: int(s.split("_")[1]))
    for c in eig_cols:
        y = np.log10(np.clip(df[c].values, 1e-12, None))
        ax_eigs.plot(df["epoch"], y, lw=2, label=c)
    ax_eigs.set_title("Top-k NTK eigenvalues (log10)")
    ax_eigs.set_xlabel("epoch"); ax_eigs.set_ylabel("log10(λ)")
    ax_eigs.grid(True, alpha=0.3)
    if eig_cols:
        ax_eigs.legend(loc="best", ncol=min(3, len(eig_cols)))

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"[OK] saved plot -> {save_path}")
    return fig
