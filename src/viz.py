import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os


def make_live_animation(
    history, results_dir="results", fname="training_metrics.mp4", fps=4
):
    """history: dict with keys 'train_loss', 'test_loss', 'sharpness', 'layer_norms', 'ntk_eigs', 'hessian_eigs'
    - layer_norms: list of dicts per epoch
    - ntk_eigs: list of lists (per epoch top-k)
    - hessian_eigs: list of lists (per epoch top-k)
    """
    os.makedirs(results_dir, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    ax_loss = axes[0, 0]
    ax_sharp = axes[0, 1]
    ax_layer = axes[1, 0]
    ax_ntk = axes[1, 1]
    ax_hess = axes[2, 0]
    ax_empty = axes[2, 1]
    ax_empty.axis("off")

    epochs = len(history["train_loss"])

    def init():
        ax_loss.clear()
        ax_sharp.clear()
        ax_layer.clear()
        ax_ntk.clear()
        ax_hess.clear()
        ax_loss.set_title("Train & Test Loss")
        ax_loss.set_xlabel("epoch")
        ax_sharp.set_title("Sharpness (max Hessian eigenval)")
        ax_sharp.set_xlabel("epoch")
        ax_layer.set_title("Layer spectral norms")
        ax_layer.set_xlabel("epoch")
        ax_ntk.set_title("Top-k NTK eigenvalues")
        ax_ntk.set_xlabel("epoch")
        ax_hess.set_title("Top-k Hessian eigenvalues")
        ax_hess.set_xlabel("epoch")
        return []

    def update(i):
        ax_loss.clear()
        ax_sharp.clear()
        ax_layer.clear()
        ax_ntk.clear()
        ax_hess.clear()
        x = np.arange(1, i + 2)
        ax_loss.plot(x, history["train_loss"][: i + 1], label="train")
        ax_loss.plot(x, history["test_loss"][: i + 1], label="test")
        ax_loss.legend()

        ax_sharp.plot(x, history["sharpness"][: i + 1])

        # layer norms: history['layer_norms'] is a list of dicts
        layer_names = []
        if len(history["layer_norms"]) > 0:
            layer_names = list(history["layer_norms"][0].keys())
            for lname in layer_names:
                vals = [d.get(lname, np.nan) for d in history["layer_norms"][: i + 1]]
                ax_layer.plot(x, vals, label=lname)
            ax_layer.legend(fontsize=8)

        # NTK top-k
        if len(history["ntk_eigs"]) > 0:
            ntk_k = len(history["ntk_eigs"][0])
            for k in range(min(5, ntk_k)):
                vals = [
                    arr[k] if len(arr) > k else np.nan
                    for arr in history["ntk_eigs"][: i + 1]
                ]
                ax_ntk.plot(x, vals, label=f"ntk_{k}")
            ax_ntk.legend(fontsize=8)

        if len(history["hessian_eigs"]) > 0:
            h_k = len(history["hessian_eigs"][0])
            for k in range(min(5, h_k)):
                vals = [
                    arr[k] if len(arr) > k else np.nan
                    for arr in history["hessian_eigs"][: i + 1]
                ]
                ax_hess.plot(x, vals, label=f"hess_{k}")
            ax_hess.legend(fontsize=8)

        fig.suptitle(f"Epoch {i + 1}/{epochs}")
        return []

    anim = animation.FuncAnimation(
        fig, update, frames=epochs, init_func=init, blit=False
    )
    outpath = os.path.join(results_dir, fname)
    try:
        anim.save(outpath, fps=4, dpi=150)
        print("Saved animation to", outpath)
    except Exception as e:
        print(
            "Could not save mp4 (ffmpeg may be missing). Saving gif instead... Error:",
            e,
        )
        outgif = outpath.rsplit(".", 1)[0] + ".gif"
        anim.save(outgif, writer="pillow", fps=4)
        print("Saved animation to", outgif)
    plt.close(fig)
