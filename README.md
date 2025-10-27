# ReLU Net Live Metrics

This repository contains a PyTorch-based training script for a simple ReLU fully-connected network on MNIST
and tools to compute & visualize (live/video) the following metrics as training proceeds:

- Training and test loss over epochs (live plot & saved video)
- Sharpness (maximum eigenvalue of the loss Hessian)
- Layer spectral norms (operator norms of weight matrices)
- Top-k NTK eigenvalues (Neural Tangent Kernel top eigenvalues on a subset)
- Top-k Hessian eigenvalues (computed via Hessian-vector products + deflation)

**Notes & caveats**
- Computing Hessian and NTK eigenvalues is expensive. Scripts provide options to compute these on small subsets.
- Saving animations as video requires an ffmpeg writer installed on your system. If ffmpeg is not available,
  matplotlib may save as `gif` instead, or you can install ffmpeg via your package manager.
- The code is intentionally simple/clear rather than highly optimized. For large models/datasets you'll need
  to reduce the sizes used for NTK/Hessian computations or use specialized libraries (e.g., `backpack`, `functorch`, `pyhessian`).

## Quick start (example)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --epochs 10 --batch-size 128 --device cpu --ntk-subset 128 --hessian-topk 3 --ntk-topk 5
```

This will run training and produce a `results/` folder with logs and an animation `results/training_metrics.mp4` (or `.gif`).

## Repo layout
- `src/model.py` model definition (simple MLP with ReLU)
- `src/data.py` dataset loading utilities (torchvision MNIST)
- `src/utils.py` helpers: flatten params, hvp, power iterations, spectral norms, NTK computation
- `src/viz.py` live plotting/animation utilities
- `src/train.py` training loop and metric computations

See docstrings in the `src/` files for usage details.
