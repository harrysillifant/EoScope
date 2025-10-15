from __future__ import annotations
from typing import Callable, Tuple
import importlib
import numpy as np
import torch
from .config import Config
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.nn as nn
from matplotlib import pyplot as plt
from IPython.display import clear_output
from tqdm import tqdm


def set_seeds(seed: int):
    import random
    import os

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _target_sin(x: np.ndarray) -> np.ndarray:
    return (
        np.sin(x).astype(np.float32)
        + np.sin(2 * x).astype(np.float32)
        + np.sin(4 * x).astype(np.float32)
    )


def _target_poly(x: np.ndarray, a=0.0, b=0.0, c=1.0, d=0.0) -> np.ndarray:
    return (a * x**3 + b * x**2 + c * x + d).astype(np.float32)


def _resolve_custom(path: str) -> Callable[[np.ndarray], np.ndarray]:
    if ":" not in path:
        raise ValueError("custom function path must be 'package.module:func_name'")
    m, f = path.split(":", 1)
    mod = importlib.import_module(m)
    fn = getattr(mod, f)
    if not callable(fn):
        raise ValueError(f"{path} is not callable")
    return fn


def resolve_target_fn(cfg: Config, override_fn=None):
    if override_fn is not None:
        return override_fn, {}

    name = (cfg.function_name or "sin").lower()
    params = cfg.function_params or {}
    if name == "sin":
        return _target_sin, {}
    if name == "poly":

        def fn(x: np.ndarray):
            return _target_poly(x, **params)

        return fn, params
    if name == "custom":
        assert cfg.custom_loss is None  # unrelated; just for clarity
        path = params.get("path", None)
        if path is None:
            raise ValueError(
                "For function_name='custom', set function_params={'path': 'module:func'}"
            )
        return _resolve_custom(path), {"path": path}
    raise ValueError(f"unknown function_name: {cfg.function_name}")


def generate_data(
    cfg: Config, target_fn: Callable[[np.ndarray], np.ndarray]
) -> Tuple[torch.Tensor, torch.Tensor]:
    # training x in [x_min, x_max]
    x = np.random.uniform(cfg.x_min, cfg.x_max, size=cfg.n_samples).astype(np.float32)
    X = x[:, None]  # (N,1)
    y_clean = target_fn(X).astype(np.float32)
    noise = (cfg.noise_std * np.random.randn(cfg.n_samples)).astype(np.float32)
    y = y_clean + noise[:, None]
    return torch.from_numpy(X), torch.from_numpy(y)


def build_vis_grid(cfg: Config, target_fn: Callable[[np.ndarray], np.ndarray]):
    span = cfg.x_max - cfg.x_min
    lo = cfg.x_min - cfg.vis_pad_frac * span
    hi = cfg.x_max + cfg.vis_pad_frac * span
    Xv = np.linspace(lo, hi, cfg.n_plot_points, dtype=np.float32)[:, None]
    y_true = target_fn(Xv).astype(np.float32)
    return Xv, y_true

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
