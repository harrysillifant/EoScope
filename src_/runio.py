# sinfit/runio.py
from __future__ import annotations
import json
import os
import re
from dataclasses import asdict
from typing import Optional
from .config import Config

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _next_index_name(root: str, base: str) -> str:
    """
    Find the next available numbered folder:
      base_001, base_002, ...
    If none exist, returns base_001.
    """
    _ensure_dir(root)
    pattern = re.compile(rf"^{re.escape(base)}_(\d+)$")
    max_idx = 0
    for name in os.listdir(root):
        m = pattern.match(name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return f"{base}_{max_idx + 1:03d}"

def prepare_run_dir(cfg: Config) -> str:
    """
    Create and return a unique directory for this run inside cfg.runs_root.
    If cfg.project_name is None -> base='run', else base=slug(project_name).
    """
    base = _slugify(cfg.project_name) if cfg.project_name else "run"
    run_name = _next_index_name(cfg.runs_root, base)
    run_dir = os.path.join(cfg.runs_root, run_name)
    _ensure_dir(run_dir)

    # Set output paths on cfg
    cfg.run_dir = run_dir
    cfg.out_mp4 = os.path.join(run_dir, "animation.mp4")
    cfg.out_gif = os.path.join(run_dir, "animation.gif")
    cfg.ntk_out_mp4 = os.path.join(run_dir, "ntk_animation.mp4")
    cfg.ntk_out_gif = os.path.join(run_dir, "ntk_animation.gif")
    cfg.metrics_csv      = os.path.join(run_dir, "metrics.csv")
    cfg.offline_anim_mp4 = os.path.join(run_dir, "offline_ntk_anim.mp4")
    cfg.offline_anim_gif = os.path.join(run_dir, "offline_ntk_anim.gif")
    cfg.fits_path = os.path.join(run_dir, "fits.dat")
    cfg.fits_meta_path = os.path.join(run_dir, "fits_meta.json")
    cfg.fits_x_path = os.path.join(run_dir, "fits_grid.npy")

    # Save config.json (pretty-printed)
    save_config(cfg)

    return run_dir

def save_config(cfg: Config) -> None:
    assert cfg.run_dir is not None, "run_dir must be set before saving config"
    path = os.path.join(cfg.run_dir, "config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

def _slugify(name: str) -> str:
    # simple slug: lowercase, alnum + dashes/underscores
    s = name.strip().lower()
    s = re.sub(r"[^\w\-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "run"
