from __future__ import annotations
import os
import json
import dataclasses
from .config import Config


def prepare_run_dir(cfg: Config) -> str:
    base = os.path.abspath("runs")
    os.makedirs(base, exist_ok=True)

    name = cfg.project_name or "run"
    run_dir = os.path.join(base, name)
    if os.path.exists(run_dir):
        # increment suffix
        i = 1
        while os.path.exists(f"{run_dir}_{i}"):
            i += 1
        run_dir = f"{run_dir}_{i}"
    os.makedirs(run_dir, exist_ok=True)

    cfg.run_dir = run_dir
    cfg.metrics_csv     = os.path.join(run_dir, "metrics.csv")
    cfg.fits_path       = os.path.join(run_dir, "fits.dat")
    cfg.fits_meta_path  = os.path.join(run_dir, "fits_meta.json")
    cfg.fits_x_path     = os.path.join(run_dir, "fits_grid.npy")
    cfg.train_points_path = os.path.join(run_dir, "train_points.npy")
    return run_dir

def save_config_json(cfg: Config, filename: str = "config.json") -> str:
    """
    Dump the full Config dataclass to a JSON file in the run directory.
    Returns the full path to the JSON file.
    """
    assert cfg.run_dir is not None, "Call prepare_run_dir(cfg) before save_config_json."
    path = os.path.join(cfg.run_dir, filename)
    data = dataclasses.asdict(cfg)
    # ensure deterministic key order & readable formatting
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return path