# src/fit_load.py
from __future__ import annotations
import json, os
import numpy as np

def load_fits(run_dir: str):
    meta = json.load(open(os.path.join(run_dir, "fits_meta.json"), "r"))
    grid = np.load(os.path.join(run_dir, meta["grid_file"]))
    arr  = np.memmap(os.path.join(run_dir, "fits.dat"),
                     mode="r", dtype=meta["dtype"], shape=tuple(meta["shape"]))
    return grid, arr, meta
