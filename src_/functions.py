# sinfit/functions.py
from __future__ import annotations
from typing import Callable, Dict
import numpy as np

# Each function maps np.ndarray -> np.ndarray (same shape)
def f_sin(x: np.ndarray, **kw) -> np.ndarray:
    return np.sin(x)

def f_cos(x: np.ndarray, **kw) -> np.ndarray:
    return np.cos(x)

def f_sinc(x: np.ndarray, **kw) -> np.ndarray:
    # normalized sinc: sin(pi x)/(pi x)
    return np.sinc(x / np.pi)

def f_poly(x: np.ndarray, **kw) -> np.ndarray:
    # Example polynomial: ax^3 + bx^2 + cx + d; defaults below
    a = kw.get("a", 0.0)
    b = kw.get("b", 0.0)
    c = kw.get("c", 1.0)
    d = kw.get("d", 0.0)
    return a * x**3 + b * x**2 + c * x + d

def f_square_wave(x: np.ndarray, **kw) -> np.ndarray:
    # simple square wave with period 2π
    period = kw.get("period", 2*np.pi)
    phase  = kw.get("phase", 0.0)
    w = 2*np.pi/period
    return np.sign(np.sin(w * (x - phase)))

def f_sawtooth(x: np.ndarray, **kw) -> np.ndarray:
    # sawtooth with period 2π in [-1,1]
    period = kw.get("period", 2*np.pi)
    phase  = kw.get("phase", 0.0)
    w = 2*np.pi/period
    t = (w * (x - phase)) / (2*np.pi)  # wrap to [0,1)
    frac = t - np.floor(t)
    return 2.0 * (frac - 0.5)

# add 2-D variants; accept x as (N,) or (N,1) (1-D) OR (N,2) (2-D)

def f_sin_cos2d(x: np.ndarray, **kw) -> np.ndarray:
    # f(x,y) = sin(x) + 0.5*cos(y)
    x = np.asarray(x)
    if x.ndim == 1 or (x.ndim==2 and x.shape[1]==1):
        return np.sin(x.squeeze())
    assert x.shape[1] == 2, "f_sin_cos2d expects (N,2) for 2-D"
    return np.sin(x[:,0]) + 0.5*np.cos(x[:,1])

def f_rbf2d(x: np.ndarray, **kw) -> np.ndarray:
    # Gaussian bump centered at (cx,cy) with width s
    cx = kw.get("cx", 0.0); cy = kw.get("cy", 0.0); s = kw.get("s", 1.5)
    x = np.asarray(x)
    if x.ndim == 1 or (x.ndim==2 and x.shape[1]==1):
        return np.exp(-(x.squeeze()**2)/(2*s*s))
    assert x.shape[1] == 2
    dx = x[:,0]-cx; dy = x[:,1]-cy
    return np.exp(-(dx*dx+dy*dy)/(2*s*s))

# Registry: add your own easily
FUNCTIONS: Dict[str, Callable[..., np.ndarray]] = {
    "sin": f_sin,
    "cos": f_cos,
    "sinc": f_sinc,
    "poly": f_poly,
    "square": f_square_wave,
    "sawtooth": f_sawtooth,
    "sin_cos2d": f_sin_cos2d,
    "rbf2d":     f_rbf2d,
}
