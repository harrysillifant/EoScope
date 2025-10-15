# sinfit/losses.py
from __future__ import annotations
from typing import Callable, Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn

Tensor = torch.Tensor

# ----- RHS registry for PDEs (can extend) -----
def rhs_zero(x: Tensor) -> Tensor:
    return torch.zeros_like(x)

RHS: Dict[str, Callable[[Tensor], Tensor]] = {
    "zero": rhs_zero
}

# ----- Small helper for L2 over parameters -----
def l2_norm_sq(model: nn.Module) -> Tensor:
    return sum((p**2).sum() for p in model.parameters() if p.requires_grad)

# ----- Autograd-based 1D derivatives wrt input x -----
def second_derivative_1d(model: nn.Module, x: Tensor) -> Tensor:
    """
    x: shape (N,1), requires_grad=True
    returns u_xx: shape (N,1)
    """
    x.requires_grad_(True)
    u = model(x)                           # (N,1)
    # first derivative
    (du_dx,) = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)
    # second derivative
    (d2u_dx2,) = torch.autograd.grad(du_dx, x, torch.ones_like(du_dx), create_graph=True, retain_graph=True)
    return d2u_dx2

# ----- PDE residual builders -----
def pde_residual_poisson1d(model: nn.Module, x: Tensor, g: Callable[[Tensor], Tensor]) -> Tensor:
    # u_xx - g(x) = 0  => residual = u_xx - g
    u_xx = second_derivative_1d(model, x)
    return u_xx - g(x)

def pde_residual_helmholtz1d(model: nn.Module, x: Tensor, g: Callable[[Tensor], Tensor], k: float) -> Tensor:
    # u_xx + k^2 u - g(x) = 0
    u = model(x)
    u_xx = second_derivative_1d(model, x)
    return u_xx + (k**2) * u - g(x)

# ----- Factory returning a callable criterion(model, x, y_pred, y_true) -----
def get_composite_loss(name: str, params: Dict[str, Any], device: torch.device) -> Callable[[nn.Module, Tensor, Tensor, Tensor], Tensor]:
    name = name.lower()
    base: nn.Module
    if name in ("mse", "mse_loss"):
        base = nn.MSELoss()
    elif name in ("l1", "mae", "l1_loss"):
        base = nn.L1Loss()
    else:
        raise ValueError(f"Unknown base loss: {name}")

    l2_w: float = float(params.get("l2_weight", 0.0))
    pde: Optional[str] = params.get("pde", None)
    pde_w: float = float(params.get("pde_weight", 0.0))
    n_colloc: int = int(params.get("n_collocation", 0))
    colloc_min = params.get("colloc_min", None)
    colloc_max = params.get("colloc_max", None)
    rhs_key: Optional[str] = params.get("rhs", "zero")
    rhs_fn = RHS.get(rhs_key, rhs_zero)

    helm_k: float = float(params.get("helmholtz_k", 1.0))

    def criterion(model: nn.Module, x: Tensor, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # Base data loss
        loss = base(y_pred, y_true)

        # L2 regularization on parameters
        if l2_w > 0.0:
            loss = loss + l2_w * l2_norm_sq(model)

        # PDE residual on collocation (or training) points
        if pde_w > 0.0 and pde is not None:
            # choose collocation points
            if n_colloc > 0:
                lo = float(colloc_min) if colloc_min is not None else float(x.min().item())
                hi = float(colloc_max) if colloc_max is not None else float(x.max().item())
                x_c = torch.linspace(lo, hi, n_colloc, device=device, dtype=x.dtype).unsqueeze(1)
            else:
                x_c = x

            # Residual
            if pde.lower() == "poisson1d":
                r = pde_residual_poisson1d(model, x_c, rhs_fn)
            elif pde.lower() == "helmholtz1d":
                r = pde_residual_helmholtz1d(model, x_c, rhs_fn, helm_k)
            else:
                raise ValueError(f"Unsupported PDE: {pde}")

            # Mean squared residual
            pde_term = (r**2).mean()
            loss = loss + pde_w * pde_term

        return loss

    return criterion
