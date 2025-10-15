# sinfit/linalg.py
from __future__ import annotations
from typing import Iterable, List, Tuple, Callable
import torch
from torch import nn
import numpy as np

# ---------- Utilities to flatten/unflatten parameter lists ----------

def _flatten_tensors(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])

def _like_params(params: Iterable[torch.Tensor], flat: torch.Tensor) -> List[torch.Tensor]:
    """Reshape 'flat' into a list with the same shapes as 'params'."""
    outs, offset = [], 0
    for p in params:
        numel = p.numel()
        outs.append(flat[offset:offset + numel].view_as(p))
        offset += numel
    return outs

def flatten_params(model: nn.Module) -> torch.Tensor:
    """Flatten current parameters into a single vector (detached CPU tensor)."""
    with torch.no_grad():
        return _flatten_tensors([p.detach().cpu() for p in model.parameters()])

# ---------- Hessian-vector product (HVP) via Pearlmutter trick ----------

def hvp(loss: torch.Tensor, params: List[torch.Tensor], v_flat: torch.Tensor) -> torch.Tensor:
    """
    Compute Hessian(loss wrt params) @ v using autograd.
    Args:
        loss: scalar loss (create_graph=True must be used when obtaining grads).
        params: list of parameter tensors (requires_grad=True).
        v_flat: flattened vector to multiply by H (on the same device as params).
    Returns:
        H @ v as a flattened tensor on the same device.
    """
    # First-order grads
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    # Dot(grads, v)
    v_list = _like_params(params, v_flat)
    grad_dot_v = torch.zeros((), device=loss.device)
    for g, v in zip(grads, v_list):
        grad_dot_v = grad_dot_v + (g * v).sum()
    # Differentiate dot(grads, v) to get HVP
    hv = torch.autograd.grad(grad_dot_v, params, retain_graph=True)
    return _flatten_tensors(hv)

# ---------- Power iteration for top eigenpair ----------

@torch.no_grad()
def _normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / (v.norm() + eps)

CriterionFn = Callable[[nn.Module, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

# ... (existing flatten/HVP helpers unchanged)

def top_hessian_eigenpair(
    model: nn.Module,
    criterion: CriterionFn,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    iters: int = 20,
    init_vec: torch.Tensor | None = None,
) -> Tuple[float, torch.Tensor]:
    device = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]

    model.zero_grad(set_to_none=True)
    y_pred = model(x_batch)
    loss = criterion(model, x_batch, y_pred, y_batch)  # <-- use composite loss

    if init_vec is None:
        v = torch.randn(sum(p.numel() for p in params), device=device)
    else:
        v = init_vec.to(device)
    v = _normalize(v)

    for _ in range(iters):
        hv = hvp(loss, params, v)
        hv_det = hv.detach()
        lam = float((v * hv_det).sum().item())
        v = _normalize(hv_det)

    hv = hvp(loss, params, v)
    lam = float((v * hv).sum().item())
    v = _normalize(v)
    return lam, v


def _proj_off(v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Project v off unit (or non-unit) vector u:  v - u (u·v)."""
    uu = u / (u.norm() + 1e-12)
    return v - uu * torch.dot(uu, v)

def second_hessian_eigenvalue_deflated(
    model: nn.Module,
    criterion: CriterionFn,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    v1: torch.Tensor,
    iters: int = 20,
    init_vec: torch.Tensor | None = None,
    max_retries: int = 5,
) -> Tuple[float, torch.Tensor]:
    """
    Power iteration on the deflated operator P H P (P = I - v1 v1^T) to estimate λ2.
    Re-orthogonalizes every step and resamples if the iterate nearly vanishes.
    Returns (lam2, v2).  v2 is ~unit and ~orthogonal to v1.
    """
    device = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]

    # Build scalar loss with graph for HVPs
    model.zero_grad(set_to_none=True)
    y_pred = model(x_batch)
    loss = criterion(model, x_batch, y_pred, y_batch)  # scalar

    # Ensure v1 is unit-norm on the right device
    v1 = v1.to(device)
    v1 = v1 / (v1.norm() + 1e-12)

    P = sum(p.numel() for p in params)

    def _safe_init() -> torch.Tensor:
        v = torch.randn(P, device=device)
        v = _proj_off(v, v1)
        n = v.norm()
        tries = 0
        while n < 1e-8 and tries < max_retries:
            v = torch.randn(P, device=device)
            v = _proj_off(v, v1)
            n = v.norm()
            tries += 1
        return v / (n + 1e-12)

    # Initialize
    v = init_vec.to(device) if init_vec is not None else _safe_init()
    v = _proj_off(v, v1)
    v = _normalize(v)

    # Power iteration with deflation and Gram–Schmidt each step
    for _ in range(iters):
        hv = hvp(loss, params, v).detach()      # H v
        hv = _proj_off(hv, v1)                  # deflate: P (Hv)
        n = hv.norm()
        if not torch.isfinite(n) or float(n.item()) < 1e-10:
            v = _safe_init()
        else:
            v = hv / (n + 1e-12)

    # Rayleigh quotient on the orthogonalized eigenvector
    hv_final = hvp(loss, params, v)
    lam2 = float(torch.dot(v, hv_final).item())

    # Guard against non-finite outputs
    if not (torch.isfinite(hv_final).all() and np.isfinite(lam2)):
        # last-chance recompute with a fresh init (no warm start)
        v = _safe_init()
        for _ in range(max(6, iters // 2)):
            hv = hvp(loss, params, v).detach()
            hv = _proj_off(hv, v1)
            n = hv.norm()
            if not torch.isfinite(n) or float(n.item()) < 1e-10:
                v = _safe_init()
            else:
                v = hv / (n + 1e-12)
        hv_final = hvp(loss, params, v)
        lam2 = float(torch.dot(v, hv_final).item())

    v = _normalize(_proj_off(v, v1))
    return lam2, v