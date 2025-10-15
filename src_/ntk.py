# sinfit/ntk.py
from __future__ import annotations
from typing import List, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn

Tensor = torch.Tensor

def layer_param_groups(model: nn.Module) -> List[List[nn.Parameter]]:
    """
    Return parameter groups per 'layer' (here: per nn.Linear module).
    Adjust if you change the architecture.
    """
    groups: List[List[nn.Parameter]] = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            ps = [p for p in m.parameters() if p.requires_grad]
            if ps:
                groups.append(ps)
    return groups

def _flatten_params(params: Sequence[Tensor]) -> Tensor:
    return torch.cat([p.reshape(-1) for p in params])

@torch.no_grad()
def _device_of(model: nn.Module) -> torch.device:
    return next(model.parameters()).device

def compute_ntk_per_layer(
    model: nn.Module,
    X_probe: Tensor,            # (M, D) on the same device as model
) -> List[Tensor]:
    """
    Compute per-layer NTKs: for each layer ℓ, K_ℓ[i,j] = ∂f(x_i)/∂θ_ℓ ⋅ ∂f(x_j)/∂θ_ℓ
    Args:
        model: scalar-output network
        X_probe: (M, D) tensor
    Returns:
        list of K_ℓ tensors, each (M, M) on CPU (for plotting)
    Notes:
        - Cost ~ O(M * P_ℓ) to build Jacobians G_ℓ, and K_ℓ = G_ℓ G_ℓ^T
        - Keep M small (e.g., 16–64) and model moderate.
    """
    device = _device_of(model)
    model.eval()

    groups = layer_param_groups(model)  # list of param-lists per layer
    M = X_probe.shape[0]

    # Build Jacobian per group: G_l in R^{M x P_l}
    K_layers: List[Tensor] = []
    for ps in groups:
        P_l = sum(p.numel() for p in ps)
        G = torch.zeros(M, P_l, device=device)

        # Compute gradient wrt this group's params for each probe point
        # We don't need higher-order grads; do per-sample forward/backward
        idx = 0
        for i in range(M):
            x_i = X_probe[i:i+1]            # (1, D)
            y_i = model(x_i).squeeze()      # scalar
            grads = torch.autograd.grad(y_i, ps, retain_graph=True, allow_unused=False)
            g_flat = torch.cat([g.reshape(-1) for g in grads])
            G[i, :] = g_flat

        # NTK for this group: K = G G^T
        K = G @ G.t()         # (M, M)
        K_layers.append(K.detach().cpu())

    return K_layers

def sample_ntk_probes_1d(
    x_min: float, x_max: float, M: int, device: torch.device
) -> Tensor:
    xs = torch.linspace(x_min, x_max, M, device=device, dtype=torch.float32)
    return xs.unsqueeze(1)  # (M,1)

def sample_ntk_probes_2d(
    x1_min: float, x1_max: float,
    x2_min: float, x2_max: float,
    M_total: int, device: torch.device
) -> Tensor:
    m = int(np.sqrt(max(1, M_total)))
    x1 = torch.linspace(x1_min, x1_max, m, device=device, dtype=torch.float32)
    x2 = torch.linspace(x2_min, x2_max, m, device=device, dtype=torch.float32)
    X1, X2 = torch.meshgrid(x1, x2, indexing="xy")
    X = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=1)  # (m*m, 2)
    return X
