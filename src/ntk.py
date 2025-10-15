from __future__ import annotations
import torch
import torch.nn as nn

@torch.enable_grad()
def compute_ntk_full(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """
    1D NTK for scalar output: K_ij = <∂f(x_i)/∂θ, ∂f(x_j)/∂θ>.
    Returns K: (M,M) on CPU.
    """
    device = next(model.parameters()).device
    M = X.shape[0]
    params = [p for p in model.parameters() if p.requires_grad]

    J = []
    for i in range(M):
        xi = X[i:i+1].to(device)     # (1,1)
        yi = model(xi)               # (1,1)
        grad = torch.autograd.grad(yi, params, retain_graph=True, create_graph=False, allow_unused=False)
        gi = torch.cat([g.reshape(-1) for g in grad])  # (P,)
        J.append(gi.detach())

    J = torch.stack(J, dim=0)  # (M,P)
    K = J @ J.t()              # (M,M)
    return K.detach().cpu()

def topk_eigs_from_ntk(K: torch.Tensor, k: int):
    # K is (M,M) CPU tensor. Use symmetric eigvals in descending order.
    import numpy as np
    Knp = K.numpy()
    w = np.linalg.eigvalsh((Knp + Knp.T) * 0.5)
    w = w[::-1]
    if k <= 0:
        return w
    return w[:min(k, len(w))]
