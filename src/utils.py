import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import math
import numpy as np
import torch.nn as nn
from typing import List


def flatten_params(model):
    params = [p for p in model.parameters() if p.requires_grad]
    vec = parameters_to_vector(params).detach()
    return vec, params


def _grad(loss, params, create_graph=False):
    grads = torch.autograd.grad(
        loss, params, create_graph=create_graph, retain_graph=True, allow_unused=True
    )
    grads = [g if g is not None else torch.zeros_like(
        p) for g, p in zip(grads, params)]
    return torch.cat([g.contiguous().view(-1) for g in grads])


def hvp(loss, params, v):
    """Compute Hessian-vector product Hv where H =
    abla^2 loss and v is a vector in parameter space.
        - loss should be a scalar (torch.Tensor)
        - params is a list of parameters
        - v is a 1D torch tensor matching flattened params
        Returns flattened Hv tensor.
    """
    grad_flat = _grad(loss, params, create_graph=True)
    # dot grad with v
    dot = torch.dot(grad_flat, v)
    hv = torch.autograd.grad(dot, params, retain_graph=True)
    hv = [h if h is not None else torch.zeros_like(
        p) for h, p in zip(hv, params)]
    return torch.cat([h.contiguous().view(-1) for h in hv]).detach()


def power_iteration_hessian_max(
    loss_fn, model, params, inputs, targets, n_iters=20, tol=1e-4, device="cpu"
):
    """Estimate top eigenvalue and eigenvector of Hessian via power iteration using hvp.
    loss_fn: callable (model, inputs, targets) -> scalar loss
    params: list(model.parameters())
    Returns (lambda_max, vec)
    """
    # initialize random vector
    flat, _ = flatten_params(model)
    v = torch.randn_like(flat).to(device)
    v /= v.norm()
    last_ray = None
    for i in range(n_iters):
        model.zero_grad()
        loss = loss_fn(model, inputs, targets)
        Hv = hvp(loss, params, v)
        ray = torch.dot(v, Hv).item()
        if last_ray is not None and abs(ray - last_ray) < tol:
            break
        last_ray = ray
        v = Hv
        nrm = v.norm()
        if nrm.item() == 0:
            break
        v = v / nrm
    # Rayleigh quotient as eigenvalue estimate
    model.zero_grad()
    loss = loss_fn(model, inputs, targets)
    Hv = hvp(loss, params, v)
    lambda_max = torch.dot(v, Hv).item()
    return lambda_max, v


def spectral_norm_matrix(mat, n_iters=20):
    # mat: 2D torch tensor
    u = torch.randn(mat.size(1), device=mat.device)
    u /= u.norm()
    for _ in range(n_iters):
        v = mat @ u
        if v.norm().item() == 0:
            break
        v = v / v.norm()
        u = mat.t() @ v
        if u.norm().item() == 0:
            break
        u = u / u.norm()
    sigma = (mat @ u).norm().item()
    return sigma


def layer_spectral_norms(model, n_iters=20):
    norms = {}
    for name, p in model.named_parameters():
        if p.ndim == 2:  # weight matrix
            norms[name] = spectral_norm_matrix(p.detach(), n_iters=n_iters)
    return norms


def compute_ntk_gram(model, inputs, output_index=None, device="cpu"):
    """Compute Gram matrix J J^T where J_{i,:} = grad_params( scalar_output_i )
    For scalar_output_i: we use the logit for `output_index` if provided, otherwise sum of logits.
    inputs: tensor of shape (m, C, H, W)
    Returns Gram matrix (m x m) on device and the flattened jacobians if needed.
    """
    model.zero_grad()
    params = [p for p in model.parameters() if p.requires_grad]
    m = inputs.size(0)
    grads = []
    for i in range(m):
        x = inputs[i: i + 1]
        logits = model(x)
        if output_index is None:
            scalar = logits.sum()
        else:
            scalar = logits[0, output_index]
        grad_vec = _grad(scalar, params, create_graph=False).detach()
        grads.append(grad_vec.unsqueeze(0))
    G = torch.cat(grads, 0)  # m x P
    gram = (G @ G.t()).cpu().numpy()
    return gram


def topk_eigvals_numpy(mat, k=5):
    # mat: numpy array
    import numpy.linalg as la

    vals, vecs = la.eigh(mat)
    vals = vals[::-1]
    return vals[:k]


def hessian_topk_via_deflation(
    loss_fn, model, params, inputs, targets, k=3, n_iters=20, device="cpu"
):
    """Compute top-k Hessian eigenvalues via repeated power iteration + deflation.
    This is a simple implementation (not optimized). Use small k and small models/datasets.
    """
    flat, _ = flatten_params(model)
    P = flat.numel()
    eigenvals = []
    eigenvecs = []

    def project_out(v, vecs):
        if len(vecs) == 0:
            return v
        vv = v.clone()
        for u in vecs:
            vv -= torch.dot(vv, u) * u
        return vv

    for ki in range(k):
        v = torch.randn(P, device=flat.device)
        v /= v.norm()
        last_ray = None
        for it in range(n_iters):
            model.zero_grad()
            loss = loss_fn(model, inputs, targets)
            Hv = hvp(loss, params, v)
            # deflate Hv against previously found eigenvecs
            Hv = project_out(Hv, eigenvecs)
            # normalize
            nrm = Hv.norm()
            if nrm.item() == 0:
                break
            v = Hv / nrm
            ray = torch.dot(v, Hv).item()
            if last_ray is not None and abs(ray - last_ray) < 1e-4:
                break
            last_ray = ray
        # final Rayleigh quotient
        model.zero_grad()
        loss = loss_fn(model, inputs, targets)
        Hv = hvp(loss, params, v)
        lam = torch.dot(v, Hv).item()
        eigenvals.append(lam)
        # orthonormalize
        v_orth = project_out(v, eigenvecs)
        if v_orth.norm().item() > 0:
            v_orth = v_orth / v_orth.norm()
            eigenvecs.append(v_orth)
    return eigenvals


def num_linear_regions(model: nn.Module, X: torch.Tensor, device: str = "cpu") -> int:
    """
    Approximate count of distinct linear regions for a ReLU MLP in d-D input space.

    Counts the number of *unique activation patterns* (ReLU on/off combinations)
    across all sampled input points X ∈ R^{N x d}.

    Args:
        model: torch.nn.Module (MLP with ReLUs)
        X: torch.Tensor, shape (N, d) – sample points in input space

    Returns:
        int – number of distinct linear regions found among sampled points
    """

    # --- 1. Register hooks to capture pre-activation tensors before each ReLU ---
    preacts: List[torch.Tensor] = []
    handles = []

    def _make_hook():
        def _hook(mod, inp, out):
            z = inp[0].detach().cpu()
            preacts.append(z)

        return _hook

    for m in model.modules():
        if isinstance(m, nn.ReLU):
            handles.append(m.register_forward_hook(_make_hook()))

    # --- 2. Run forward pass over all samples ---
    _ = model(X.to(next(model.parameters()).device))

    for h in handles:
        h.remove()

    if not preacts:
        return 1  # No ReLUs ⇒ single smooth region

    # --- 3. Build binary activation masks (z > 0) per ReLU layer ---
    masks = [(z > 0).to(torch.int8)
             for z in preacts]  # each shape: (N, hidden)
    # print("Marks", masks.shape)
    mask_concat = torch.cat(masks, dim=1)  # (N, total_hidden)

    # --- 4. Count distinct activation patterns across all sampled points ---
    # Each unique row = one linear region
    unique_patterns = torch.unique(mask_concat, dim=0)
    num_regions = unique_patterns.shape[0]

    return num_regions
