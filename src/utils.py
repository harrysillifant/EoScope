import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn import functional as F
import math
import numpy as np
import torch.nn as nn
from typing import List, Iterable, Tuple, Callable, Optional

CriterionFn = Callable[
    [nn.Module, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]


def _flatten_tensors(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


def _like_params(
    params: Iterable[torch.Tensor], flat: torch.Tensor
) -> List[torch.Tensor]:
    """Reshape 'flat' into a list with the same shapes as 'params'."""
    outs, offset = [], 0
    for p in params:
        numel = p.numel()
        outs.append(flat[offset : offset + numel].view_as(p))
        offset += numel
    return outs


def flatten_params(model: nn.Module) -> torch.Tensor:
    """Flatten current parameters into a single vector (detached CPU tensor)."""
    with torch.no_grad():
        return _flatten_tensors([p.detach().cpu() for p in model.parameters()])


# def flatten_params(model):
#     params = [p for p in model.parameters() if p.requires_grad]
#     vec = parameters_to_vector(params).detach()
#     return vec, params


@torch.no_grad()
def _normalize(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return v / (v.norm() + eps)


def _grad(loss, params, create_graph=False):
    grads = torch.autograd.grad(
        loss, params, create_graph=create_graph, retain_graph=True, allow_unused=True
    )
    grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
    return torch.cat([g.contiguous().view(-1) for g in grads])


# def hvp(loss, params, v):
#     """Compute Hessian-vector product Hv where H =
#     abla^2 loss and v is a vector in parameter space.
#         - loss should be a scalar (torch.Tensor)
#         - params is a list of parameters
#         - v is a 1D torch tensor matching flattened params
#         Returns flattened Hv tensor.
#     """
#     grad_flat = _grad(loss, params, create_graph=True)
#     dot = torch.dot(grad_flat, v)
#     hv = torch.autograd.grad(dot, params, retain_graph=True)
#     hv = [h if h is not None else torch.zeros_like(
#         p) for h, p in zip(hv, params)]
#     return torch.cat([h.contiguous().view(-1) for h in hv]).detach()
#
def hvp(
    loss: torch.Tensor, params: List[torch.Tensor], v_flat: torch.Tensor
) -> torch.Tensor:
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


# def top_hessian_eigenpair(
#     model: nn.Module,
#     criterion: CriterionFn,
#     x_batch: torch.Tensor,
#     y_batch: torch.Tensor,
#     iters: int = 20,
#     init_vec: torch.Tensor | None = None,
# ) -> Tuple[float, torch.Tensor]:
#     device = next(model.parameters()).device
#     params = [p for p in model.parameters() if p.requires_grad]
#
#     model.zero_grad(set_to_none=True)
#     y_pred = model(x_batch)
#     loss = criterion(model, x_batch, y_pred, y_batch)  # <-- use composite loss
#
#     if init_vec is None:
#         v = torch.randn(sum(p.numel() for p in params), device=device)
#     else:
#         v = init_vec.to(device)
#     v = _normalize(v)
#
#     for _ in range(iters):
#         hv = hvp(loss, params, v)
#         hv_det = hv.detach()
#         lam = float((v * hv_det).sum().item())
#         v = _normalize(hv_det)
#
#     hv = hvp(loss, params, v)
#     lam = float((v * hv).sum().item())
#     v = _normalize(v)
#     return lam, v
#
#
def top_hessian_eigenpair(
    model: nn.Module,
    criterion,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    iters: int = 20,
    init_vec: torch.Tensor | None = None,
) -> Tuple[float, torch.Tensor]:
    """
    Compute the top Hessian eigenpair (largest eigenvalue and eigenvector) of the loss
    w.r.t. model parameters using power iteration and autograd Hv products.

    IMPORTANT: `criterion` is expected to have the signature:
        loss = criterion(model, inputs, targets)
    (e.g. your `loss_fn_mse`).

    Returns:
        (eigval, eigvec_flat)
        - eigval: float (Rayleigh estimate for top eigenvalue)
        - eigvec_flat: 1D torch.Tensor containing the flattened eigenvector (same device/dtype as params)
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    model.zero_grad()
    model.eval()  # avoid dropout / batchnorm updates; change to model.train() if you want training behavior

    x = x_batch.to(device)
    y = y_batch.to(device)

    # collect parameters that require grad
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise ValueError("Model has no parameters that require gradients.")

    def _flatten(tensor_list):
        return torch.cat([t.reshape(-1) for t in tensor_list])

    # compute loss (criterion takes model, inputs, targets)
    loss = criterion(model, x, y)
    if loss.dim() != 0:
        loss = loss.mean()

    # first-order gradient with create_graph=True to allow Hv products
    grads = torch.autograd.grad(loss, params, create_graph=True)
    g_flat = _flatten(grads)

    param_numel = g_flat.numel()

    # initialize v
    if init_vec is not None:
        v = init_vec.to(device=device, dtype=dtype).reshape(-1)
        if v.numel() != param_numel:
            raise ValueError(
                f"init_vec has wrong size ({v.numel()}) expected {param_numel}"
            )
        v = v / (v.norm() + 1e-12)
    else:
        v = torch.randn(param_numel, device=device, dtype=dtype)
        v = v / (v.norm() + 1e-12)

    eps = 1e-12
    eigval = 0.0

    for _ in range(iters):
        # compute g^T v
        g_dot_v = torch.dot(g_flat, v)

        # Hessian-vector product: grad(g_dot_v, params)
        Hv_tensors = torch.autograd.grad(g_dot_v, params, retain_graph=True)
        Hv_flat = _flatten(Hv_tensors)

        # detach Hv for numerical iteration (no higher-order grads needed)
        Hv_flat_det = Hv_flat.detach()

        # Rayleigh quotient estimate
        eigval = float(torch.dot(v, Hv_flat_det).item())

        hv_norm = Hv_flat_det.norm().item()
        if hv_norm < 1e-16:
            # near-zero Hv -> stop
            break

        # next v is normalized Hv
        v = Hv_flat_det / (hv_norm + eps)

    # final eigenvector (detached)
    eigvec = v.detach().clone()

    # recompute precise eigenvalue using final eigvec (recompute Hv w.r.t. eigvec)
    # recompute loss and grads with create_graph=True
    model.zero_grad()
    loss = criterion(model, x, y)
    if loss.dim() != 0:
        loss = loss.mean()
    grads = torch.autograd.grad(loss, params, create_graph=True)
    g_flat = _flatten(grads)

    g_dot_v = torch.dot(g_flat, eigvec)
    Hv_tensors = torch.autograd.grad(g_dot_v, params, retain_graph=False)
    Hv_flat = _flatten(Hv_tensors).detach()

    final_eigval = float(torch.dot(eigvec, Hv_flat).item())

    return final_eigval, eigvec


# def power_iteration_hessian_max(
#     loss_fn, model, params, inputs, targets, n_iters=20, tol=1e-4, device="cpu"
# ):
#     """Estimate top eigenvalue and eigenvector of Hessian via power iteration using hvp.
#     loss_fn: callable (model, inputs, targets) -> scalar loss
#     params: list(model.parameters())
#     Returns (lambda_max, vec)
#     """
#     flat, _ = flatten_params(model)
#     v = torch.randn_like(flat).to(device)
#     v /= v.norm()
#     last_ray = None
#     for i in range(n_iters):
#         model.zero_grad()
#         loss = loss_fn(model, inputs, targets)
#         Hv = hvp(loss, params, v)
#         ray = torch.dot(v, Hv).item()
#         if last_ray is not None and abs(ray - last_ray) < tol:
#             break
#         last_ray = ray
#         v = Hv
#         nrm = v.norm()
#         if nrm.item() == 0:
#             break
#         v = v / nrm
#     model.zero_grad()
#     loss = loss_fn(model, inputs, targets)
#     Hv = hvp(loss, params, v)
#     lambda_max = torch.dot(v, Hv).item()
#     return lambda_max, v
#


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
        x = inputs[i : i + 1]
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


def num_linear_regions_basic(
    model: nn.Module, X: torch.Tensor, device: str = "cpu"
) -> int:
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

    _ = model(X.to(next(model.parameters()).device))

    for h in handles:
        h.remove()

    if not preacts:
        return 1

    masks = [(z > 0).to(torch.int8) for z in preacts]  # (N, hidden)
    mask_concat = torch.cat(masks, dim=1)  # (N, total_hidden)

    unique_patterns = torch.unique(mask_concat, dim=0)
    num_regions = unique_patterns.shape[0]

    return num_regions


def num_linear_regions_pier(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor, device: str = "cpu"
) -> int:
    # Sample two points with different labels in the input space
    # sample points from the line across them
    # count the number of distinct linear regions along the line

    num_samples_pairs = 10
    num_samples_line = 10
    num_regions_all = []
    while num_samples_pairs:
        idx1 = torch.randint(0, X.size(0), (1,)).item()
        idx2 = torch.randint(0, X.size(0), (1,)).item()
        ys_on_line = []
        if y[idx1] != y[idx2]:  # different labels
            x1, x2 = X[idx1], X[idx2]
            for a in np.linspace(0, 1, num_samples_line):
                x = x1 * (1 - a) + x2 * a
                x = x.unsqueeze(0)
                yh = model(x)
                ys_on_line.append(yh)
            num_samples_pairs -= 1

        num_regions = 0
        for i, _ in enumerate(ys_on_line[1:-1]):
            y_delta_2 = ys_on_line[i + 1] - ys_on_line[i]
            y_delta_1 = ys_on_line[i] - ys_on_line[i - 1]
            if torch.norm(y_delta_2 - y_delta_1) > 1e-5:
                num_regions += 1
        num_regions_all.append(num_regions)
    return np.mean(num_regions_all)


def num_linear_regions_hanin(
    model: nn.Module, X: torch.Tensor, device: str = "cpu"
) -> int:
    # Sample a point from the training data
    # Compute the number of linear regions along that line?
    # For 5 independent runs, sample 100 lines, take average
    pass


def num_linear_regions_humayan():
    # Sample point in the training or test set
    # Sample P orthonormal vectors in input space
    # Get convex hull neighborbood about the point
    # Take the vertices of convex hull and ?count linear regions on each?
    pass
