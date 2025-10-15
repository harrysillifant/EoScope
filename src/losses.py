from __future__ import annotations
from typing import Callable, Optional, Union
import importlib
import torch
import torch.nn as nn

CriterionFn = Callable[[nn.Module, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

def _wrap_reduction(loss_module: nn.Module) -> CriterionFn:
    def criterion(model: nn.Module, x: torch.Tensor, y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return loss_module(y_hat, y_true)
    return criterion

def _resolve_custom(custom: Union[str, CriterionFn, None]) -> Optional[CriterionFn]:
    if custom is None:
        return None
    if callable(custom):
        return custom
    if isinstance(custom, str):
        if ":" not in custom:
            raise ValueError("custom loss string must be 'package.module:func'")
        m, f = custom.split(":", 1)
        mod = importlib.import_module(m)
        fn = getattr(mod, f, None)
        if not callable(fn):
            raise ValueError(f"could not find callable {f} in module {m}")
        return fn
    raise TypeError(f"unsupported custom loss type: {type(custom)}")

def get_loss_fn(name_or_callable: Union[str, CriterionFn], custom: Union[str, CriterionFn, None] = None) -> CriterionFn:
    if callable(name_or_callable):
        return name_or_callable  # already correct signature
    name = str(name_or_callable).lower().strip()
    if name in ("mse", "mse_loss"):
        return _wrap_reduction(nn.MSELoss())
    if name in ("l1", "mae", "l1_loss"):
        return _wrap_reduction(nn.L1Loss())
    if name in ("custom", "user"):
        fn = _resolve_custom(custom)
        if fn is None:
            raise ValueError("loss_name='custom' requires `custom` callable or 'module:func'")
        return fn
    raise ValueError(f"unknown loss: {name_or_callable}")
