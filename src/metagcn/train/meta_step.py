"""Meta-learning step that updates per-class loss weights."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def weighted_cross_entropy(
    logits: torch.Tensor, target: torch.Tensor, class_weights: torch.Tensor
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    sample_weights = class_weights[target]
    return (sample_weights * nll).mean()


def meta_update_class_weights(
    model: nn.Module,
    train_batch: dict[str, torch.Tensor],
    val_batch: dict[str, torch.Tensor],
    class_weights: torch.Tensor,
    inner_lr: float,
    outer_lr: float,
) -> torch.Tensor:
    """One MAML-style outer step on the per-class weights.

    1. Compute weighted training loss.
    2. Compute one-step SGD update on model params (differentiable).
    3. Evaluate unweighted validation loss using the *updated* params.
    4. Differentiate val loss w.r.t. the class weights and gradient-step.
    """
    train_logits = model(train_batch["x"], train_batch["adj"])
    train_loss = weighted_cross_entropy(train_logits, train_batch["y"], class_weights)

    grads = torch.autograd.grad(train_loss, model.parameters(), create_graph=True)
    fast_weights = [p - inner_lr * g for p, g in zip(model.parameters(), grads, strict=True)]

    val_logits = forward_with_fast_weights(model, val_batch["x"], val_batch["adj"], fast_weights)
    val_loss = F.cross_entropy(val_logits, val_batch["y"])

    weight_grad = torch.autograd.grad(val_loss, class_weights)[0]
    new_weights = class_weights - outer_lr * weight_grad
    return _project_simplex(new_weights)


def forward_with_fast_weights(
    model: nn.Module, x: torch.Tensor, adj: torch.Tensor, fast_weights: list[torch.Tensor]
) -> torch.Tensor:
    # Two-layer GCN: weight, bias, weight, bias.
    w1, b1, w2, b2 = fast_weights
    h = F.relu((x @ w1 if not adj.is_sparse else x @ w1) + b1)
    h = adj @ h if not adj.is_sparse else torch.sparse.mm(adj, h)
    out = h @ w2 + b2
    return adj @ out if not adj.is_sparse else torch.sparse.mm(adj, out)


def _project_simplex(v: torch.Tensor, z: float = 1.0) -> torch.Tensor:
    """Projection onto the probability simplex of dimension k summing to z."""
    n = v.shape[0]
    u = torch.sort(v, descending=True).values
    cssv = torch.cumsum(u, dim=0) - z
    ind = torch.arange(1, n + 1, device=v.device, dtype=v.dtype)
    cond = u - cssv / ind > 0
    rho = cond.nonzero(as_tuple=False).max().item() + 1
    theta = cssv[rho - 1] / float(rho)
    return torch.clamp(v - theta, min=0)
