"""GCN backbone (Kipf & Welling 2017) with hooks for per-class loss weighting."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GraphConvolution(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        self.bias = nn.Parameter(torch.empty(out_dim)) if bias else None
        nn.init.xavier_uniform_(self.weight)
        if bias:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = x @ self.weight
        out = torch.sparse.mm(adj, support) if adj.is_sparse else adj @ support
        if self.bias is not None:
            out = out + self.bias
        return out


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden: int, n_classes: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.gc1 = GraphConvolution(in_dim, hidden)
        self.gc2 = GraphConvolution(hidden, n_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return self.gc2(x, adj)
