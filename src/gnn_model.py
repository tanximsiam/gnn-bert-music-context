"""GraphSAGE (and optional GAT) encoder over segment graphs with mean-pool
readout, per spec section 4.2.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, global_mean_pool


class GraphSAGEEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.3):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList([SAGEConv(dims[i], dims[i + 1]) for i in range(num_layers)])
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GATEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.3, heads: int = 4):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * num_layers
        self.convs = nn.ModuleList(
            [
                GATConv(dims[i], dims[i + 1] // heads if i < num_layers - 1 else dims[i + 1], heads=heads if i < num_layers - 1 else 1, dropout=dropout)
                for i in range(num_layers)
            ]
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def mean_pool_readout(node_embeddings: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
    return global_mean_pool(node_embeddings, batch_index)


class GNNGenreClassifier(nn.Module):
    """GraphSAGE/GAT encoder + mean-pool readout + linear head for single-label
    genre classification (softmax/cross-entropy, per project rule: no sigmoid
    for mutually-exclusive multi-class genre prediction)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.3,
        model: str = "graphsage",
    ):
        super().__init__()
        if model == "graphsage":
            self.encoder = GraphSAGEEncoder(in_dim, hidden_dim, num_layers, dropout)
        elif model == "gat":
            self.encoder = GATEncoder(in_dim, hidden_dim, num_layers, dropout)
        else:
            raise ValueError(f"Unsupported gnn model: {model!r}")
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        node_embeddings = self.encoder(x, edge_index)
        graph_embedding = mean_pool_readout(node_embeddings, batch_index)
        return self.head(graph_embedding)  # logits; use CrossEntropyLoss (applies softmax internally)
