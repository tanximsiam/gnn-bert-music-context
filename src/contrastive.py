"""Task 4: contrastive GNN-BERT dual encoder on MusicCaps, InfoNCE loss, R@K
retrieval evaluation (spec section 4.4).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bert_encoder import BertTextEncoder
from src.gnn_model import GraphSAGEEncoder, mean_pool_readout


class ContrastiveDualEncoder(nn.Module):
    """Graph encoder (GraphSAGE + mean-pool) and text encoder (BERT CLS), each
    projected into a shared `embed_dim` space and L2-normalized, matching the
    spec's InfoNCE similarity sim(u, v) = u^T v / (||u|| ||v||)."""

    def __init__(
        self,
        graph_in_dim: int,
        graph_hidden_dim: int,
        graph_num_layers: int,
        graph_dropout: float,
        bert_model_name: str,
        bert_freeze_layers: str,
        embed_dim: int,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.gnn = GraphSAGEEncoder(graph_in_dim, graph_hidden_dim, graph_num_layers, graph_dropout)
        self.bert = BertTextEncoder(bert_model_name, bert_freeze_layers)
        self.graph_proj = nn.Linear(graph_hidden_dim, embed_dim)
        self.text_proj = nn.Linear(self.bert.hidden_size, embed_dim)
        self.temperature = temperature

    def encode_graph(self, graph_x: torch.Tensor, graph_edge_index: torch.Tensor, graph_batch: torch.Tensor) -> torch.Tensor:
        node_embeddings = self.gnn(graph_x, graph_edge_index)
        g = mean_pool_readout(node_embeddings, graph_batch)
        return F.normalize(self.graph_proj(g), dim=-1)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h_text = self.bert(input_ids, attention_mask)
        cls = h_text[:, 0, :]
        return F.normalize(self.text_proj(cls), dim=-1)

    def forward(
        self,
        graph_x: torch.Tensor,
        graph_edge_index: torch.Tensor,
        graph_batch: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        graph_embeds = self.encode_graph(graph_x, graph_edge_index, graph_batch)
        text_embeds = self.encode_text(input_ids, attention_mask)
        return graph_embeds, text_embeds


def info_nce_loss(graph_embeds: torch.Tensor, text_embeds: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE over a batch of paired (graph, caption) embeddings:
    L_NCE = -log[exp(sim(g_i,t_i)/tau) / sum_j exp(sim(g_i,t_j)/tau)], averaged
    over both the graph->text and text->graph directions (in-batch negatives)."""
    logits = graph_embeds @ text_embeds.T / temperature  # (B, B) cosine sim (inputs are L2-normalized)
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_g2t = F.cross_entropy(logits, targets)
    loss_t2g = F.cross_entropy(logits.T, targets)
    return (loss_g2t + loss_t2g) / 2


def retrieval_recall_at_k(similarity_matrix: np.ndarray | torch.Tensor, k: int) -> float:
    """R@K: fraction of queries (rows) whose true match (diagonal index) is
    among the top-K most similar columns."""
    sim = similarity_matrix.detach().cpu().numpy() if isinstance(similarity_matrix, torch.Tensor) else np.asarray(similarity_matrix)
    n = sim.shape[0]
    top_k_idx = np.argsort(-sim, axis=1)[:, :k]
    hits = sum(1 for i in range(n) if i in top_k_idx[i])
    return hits / n
