"""GNN + BERT fusion (Task 3, spec section 4.3): early-concatenation and
cross-attention modes, plus an optional DEAM valence/arousal auxiliary head.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.bert_encoder import BertTextEncoder
from src.gnn_model import GraphSAGEEncoder, mean_pool_readout


class EarlyConcatFusion(nn.Module):
    """z = concat(g, t); predicts logits over `num_labels`."""

    def __init__(self, graph_dim: int, text_dim: int, num_labels: int, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(graph_dim + text_dim, num_labels))

    def embed(self, g: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.cat([g, t], dim=-1)

    def forward(self, g: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(g, t))


class CrossAttentionFusion(nn.Module):
    """Graph embedding queries BERT token representations.

    A = softmax(Q K^T / sqrt(d)), Q = g W_Q, K = H_text W_K
    z = concat(g, A H_text)
    """

    def __init__(self, graph_dim: int, text_dim: int, num_labels: int, dropout: float = 0.3):
        super().__init__()
        self.q_proj = nn.Linear(graph_dim, text_dim)
        self.k_proj = nn.Linear(text_dim, text_dim)
        self.text_dim = text_dim
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(graph_dim + text_dim, num_labels))

    def attention_weights(self, g: torch.Tensor, h_text: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Graph-query -> text-token attention weights A, shape (B, L). Exposed
        separately from `embed` for case-study/attention visualization."""
        q = self.q_proj(g).unsqueeze(1)  # (B, 1, text_dim)
        k = self.k_proj(h_text)  # (B, L, text_dim)
        scores = (q @ k.transpose(1, 2)) / (self.text_dim ** 0.5)  # (B, 1, L)
        scores = scores.masked_fill(attention_mask.unsqueeze(1) == 0, float("-inf"))
        return torch.softmax(scores, dim=-1).squeeze(1)  # (B, L)

    def embed(self, g: torch.Tensor, h_text: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        attn = self.attention_weights(g, h_text, attention_mask).unsqueeze(1)  # (B, 1, L)
        context = (attn @ h_text).squeeze(1)  # (B, text_dim)
        return torch.cat([g, context], dim=-1)

    def forward(self, g: torch.Tensor, h_text: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(g, h_text, attention_mask))


class FusionModel(nn.Module):
    """End-to-end GNN + BERT fusion: GraphSAGE encoder + BERT encoder + a
    fusion head selected by `mode` ('early_concat' | 'cross_attention')."""

    def __init__(
        self,
        graph_in_dim: int,
        graph_hidden_dim: int,
        graph_num_layers: int,
        graph_dropout: float,
        bert_model_name: str,
        bert_freeze_layers: str,
        num_labels: int,
        mode: str = "cross_attention",
    ):
        super().__init__()
        self.gnn = GraphSAGEEncoder(graph_in_dim, graph_hidden_dim, graph_num_layers, graph_dropout)
        self.bert = BertTextEncoder(bert_model_name, bert_freeze_layers)
        self.mode = mode
        if mode == "early_concat":
            self.fusion = EarlyConcatFusion(graph_hidden_dim, self.bert.hidden_size, num_labels)
        elif mode == "cross_attention":
            self.fusion = CrossAttentionFusion(graph_hidden_dim, self.bert.hidden_size, num_labels)
        else:
            raise ValueError(f"Unsupported fusion mode: {mode!r}")

    def forward(
        self,
        graph_x: torch.Tensor,
        graph_edge_index: torch.Tensor,
        graph_batch: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        node_embeddings = self.gnn(graph_x, graph_edge_index)
        g = mean_pool_readout(node_embeddings, graph_batch)
        h_text = self.bert(input_ids, attention_mask)
        if self.mode == "early_concat":
            t = h_text[:, 0, :]  # CLS
            return self.fusion(g, t)
        return self.fusion(g, h_text, attention_mask)

    def embed(
        self,
        graph_x: torch.Tensor,
        graph_edge_index: torch.Tensor,
        graph_batch: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the pre-head fused embedding z (for t-SNE / analysis)."""
        node_embeddings = self.gnn(graph_x, graph_edge_index)
        g = mean_pool_readout(node_embeddings, graph_batch)
        h_text = self.bert(input_ids, attention_mask)
        if self.mode == "early_concat":
            return self.fusion.embed(g, h_text[:, 0, :])
        return self.fusion.embed(g, h_text, attention_mask)

    def attention_over_tokens(
        self,
        graph_x: torch.Tensor,
        graph_edge_index: torch.Tensor,
        graph_batch: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Cross-attention weights (graph query -> each text token), shape (B, L),
        for case-study text-alignment visualization. Only defined in cross_attention mode."""
        if self.mode != "cross_attention":
            raise ValueError("attention_over_tokens is only defined for mode='cross_attention'")
        node_embeddings = self.gnn(graph_x, graph_edge_index)
        g = mean_pool_readout(node_embeddings, graph_batch)
        h_text = self.bert(input_ids, attention_mask)
        return self.fusion.attention_weights(g, h_text, attention_mask)


class EmotionRegressionHead(nn.Module):
    """Optional DEAM valence/arousal head trained off the shared GNN embedding g only
    (DEAM has no captions/tags to feed BERT)."""

    def __init__(self, graph_dim: int):
        super().__init__()
        self.head = nn.Linear(graph_dim, 2)  # [valence, arousal]

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return self.head(g)
