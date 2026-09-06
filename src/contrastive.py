"""Task 4 (OPTIONAL/bonus): contrastive GNN-BERT dual encoder on MusicCaps,
InfoNCE loss, R@K retrieval evaluation. Implemented only after Tasks 1-3 are
complete, per project development order. Must not block Tasks 1-3.
"""
from __future__ import annotations


class ContrastiveDualEncoder:
    def __init__(self, graph_dim: int, text_dim: int, embed_dim: int, temperature: float = 0.07):
        raise NotImplementedError("Optional Phase 12: Task 4 contrastive dual encoder.")


def info_nce_loss(graph_embeds, text_embeds, temperature: float = 0.07):
    raise NotImplementedError("Optional Phase 12: InfoNCE contrastive loss.")


def retrieval_recall_at_k(similarity_matrix, k: int):
    raise NotImplementedError("Optional Phase 12: R@K retrieval metric.")
