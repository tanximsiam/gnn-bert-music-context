"""BERT/DistilBERT text encoder with multi-label tag classification head.

Task 1: multi-label tag classifier, y_k = sigmoid(w_k^T CLS(BERT(x)) + b_k),
trained with per-tag BCE on raw artist bio text.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


def apply_bert_freeze(bert: nn.Module, freeze_layers: str) -> None:
    """Shared freeze logic: 'all' freezes the whole encoder, 'none' leaves it
    fully trainable, and an integer string unfreezes only the top N layers."""
    if freeze_layers == "none":
        return
    if freeze_layers == "all":
        for param in bert.parameters():
            param.requires_grad = False
        return
    n = int(freeze_layers)
    for param in bert.parameters():
        param.requires_grad = False
    layers = bert.transformer.layer if hasattr(bert, "transformer") else bert.encoder.layer
    for layer in layers[-n:]:
        for param in layer.parameters():
            param.requires_grad = True


class BertTagClassifier(nn.Module):
    """Frozen or partially-unfrozen BERT encoder + linear multi-label tag head."""

    def __init__(self, model_name: str, num_tags: int, freeze_layers: str = "all"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.head = nn.Linear(hidden_size, num_tags)
        apply_bert_freeze(self.bert, freeze_layers)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]  # [CLS] token representation
        return self.head(cls)  # logits; use BCEWithLogitsLoss for multi-label


class BertTextEncoder(nn.Module):
    """Bare BERT encoder (no head) returning full token hidden states, for use
    by Task 3 fusion models (early-concat needs CLS, cross-attention needs the
    full token sequence)."""

    def __init__(self, model_name: str, freeze_layers: str = "all"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        apply_bert_freeze(self.bert, freeze_layers)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state  # (B, L, H); [:, 0, :] is the CLS embedding
