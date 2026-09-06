"""Tensor-shape agreement tests across BERT/GNN/fusion/CNN."""
import torch
from torch_geometric.data import Batch, Data

from src.gnn_model import GNNGenreClassifier


def test_gnn_output_matches_hidden_dim():
    torch.manual_seed(0)
    graphs = [
        Data(x=torch.randn(6, 64), edge_index=torch.randint(0, 6, (2, 10))),
        Data(x=torch.randn(4, 64), edge_index=torch.randint(0, 4, (2, 6))),
    ]
    batch = Batch.from_data_list(graphs)
    model = GNNGenreClassifier(in_dim=64, hidden_dim=32, num_layers=2, num_classes=16)
    logits = model(batch.x, batch.edge_index, batch.batch)
    assert logits.shape == (2, 16)


def test_bert_cls_output_matches_hidden_size():
    from src.bert_encoder import BertTagClassifier

    torch.manual_seed(0)
    model = BertTagClassifier("distilbert-base-uncased", num_tags=20, freeze_layers="all")
    input_ids = torch.randint(0, 1000, (3, 16))
    attention_mask = torch.ones(3, 16, dtype=torch.long)
    logits = model(input_ids, attention_mask)
    assert logits.shape == (3, 20)


def test_fusion_output_matches_num_labels():
    from src.fusion_model import FusionModel

    torch.manual_seed(0)
    graphs = [
        Data(x=torch.randn(6, 64), edge_index=torch.randint(0, 6, (2, 10))),
        Data(x=torch.randn(4, 64), edge_index=torch.randint(0, 4, (2, 6))),
    ]
    graph_batch = Batch.from_data_list(graphs)
    input_ids = torch.randint(0, 1000, (2, 16))
    attention_mask = torch.ones(2, 16, dtype=torch.long)

    for mode in ("early_concat", "cross_attention"):
        model = FusionModel(
            graph_in_dim=64,
            graph_hidden_dim=32,
            graph_num_layers=2,
            graph_dropout=0.3,
            bert_model_name="distilbert-base-uncased",
            bert_freeze_layers="all",
            num_labels=20,
            mode=mode,
        )
        logits = model(graph_batch.x, graph_batch.edge_index, graph_batch.batch, input_ids, attention_mask)
        assert logits.shape == (2, 20), f"mode={mode}"


def test_cnn_output_matches_num_classes():
    from src.cnn_baseline import MelSpecCNN

    torch.manual_seed(0)
    model = MelSpecCNN(n_mels=128, num_classes=16)
    x = torch.randn(3, 1, 128, 200)
    logits = model(x)
    assert logits.shape == (3, 16)
