"""Entry point for evaluation. Usage: python -m src.evaluate --task {1,2,3,4} --checkpoint PATH

Computes Macro-F1/Micro-F1/AUC-PR (tag tasks), MAE/R^2 (DEAM emotion, if
enabled), or R@1/5/10 (Task 4 retrieval), and writes results to
results/metrics/ and results/plots/. Never fabricates numbers: if no
checkpoint/run exists yet, this must fail loudly rather than print
placeholder metrics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader as TorchDataLoader
from transformers import AutoTokenizer

from src.bert_encoder import BertTagClassifier
from src.datasets import (
    build_task1_dataset,
    build_task1_top_tags,
    load_corrupted_track_ids,
    load_fma_metadata,
    load_fma_splits,
)
from src.fusion_model import FusionModel
from src.gnn_model import GNNGenreClassifier
from src.graph_builder import build_or_load_track_graph
from src.train import MultiModalTagDataset, fusion_collate
from src.utils import get_logger, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CSE425 GNN-BERT model.")
    parser.add_argument("--task", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    return parser.parse_args()


def run_task3_analysis(config: dict, checkpoint_path: str, logger) -> None:
    """Phase 10: t-SNE of the fused embedding z (colored by genre) + 3 case
    studies (graph structure + masked text + true/predicted tags across all
    4 ablation variants). Requires a completed `train_task3` run directory
    (with per-variant *_best_model.pt checkpoints and metrics.json)."""
    run_dir = Path(checkpoint_path).parent
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"{metrics_path} not found — run `python -m src.train --task 3` first.")
    with metrics_path.open() as f:
        task3_metrics = json.load(f)
    top_tags = task3_metrics["top_tags"]
    num_labels = len(top_tags)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tracks = load_fma_metadata(config["dataset"]["metadata_root"], subset="medium")
    corrupted_ids = load_corrupted_track_ids()
    splits = load_fma_splits(tracks, exclude_track_ids=corrupted_ids)
    task_df = build_task1_dataset(tracks, top_tags)
    split_of = {tid: name for name, ids in splits.items() for tid in ids}
    task_df = task_df.assign(split=task_df["track_id"].map(split_of))
    test_df = task_df[task_df["split"] == "test"].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(config["bert"]["model_name"])
    cache_dir = Path("data/processed/graph_cache")
    test_ds = MultiModalTagDataset(
        test_df,
        audio_root=config["dataset"]["root"],
        cache_dir=cache_dir,
        audio_config=config["audio"],
        graph_config=config["graph"],
        sample_rate=config["dataset"]["sample_rate"],
        tokenizer=tokenizer,
        max_length=config["bert"]["max_length"],
    )
    test_loader = TorchDataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=fusion_collate)

    in_dim = 2 * config["audio"]["n_mfcc"] + (24 if config["audio"]["use_chroma"] else 0)
    model = FusionModel(
        graph_in_dim=in_dim,
        graph_hidden_dim=config["gnn"]["hidden_dim"],
        graph_num_layers=config["gnn"]["num_layers"],
        graph_dropout=config["gnn"]["dropout"],
        bert_model_name=config["bert"]["model_name"],
        bert_freeze_layers=config["bert"]["freeze_layers"],
        num_labels=num_labels,
        mode="cross_attention",
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_z, all_genres, all_track_ids = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            g = batch["graph_batch"].to(device)
            z = model.embed(g.x, g.edge_index, g.batch, batch["input_ids"].to(device), batch["attention_mask"].to(device))
            all_z.append(z.cpu().numpy())
            track_ids = g.track_id.cpu().tolist() if torch.is_tensor(g.track_id) else g.track_id
            all_track_ids += track_ids
    all_z = np.concatenate(all_z, axis=0)
    genre_by_track = dict(zip(tracks["track_id"], tracks["genre_top"]))
    all_genres = [genre_by_track[tid] for tid in all_track_ids]

    logger.info("Running t-SNE on %d test embeddings (dim=%d)...", len(all_z), all_z.shape[1])
    z_2d = TSNE(n_components=2, random_state=42, init="pca", perplexity=min(30, len(all_z) - 1)).fit_transform(all_z)

    unique_genres = sorted(set(all_genres))
    cmap = plt.get_cmap("tab20", len(unique_genres))
    plt.figure(figsize=(9, 7))
    for i, genre in enumerate(unique_genres):
        idx = [j for j, g_ in enumerate(all_genres) if g_ == genre]
        plt.scatter(z_2d[idx, 0], z_2d[idx, 1], s=12, color=cmap(i), label=genre)
    plt.legend(fontsize=6, markerscale=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.title("t-SNE of cross-attention fused embedding z (test set, colored by genre)")
    plt.tight_layout()
    plt.savefig(run_dir / "tsne_genre.png", dpi=150)
    plt.close()
    logger.info("Saved t-SNE plot to %s", run_dir / "tsne_genre.png")

    _save_case_studies(config, run_dir, task3_metrics, tracks, test_df, top_tags, device, logger)


def _load_variant_model(variant: str, config: dict, run_dir: Path, num_labels: int, in_dim: int, device: str):
    path = run_dir / f"{variant}_best_model.pt"
    if variant == "gnn_only":
        model = GNNGenreClassifier(
            in_dim=in_dim, hidden_dim=config["gnn"]["hidden_dim"], num_layers=config["gnn"]["num_layers"],
            num_classes=num_labels, dropout=config["gnn"]["dropout"], model=config["gnn"]["model"],
        ).to(device)
    elif variant == "bert_only":
        model = BertTagClassifier(
            config["bert"]["model_name"], num_tags=num_labels, freeze_layers=config["bert"]["freeze_layers"]
        ).to(device)
    else:
        mode = "early_concat" if variant == "early_concat" else "cross_attention"
        model = FusionModel(
            graph_in_dim=in_dim, graph_hidden_dim=config["gnn"]["hidden_dim"],
            graph_num_layers=config["gnn"]["num_layers"], graph_dropout=config["gnn"]["dropout"],
            bert_model_name=config["bert"]["model_name"], bert_freeze_layers=config["bert"]["freeze_layers"],
            num_labels=num_labels, mode=mode,
        ).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def _save_case_studies(config, run_dir, task3_metrics, tracks, test_df, top_tags, device, logger) -> None:
    """3 case studies: graph structure + masked text + true tags vs. each
    ablation variant's predicted tags (using its own tuned thresholds)."""
    rng = np.random.RandomState(42)
    sample_rows = test_df.iloc[rng.choice(len(test_df), size=min(3, len(test_df)), replace=False)]

    in_dim = 2 * config["audio"]["n_mfcc"] + (24 if config["audio"]["use_chroma"] else 0)
    num_labels = len(top_tags)
    variants = ["gnn_only", "bert_only", "early_concat", "cross_attention"]
    models = {v: _load_variant_model(v, config, run_dir, num_labels, in_dim, device) for v in variants}
    thresholds = {v: np.array(task3_metrics["ablations"][v]["per_tag_thresholds"]) for v in variants}

    tokenizer = AutoTokenizer.from_pretrained(config["bert"]["model_name"])
    cache_dir = Path("data/processed/graph_cache")
    case_studies = []

    for _, row in sample_rows.iterrows():
        track_id = row["track_id"]
        graph = build_or_load_track_graph(
            track_id, config["dataset"]["root"], cache_dir, config["dataset"]["sample_rate"],
            config["audio"]["segment_seconds"], config["audio"]["n_mfcc"], config["audio"]["use_chroma"],
            config["graph"]["similarity_threshold"], config["graph"]["bidirectional_edges"], config["graph"]["self_loops"],
        )
        enc = tokenizer(row["text"], truncation=True, padding="max_length", max_length=config["bert"]["max_length"], return_tensors="pt")
        input_ids, attention_mask = enc["input_ids"].to(device), enc["attention_mask"].to(device)
        g_batch = graph.clone()
        g_batch.batch = torch.zeros(g_batch.x.size(0), dtype=torch.long)
        gx, gei, gb = g_batch.x.to(device), g_batch.edge_index.to(device), g_batch.batch.to(device)

        true_tags = [top_tags[k] for k in range(num_labels) if row["labels"][k] == 1]
        predictions = {}
        with torch.no_grad():
            for variant, model in models.items():
                if variant == "gnn_only":
                    logits = model(gx, gei, gb)
                elif variant == "bert_only":
                    logits = model(input_ids, attention_mask)
                else:
                    logits = model(gx, gei, gb, input_ids, attention_mask)
                probs = torch.sigmoid(logits)[0].cpu().numpy()
                pred_tags = [top_tags[k] for k in range(num_labels) if probs[k] >= thresholds[variant][k]]
                predictions[variant] = pred_tags

        num_temporal = int((graph.edge_index[0] - graph.edge_index[1]).abs().eq(1).sum().item())
        case_studies.append({
            "track_id": int(track_id),
            "genre": tracks.set_index("track_id").loc[track_id, "genre_top"],
            "masked_text": row["text"][:300],
            "num_graph_nodes": int(graph.x.shape[0]),
            "num_graph_edges": int(graph.edge_index.shape[1]),
            "num_temporal_edges_approx": num_temporal,
            "true_tags": true_tags,
            "predicted_tags_by_variant": predictions,
        })

    with (run_dir / "case_studies.json").open("w") as f:
        json.dump(case_studies, f, indent=2)
    logger.info("Saved %d case studies to %s", len(case_studies), run_dir / "case_studies.json")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    logger = get_logger(f"evaluate.task{args.task}")
    logger.info("Evaluating checkpoint: %s", args.checkpoint)

    if args.task == 1:
        raise NotImplementedError("Task 1 evaluation implemented in Phase 6.")
    if args.task == 2:
        raise NotImplementedError("Task 2 evaluation implemented in Phases 4-5.")
    if args.task == 3:
        run_task3_analysis(config, args.checkpoint, logger)
        return
    raise NotImplementedError("Task 4 (optional) evaluation implemented in Phase 12.")


if __name__ == "__main__":
    main()
