"""Entry point for evaluation. Usage: python -m src.evaluate --task {1,2,3,4} --checkpoint PATH

Computes Macro-F1/Micro-F1/AUC-PR (tag tasks, Tasks 1/3) or R@1/5/10 (Task 4
retrieval), and writes results to results/metrics/ and results/plots/. DEAM
valence/arousal MAE/R^2 is NOT implemented (DEAM requires manual download and
is currently disabled — see config.yaml's `emotion` section and
src/datasets.py:load_deam). Never fabricates numbers: if no checkpoint/run
exists yet, this must fail loudly rather than print placeholder metrics.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from transformers import AutoTokenizer

from src.bert_encoder import BertTagClassifier
from src.contrastive import ContrastiveDualEncoder, retrieval_recall_at_k
from src.datasets import (
    build_musiccaps_splits,
    build_musiccaps_tag_dataset,
    build_task3_dataset,
    load_corrupted_track_ids,
    load_fma_metadata,
    load_fma_splits,
    load_musiccaps,
)
from src.fusion_model import FusionModel
from src.gnn_model import GNNGenreClassifier
from src.graph_builder import SegmentGraphDataset, build_genre_label_map, build_or_load_track_graph
from src.train import (
    MultiModalTagDataset,
    MusicCapsContrastiveDataset,
    TagTextDataset,
    contrastive_collate,
    fusion_collate,
)
from src.utils import get_logger, load_config

# Curated mood/atmosphere descriptors used to color Task 3's t-SNE plot (spec: "genre and
# mood"). Intersected at runtime with whatever top-K tags were actually selected, since the
# tag vocabulary is data-driven (top-20 by frequency on TRAIN), not fixed in advance.
MOOD_KEYWORDS = {
    "psychedelic", "horror", "noise", "experimental", "ambient", "dark", "melancholic",
    "uplifting", "energetic", "calm", "dreamy", "eerie", "peaceful", "happy", "sad", "angry",
}


def _mean_auc_pr(labels: np.ndarray, probs: np.ndarray) -> float:
    scores = [
        average_precision_score(labels[:, k], probs[:, k]) for k in range(labels.shape[1]) if labels[:, k].sum() > 0
    ]
    return float(np.mean(scores)) if scores else 0.0


def run_task1_evaluation(config: dict, checkpoint_path: str, logger) -> None:
    """Reload a trained Task 1 BERT checkpoint and recompute test metrics
    independently of train.py, using the same tag vocabulary and per-tag
    thresholds recorded in that run's metrics.json (tuned on validation only).
    Task 1 is the MusicCaps caption -> tag proxy classifier (spec section 4.1)."""
    run_dir = Path(checkpoint_path).parent
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"{metrics_path} not found — run `python -m src.train --task 1` first.")
    with metrics_path.open() as f:
        train_metrics = json.load(f)
    top_tags = train_metrics["top_tags"]
    thresholds = np.array(train_metrics["per_tag_thresholds"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cc = config["contrastive"]
    mc_df = pd.read_csv(Path(cc["dataset_root"]) / "musiccaps.csv")
    mc_df["aspect_list"] = mc_df["aspect_list"].apply(ast.literal_eval)
    mc_splits = build_musiccaps_splits(mc_df)
    task1_df = build_musiccaps_tag_dataset(mc_df, top_tags)
    split_of = {yid: name for name, ids in mc_splits.items() for yid in ids}
    task1_df = task1_df.assign(split=task1_df["ytid"].map(split_of))
    test_df = task1_df[task1_df["split"] == "test"]

    tokenizer = AutoTokenizer.from_pretrained(config["bert"]["model_name"])
    test_ds = TagTextDataset(test_df, tokenizer, config["bert"]["max_length"])
    test_loader = TorchDataLoader(test_ds, batch_size=config["training"]["batch_size"], shuffle=False)

    model = BertTagClassifier(
        config["bert"]["model_name"], num_tags=len(top_tags), freeze_layers=config["bert"]["freeze_layers"]
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            all_probs += torch.sigmoid(logits).cpu().tolist()
            all_labels += batch["labels"].tolist()
    probs, labels = np.array(all_probs), np.array(all_labels)
    preds = (probs >= thresholds).astype(int)
    result = {
        "checkpoint": str(checkpoint_path),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "auc_pr": _mean_auc_pr(labels, probs),
        "num_test": len(test_df),
    }
    logger.info("[Task 1 eval] macro_f1=%.4f micro_f1=%.4f auc_pr=%.4f", result["macro_f1"], result["micro_f1"], result["auc_pr"])
    out_path = Path("results/metrics") / f"task1_eval_{run_dir.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved %s", out_path)


def run_task2_evaluation(config: dict, checkpoint_path: str, logger) -> None:
    """Reload a trained Task 2 GNN checkpoint and recompute test metrics
    independently of train.py, using the genre label map recorded in that
    run's metrics.json."""
    run_dir = Path(checkpoint_path).parent
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"{metrics_path} not found — run `python -m src.train --task 2` first.")
    with metrics_path.open() as f:
        train_metrics = json.load(f)
    label_map = train_metrics["label_map"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    subset = config["dataset"]["name"].replace("fma_", "")
    tracks = load_fma_metadata(config["dataset"]["metadata_root"], subset=subset)
    corrupted_ids = load_corrupted_track_ids()
    splits = load_fma_splits(tracks, exclude_track_ids=corrupted_ids)

    cache_dir = Path("data/processed/graph_cache")
    test_ds = SegmentGraphDataset(
        splits["test"], tracks_df=tracks, audio_root=config["dataset"]["root"], cache_dir=cache_dir,
        label_map=label_map, audio_config=config["audio"], graph_config=config["graph"],
        sample_rate=config["dataset"]["sample_rate"],
    )
    test_loader = PyGDataLoader(test_ds, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=4)

    in_dim = 2 * config["audio"]["n_mfcc"] + (24 if config["audio"]["use_chroma"] else 0)
    model = GNNGenreClassifier(
        in_dim=in_dim, hidden_dim=config["gnn"]["hidden_dim"], num_layers=config["gnn"]["num_layers"],
        num_classes=len(label_map), dropout=config["gnn"]["dropout"], model=config["gnn"]["model"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            all_preds += logits.argmax(-1).tolist()
            all_labels += batch.y.tolist()

    result = {
        "checkpoint": str(checkpoint_path),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "micro_f1": f1_score(all_labels, all_preds, average="micro", zero_division=0),
        "num_test": len(all_labels),
    }
    logger.info("[Task 2 eval] macro_f1=%.4f micro_f1=%.4f", result["macro_f1"], result["micro_f1"])
    out_path = Path("results/metrics") / f"task2_eval_{run_dir.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved %s", out_path)


def run_task4_evaluation(config: dict, checkpoint_path: str, logger) -> None:
    """Reload a trained Task 4 contrastive checkpoint and recompute test-set
    R@1/5/10 retrieval metrics (both directions) independently of train.py."""
    run_dir = Path(checkpoint_path).parent
    cc = config["contrastive"]
    df = load_musiccaps(Path(cc["dataset_root"]) / "musiccaps.csv", Path(cc["dataset_root"]) / "audio")
    splits = build_musiccaps_splits(df)
    split_of = {yid: name for name, ids in splits.items() for yid in ids}
    df = df.assign(split=df["ytid"].map(split_of))
    test_df = df[df["split"] == "test"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(config["bert"]["model_name"])
    cache_dir = Path("data/processed/musiccaps_graph_cache")
    test_ds = MusicCapsContrastiveDataset(
        test_df, cache_dir=cache_dir, audio_config=config["audio"], graph_config=config["graph"],
        sample_rate=config["dataset"]["sample_rate"], tokenizer=tokenizer, max_length=config["bert"]["max_length"],
    )
    test_loader = TorchDataLoader(test_ds, batch_size=config["training"]["batch_size"], shuffle=False, collate_fn=contrastive_collate)

    in_dim = 2 * config["audio"]["n_mfcc"] + (24 if config["audio"]["use_chroma"] else 0)
    model = ContrastiveDualEncoder(
        graph_in_dim=in_dim, graph_hidden_dim=config["gnn"]["hidden_dim"], graph_num_layers=config["gnn"]["num_layers"],
        graph_dropout=config["gnn"]["dropout"], bert_model_name=config["bert"]["model_name"],
        bert_freeze_layers=config["bert"]["freeze_layers"], embed_dim=cc["embed_dim"], temperature=cc["temperature"],
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    graph_embeds, text_embeds = [], []
    with torch.no_grad():
        for batch in test_loader:
            g = batch["graph_batch"].to(device)
            graph_embeds.append(model.encode_graph(g.x, g.edge_index, g.batch).cpu().numpy())
            text_embeds.append(model.encode_text(batch["input_ids"].to(device), batch["attention_mask"].to(device)).cpu().numpy())
    graph_embeds, text_embeds = np.concatenate(graph_embeds), np.concatenate(text_embeds)
    sim = graph_embeds @ text_embeds.T

    result = {
        "checkpoint": str(checkpoint_path),
        "audio_to_caption_r1": retrieval_recall_at_k(sim, 1),
        "audio_to_caption_r5": retrieval_recall_at_k(sim, 5),
        "audio_to_caption_r10": retrieval_recall_at_k(sim, 10),
        "caption_to_audio_r1": retrieval_recall_at_k(sim.T, 1),
        "caption_to_audio_r5": retrieval_recall_at_k(sim.T, 5),
        "caption_to_audio_r10": retrieval_recall_at_k(sim.T, 10),
        "num_test": len(test_df),
    }
    logger.info("[Task 4 eval] %s", result)
    out_path = Path("results/metrics") / f"task4_eval_{run_dir.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved %s", out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CSE425 GNN-BERT model.")
    parser.add_argument("--task", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--variant", type=str, default=None, choices=["early_concat", "cross_attention"],
        help="Task 3 only: which fusion variant's checkpoint is being evaluated for the "
        "t-SNE embedding. Defaults to inferring from the checkpoint filename "
        "(e.g. early_concat_best_model.pt -> early_concat).",
    )
    return parser.parse_args()


def _infer_task3_variant(checkpoint_path: str, explicit_variant: str | None) -> str:
    """Task 3 checkpoints are best_model.pt files named {variant}_best_model.pt for
    each of the 4 ablations. Only early_concat/cross_attention have a fused embedding
    z (via FusionModel.embed) suitable for the t-SNE plot, so this must match whichever
    checkpoint was actually passed in rather than always assuming cross_attention."""
    if explicit_variant is not None:
        return explicit_variant
    name = Path(checkpoint_path).name
    for variant in ("early_concat", "cross_attention"):
        if name.startswith(variant):
            return variant
    raise ValueError(
        f"Could not infer fusion variant from checkpoint filename '{name}'; pass --variant "
        "early_concat|cross_attention explicitly (gnn_only/bert_only checkpoints have no "
        "fused embedding z and aren't supported by run_task3_analysis)."
    )


def run_task3_analysis(config: dict, checkpoint_path: str, logger, variant: str | None = None) -> None:
    """Phase 10: t-SNE of the fused embedding z (colored by genre) + 3 case
    studies (graph structure + text + true/predicted tags across all
    4 ablation variants). Requires a completed `train_task3` run directory
    (with per-variant *_best_model.pt checkpoints and metrics.json). `variant`
    selects which fusion mode the given --checkpoint was trained with
    (early_concat or cross_attention); inferred from the checkpoint filename
    if not given explicitly."""
    variant = _infer_task3_variant(checkpoint_path, variant)
    logger.info("Task 3 fusion variant for t-SNE embedding: %s", variant)
    run_dir = Path(checkpoint_path).parent
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"{metrics_path} not found — run `python -m src.train --task 3` first.")
    with metrics_path.open() as f:
        task3_metrics = json.load(f)
    top_tags = task3_metrics["top_tags"]
    label_names = task3_metrics.get("label_names", top_tags)
    num_labels = len(label_names)
    num_genres = sum(1 for name in label_names if name.startswith("genre:"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    subset = config["dataset"]["name"].replace("fma_", "")
    tracks = load_fma_metadata(config["dataset"]["metadata_root"], subset=subset)
    corrupted_ids = load_corrupted_track_ids()
    splits = load_fma_splits(tracks, exclude_track_ids=corrupted_ids)
    genre_label_map = build_genre_label_map(tracks)
    task_df = build_task3_dataset(tracks, top_tags, genre_label_map)
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
        mode=variant,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_z, all_track_ids = [], []
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

    # Mood proxy: whichever curated mood-ish tag (if any) is positive in this track's
    # label vector, drawn from the top-K tag vocabulary itself (DEAM valence/arousal
    # would be the ideal mood signal per spec, but DEAM isn't available — see config.yaml).
    labels_by_track = dict(zip(test_df["track_id"], test_df["labels"]))
    mood_tags = [t for t in top_tags if t.lower() in MOOD_KEYWORDS]
    tag_start = num_genres

    def mood_of(track_id: int) -> str:
        labels = labels_by_track[track_id]
        for tag in mood_tags:
            idx = tag_start + top_tags.index(tag)
            if labels[idx] == 1:
                return tag
        return "none"

    all_moods = [mood_of(tid) for tid in all_track_ids]

    logger.info("Running t-SNE on %d test embeddings (dim=%d)...", len(all_z), all_z.shape[1])
    z_2d = TSNE(n_components=2, random_state=42, init="pca", perplexity=min(30, len(all_z) - 1)).fit_transform(all_z)

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    unique_genres = sorted(set(all_genres))
    cmap = plt.get_cmap("tab20", len(unique_genres))
    for i, genre in enumerate(unique_genres):
        idx = [j for j, g_ in enumerate(all_genres) if g_ == genre]
        axes[0].scatter(z_2d[idx, 0], z_2d[idx, 1], s=12, color=cmap(i), label=genre)
    axes[0].legend(fontsize=6, markerscale=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    axes[0].set_title("t-SNE of fused embedding z, colored by genre")

    unique_moods = sorted(set(all_moods))
    mood_cmap = plt.get_cmap("tab10", len(unique_moods))
    for i, mood in enumerate(unique_moods):
        idx = [j for j, m_ in enumerate(all_moods) if m_ == mood]
        axes[1].scatter(z_2d[idx, 0], z_2d[idx, 1], s=12, color=mood_cmap(i), label=mood)
    axes[1].legend(fontsize=7, markerscale=2, bbox_to_anchor=(1.02, 1), loc="upper left")
    axes[1].set_title("t-SNE of fused embedding z, colored by mood tag")
    plt.tight_layout()
    plt.savefig(run_dir / "tsne_genre_mood.png", dpi=150)
    plt.close()
    logger.info("Saved t-SNE plot to %s", run_dir / "tsne_genre_mood.png")

    _save_case_studies(config, run_dir, task3_metrics, tracks, test_df, top_tags, label_names, device, logger)


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


def _graph_path_summary(edge_index: torch.Tensor) -> dict:
    """Break edge_index into the temporal chain (deterministic |i-j|==1 edges) and
    the sparse similarity "shortcut" edges, for a human-readable graph-path view."""
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    temporal = sorted({(min(s, d), max(s, d)) for s, d in zip(src, dst) if abs(s - d) == 1})
    similarity = sorted({(min(s, d), max(s, d)) for s, d in zip(src, dst) if abs(s - d) > 1})
    return {
        "temporal_path": [f"seg{a}->seg{b}" for a, b in temporal],
        "similarity_shortcuts": [f"seg{a}~seg{b}" for a, b in similarity],
    }


def _save_case_studies(config, run_dir, task3_metrics, tracks, test_df, top_tags, label_names, device, logger) -> None:
    """3 case studies: graph path (temporal chain + similarity shortcuts) + text +
    true tags vs. each ablation variant's predicted tags (own tuned thresholds),
    plus cross-attention's graph-query -> text-token alignment weights (spec:
    "graph paths + caption/lyric alignment")."""
    rng = np.random.RandomState(42)
    sample_rows = test_df.iloc[rng.choice(len(test_df), size=min(3, len(test_df)), replace=False)]

    in_dim = 2 * config["audio"]["n_mfcc"] + (24 if config["audio"]["use_chroma"] else 0)
    num_labels = len(label_names)
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

        true_tags = [label_names[k] for k in range(num_labels) if row["labels"][k] == 1]
        predictions = {}
        text_alignment = None
        with torch.no_grad():
            for variant, model in models.items():
                if variant == "gnn_only":
                    logits = model(gx, gei, gb)
                elif variant == "bert_only":
                    logits = model(input_ids, attention_mask)
                else:
                    logits = model(gx, gei, gb, input_ids, attention_mask)
                probs = torch.sigmoid(logits)[0].cpu().numpy()
                pred_tags = [label_names[k] for k in range(num_labels) if probs[k] >= thresholds[variant][k]]
                predictions[variant] = pred_tags
                if variant == "cross_attention":
                    attn = model.attention_over_tokens(gx, gei, gb, input_ids, attention_mask)[0].cpu().numpy()
                    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().tolist())
                    top_k = attn.argsort()[::-1][:10]
                    text_alignment = [
                        {"token": tokens[i], "weight": round(float(attn[i]), 4)}
                        for i in top_k if tokens[i] not in ("[PAD]", "[CLS]", "[SEP]")
                    ]

        case_studies.append({
            "track_id": int(track_id),
            "genre": tracks.set_index("track_id").loc[track_id, "genre_top"],
            "text": row["text"][:300],
            "graph_path": _graph_path_summary(graph.edge_index),
            "num_graph_nodes": int(graph.x.shape[0]),
            "num_graph_edges": int(graph.edge_index.shape[1]),
            "true_tags": true_tags,
            "predicted_tags_by_variant": predictions,
            "cross_attention_text_alignment": text_alignment,
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
        run_task1_evaluation(config, args.checkpoint, logger)
        return
    if args.task == 2:
        run_task2_evaluation(config, args.checkpoint, logger)
        return
    if args.task == 3:
        run_task3_analysis(config, args.checkpoint, logger, variant=args.variant)
        return
    run_task4_evaluation(config, args.checkpoint, logger)


if __name__ == "__main__":
    main()
