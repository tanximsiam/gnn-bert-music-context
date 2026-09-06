"""Entry point for training. Usage: python -m src.train --task {1,2,3,4} [--config config.yaml]

Each task's actual training loop is implemented in its corresponding phase
(Task 1: Phase 6, Task 2: Phases 4-5, Task 3: Phases 7-9, Task 4: Phase 12).
This script owns the shared boilerplate: config loading, seeding, logging,
and unique run-directory creation, so every task follows it consistently.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.loader import DataLoader
from transformers import AutoTokenizer

from src.baselines import majority_random_baseline
from src.bert_encoder import BertTagClassifier
from src.cnn_baseline import MelSpecCNN, MelSpecDataset
from src.datasets import (
    build_task1_dataset,
    build_task1_top_tags,
    build_task3_dataset,
    load_corrupted_track_ids,
    load_fma_metadata,
    load_fma_splits,
    task3_label_names,
)
from src.gnn_model import GNNGenreClassifier
from src.graph_builder import SegmentGraphDataset, build_genre_label_map, build_or_load_track_graph
from src.fusion_model import FusionModel
from src.utils import get_logger, load_config, make_run_dir, save_run_metadata, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CSE425 GNN-BERT model task.")
    parser.add_argument("--task", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--learning-rate", type=float, default=None, help="override config training.learning_rate")
    parser.add_argument("--epochs", type=int, default=None, help="override config training.epochs")
    parser.add_argument("--patience", type=int, default=None, help="override config training.early_stopping_patience")
    parser.add_argument("--freeze-layers", type=str, default=None, help="override config bert.freeze_layers")
    return parser.parse_args()


def tune_per_tag_thresholds(val_labels: np.ndarray, val_probs: np.ndarray) -> np.ndarray:
    """Pick, per tag, the probability threshold in [0.05, 0.95] that maximizes
    F1 on VALIDATION data only (never test) — a fixed 0.5 threshold is not
    meaningful once pos_weight has shifted the sigmoid calibration."""
    candidates = np.linspace(0.05, 0.95, 19)
    num_tags = val_labels.shape[1]
    thresholds = np.full(num_tags, 0.5)
    for k in range(num_tags):
        if val_labels[:, k].sum() == 0:
            continue
        best_f1, best_t = -1.0, 0.5
        for t in candidates:
            preds_k = (val_probs[:, k] >= t).astype(int)
            f1 = f1_score(val_labels[:, k], preds_k, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[k] = best_t
    return thresholds


class TagTextDataset(TorchDataset):
    """Tokenizes Task 1 (text, multi-hot tag labels) rows on the fly."""

    def __init__(self, df: Any, tokenizer: Any, max_length: int):
        self.texts = df["text"].tolist()
        self.labels = df["labels"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class MultiModalTagDataset(TorchDataset):
    """Task 3: pairs each track's segment graph with its text and
    multi-hot tag labels, for the fusion ablations."""

    def __init__(
        self,
        df: Any,
        audio_root: str,
        cache_dir: Path,
        audio_config: dict[str, Any],
        graph_config: dict[str, Any],
        sample_rate: int,
        tokenizer: Any,
        max_length: int,
    ):
        self.rows = df.reset_index(drop=True)
        self.audio_root = audio_root
        self.cache_dir = cache_dir
        self.audio_config = audio_config
        self.graph_config = graph_config
        self.sample_rate = sample_rate
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows.iloc[idx]
        graph = build_or_load_track_graph(
            row["track_id"],
            self.audio_root,
            self.cache_dir,
            self.sample_rate,
            self.audio_config["segment_seconds"],
            self.audio_config["n_mfcc"],
            self.audio_config["use_chroma"],
            self.graph_config["similarity_threshold"],
            self.graph_config["bidirectional_edges"],
            self.graph_config["self_loops"],
        ).clone()
        enc = self.tokenizer(
            row["text"], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        return {
            "graph": graph,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(row["labels"], dtype=torch.float32),
        }


def fusion_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "graph_batch": PyGBatch.from_data_list([item["graph"] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }


def train_task1(config: dict[str, Any], run_dir: Path, logger) -> None:
    """Task 1: multi-label BERT tag classifier on artist-disjoint splits."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device: %s", device)

    subset = config["dataset"]["name"].replace("fma_", "")
    tracks = load_fma_metadata(config["dataset"]["metadata_root"], subset=subset)
    corrupted_ids = load_corrupted_track_ids()
    splits = load_fma_splits(tracks, exclude_track_ids=corrupted_ids)

    # Target tag vocabulary chosen from TRAIN split only (rule: no test-set peeking
    # during feature/target selection).
    top_tags = build_task1_top_tags(tracks[tracks["track_id"].isin(splits["train"])], top_k=20)
    logger.info("Task 1 target tag vocabulary (top-20, from TRAIN only): %s", top_tags)

    task1_df = build_task1_dataset(tracks, top_tags)
    split_of = {tid: name for name, ids in splits.items() for tid in ids}
    task1_df = task1_df.assign(split=task1_df["track_id"].map(split_of))
    task1_df = task1_df[task1_df["split"].notna()]

    train_df = task1_df[task1_df["split"] == "train"]
    val_df = task1_df[task1_df["split"] == "val"]
    test_df = task1_df[task1_df["split"] == "test"]
    logger.info("Task 1 dataset sizes: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    tokenizer = AutoTokenizer.from_pretrained(config["bert"]["model_name"])
    max_length = config["bert"]["max_length"]
    train_ds = TagTextDataset(train_df, tokenizer, max_length)
    val_ds = TagTextDataset(val_df, tokenizer, max_length)
    test_ds = TagTextDataset(test_df, tokenizer, max_length)

    batch_size = config["training"]["batch_size"]
    train_loader = TorchDataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = TorchDataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = TorchDataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = BertTagClassifier(
        config["bert"]["model_name"], num_tags=len(top_tags), freeze_layers=config["bert"]["freeze_layers"]
    ).to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable_params, lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"]
    )

    # Per-tag pos_weight: tags are sparse (~1-3 positive out of 20 per track), so plain
    # BCE lets the model collapse to "predict nothing" (trivially low loss, F1=0 - the
    # exact failure mode observed empirically). sqrt-smoothed + clipped, computed from
    # TRAIN labels only (same recipe as the Task 2 class-weighting fix).
    train_labels_arr = np.array(train_df["labels"].tolist(), dtype=np.float32)
    num_pos = train_labels_arr.sum(axis=0)
    num_neg = len(train_labels_arr) - num_pos
    pos_weight = np.sqrt(num_neg / np.clip(num_pos, 1, None))
    pos_weight = np.clip(pos_weight, None, 10.0)
    logger.info("Task 1 pos_weight (train-only, sqrt-smoothed, clipped): %s", pos_weight.round(2).tolist())
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32).to(device))

    def mean_auc_pr(labels: np.ndarray, probs: np.ndarray) -> float:
        scores = [
            average_precision_score(labels[:, k], probs[:, k]) for k in range(labels.shape[1]) if labels[:, k].sum() > 0
        ]
        return float(np.mean(scores)) if scores else 0.0

    def run_epoch(loader: TorchDataLoader, train: bool) -> tuple[float, np.ndarray, np.ndarray]:
        model.train(train)
        total_loss, n = 0.0, 0
        all_probs, all_labels = [], []
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.set_grad_enabled(train):
                logits = model(input_ids, attention_mask)
                loss = loss_fn(logits, labels)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * input_ids.size(0)
            n += input_ids.size(0)
            all_probs += torch.sigmoid(logits).detach().cpu().tolist()
            all_labels += labels.detach().cpu().tolist()
        return total_loss / n, np.array(all_probs), np.array(all_labels)

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "val_macro_f1": [], "val_micro_f1": [], "val_auc_pr": [],
    }
    best_val_auc_pr = -1.0
    epochs_without_improvement = 0
    best_state = None
    patience = config["training"]["early_stopping_patience"]

    for epoch in range(config["training"]["epochs"]):
        train_loss, _, _ = run_epoch(train_loader, train=True)
        val_loss, val_probs, val_labels = run_epoch(val_loader, train=False)
        val_preds = (val_probs >= 0.5).astype(int)
        val_macro_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        val_micro_f1 = f1_score(val_labels, val_preds, average="micro", zero_division=0)
        val_auc_pr = mean_auc_pr(val_labels, val_probs)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_macro_f1)
        history["val_micro_f1"].append(val_micro_f1)
        history["val_auc_pr"].append(val_auc_pr)
        logger.info(
            "[BERT] epoch %d: train_loss=%.4f val_loss=%.4f val_macro_f1=%.4f val_micro_f1=%.4f val_auc_pr=%.4f",
            epoch, train_loss, val_loss, val_macro_f1, val_micro_f1, val_auc_pr,
        )

        if val_auc_pr > best_val_auc_pr:
            best_val_auc_pr = val_auc_pr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("[BERT] early stopping at epoch %d", epoch)
                break

    model.load_state_dict(best_state)
    torch.save(best_state, run_dir / "bert_best_model.pt")

    # Per-tag thresholds tuned on VALIDATION probabilities from the best checkpoint
    # (not the fixed 0.5 default), then applied unchanged to the test set.
    _, best_val_probs, best_val_labels = run_epoch(val_loader, train=False)
    thresholds = tune_per_tag_thresholds(best_val_labels, best_val_probs)
    logger.info("Task 1 per-tag thresholds (tuned on val only): %s", thresholds.round(2).tolist())

    test_loss, test_probs, test_labels = run_epoch(test_loader, train=False)
    test_preds = (test_probs >= thresholds).astype(int)
    test_macro_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    test_micro_f1 = f1_score(test_labels, test_preds, average="micro", zero_division=0)
    test_auc_pr = mean_auc_pr(test_labels, test_probs)
    logger.info(
        "[BERT] TEST: macro_f1=%.4f micro_f1=%.4f auc_pr=%.4f", test_macro_f1, test_micro_f1, test_auc_pr
    )

    test_texts = test_df["text"].tolist()
    examples = []
    for i in range(min(5, len(test_texts))):
        pred_tags = [top_tags[k] for k in range(len(top_tags)) if test_preds[i][k] == 1]
        true_tags = [top_tags[k] for k in range(len(top_tags)) if test_labels[i][k] == 1]
        examples.append({"text": test_texts[i][:300], "true_tags": true_tags, "predicted_tags": pred_tags})

    metrics = {
        "top_tags": top_tags,
        "history": history,
        "test": {"macro_f1": test_macro_f1, "micro_f1": test_micro_f1, "auc_pr": test_auc_pr, "loss": test_loss},
        "per_tag_thresholds": thresholds.tolist(),
        "example_predictions": examples,
        "dataset_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    }
    with (run_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Task 1 BERT loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(history["val_macro_f1"], label="val macro-F1")
    axes[1].plot(history["val_micro_f1"], label="val micro-F1")
    axes[1].plot(history["val_auc_pr"], label="val AUC-PR")
    axes[1].set_title("Task 1 BERT validation metrics")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(run_dir / "training_curves.png", dpi=130)
    plt.close()


def compute_class_weights(
    tracks: Any, train_ids: list[int], label_map: dict[str, int], num_classes: int
) -> np.ndarray:
    """Sqrt-smoothed, clipped inverse-frequency class weights computed from the
    TRAIN split only (no leakage). See Phase 4 tuning notes: raw inverse-freq
    weights (up to ~96x) destabilize training; sqrt + clip(max=5) works well."""
    train_genre_counts = tracks[tracks["track_id"].isin(train_ids)]["genre_top"].map(label_map).value_counts()
    class_counts = np.array([train_genre_counts.get(i, 0) for i in range(num_classes)], dtype=np.float32)
    class_weights = np.sqrt(class_counts.sum() / (num_classes * np.clip(class_counts, 1, None)))
    return np.clip(class_weights, None, 5.0)


def train_cnn_baseline(
    config: dict[str, Any],
    run_dir: Path,
    logger,
    tracks: Any,
    splits: dict[str, list[int]],
    label_map: dict[str, int],
    device: str,
) -> dict[str, Any]:
    """B2 baseline: small CNN on fixed-size log-mel spectrograms, no graph/text."""
    num_classes = len(label_map)
    cache_dir = Path("data/processed/melspec_cache")
    common_args = dict(
        tracks_df=tracks,
        audio_root=config["dataset"]["root"],
        cache_dir=cache_dir,
        label_map=label_map,
        sample_rate=config["dataset"]["sample_rate"],
        n_mels=config["audio"]["n_mels"],
    )
    train_ds = MelSpecDataset(splits["train"], **common_args)
    val_ds = MelSpecDataset(splits["val"], **common_args)
    test_ds = MelSpecDataset(splits["test"], **common_args)

    batch_size = config["training"]["batch_size"]
    train_loader = TorchDataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = TorchDataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = TorchDataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    model = MelSpecCNN(n_mels=config["audio"]["n_mels"], num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=config["training"]["weight_decay"])
    class_weights = compute_class_weights(tracks, splits["train"], label_map, num_classes)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))

    def run_epoch(loader: TorchDataLoader, train: bool) -> tuple[float, list[int], list[int]]:
        model.train(train)
        total_loss, n = 0.0, 0
        all_preds, all_labels = [], []
        for batch in loader:
            x, y = batch["x"].to(device), batch["y"].to(device)
            with torch.set_grad_enabled(train):
                logits = model(x)
                loss = loss_fn(logits, y)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
            all_preds += logits.argmax(-1).tolist()
            all_labels += y.tolist()
        return total_loss / n, all_preds, all_labels

    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_micro_f1": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None
    patience = config["training"]["early_stopping_patience"]

    for epoch in range(config["training"]["epochs"]):
        train_loss, _, _ = run_epoch(train_loader, train=True)
        val_loss, val_preds, val_labels = run_epoch(val_loader, train=False)
        val_macro_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        val_micro_f1 = f1_score(val_labels, val_preds, average="micro", zero_division=0)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_macro_f1)
        history["val_micro_f1"].append(val_micro_f1)
        logger.info(
            "[CNN] epoch %d: train_loss=%.4f val_loss=%.4f val_macro_f1=%.4f val_micro_f1=%.4f",
            epoch, train_loss, val_loss, val_macro_f1, val_micro_f1,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info("[CNN] early stopping at epoch %d", epoch)
                break

    model.load_state_dict(best_state)
    torch.save(best_state, run_dir / "cnn_best_model.pt")

    test_loss, test_preds, test_labels = run_epoch(test_loader, train=False)
    test_macro_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    test_micro_f1 = f1_score(test_labels, test_preds, average="micro", zero_division=0)
    logger.info("[CNN] TEST: cnn_macro_f1=%.4f cnn_micro_f1=%.4f", test_macro_f1, test_micro_f1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("B2 CNN loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(history["val_macro_f1"], label="val macro-F1")
    axes[1].plot(history["val_micro_f1"], label="val micro-F1")
    axes[1].set_title("B2 CNN validation F1")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(run_dir / "cnn_training_curves.png", dpi=130)
    plt.close()

    return {
        "history": history,
        "test": {"cnn_macro_f1": test_macro_f1, "cnn_micro_f1": test_micro_f1, "cnn_loss": test_loss},
    }


def train_task3(config: dict[str, Any], run_dir: Path, logger) -> None:
    """Task 3: GNN+BERT fusion ablations (BERT-only, GNN-only, early-concat,
    cross-attention) predicting genre + mood/contextual tags jointly (spec
    section 4.3), on the same tagged subset/splits so all four variants are
    directly comparable."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device: %s", device)

    subset = config["dataset"]["name"].replace("fma_", "")
    tracks = load_fma_metadata(config["dataset"]["metadata_root"], subset=subset)
    corrupted_ids = load_corrupted_track_ids()
    splits = load_fma_splits(tracks, exclude_track_ids=corrupted_ids)
    top_tags = build_task1_top_tags(tracks[tracks["track_id"].isin(splits["train"])], top_k=20)
    genre_label_map = build_genre_label_map(tracks)
    label_names = task3_label_names(top_tags, genre_label_map)
    logger.info("Task 3 target vocabulary (%d genres + top-20 tags, from TRAIN only): %s", len(genre_label_map), label_names)

    task_df = build_task3_dataset(tracks, top_tags, genre_label_map)
    split_of = {tid: name for name, ids in splits.items() for tid in ids}
    task_df = task_df.assign(split=task_df["track_id"].map(split_of))
    task_df = task_df[task_df["split"].notna()]

    train_df = task_df[task_df["split"] == "train"]
    val_df = task_df[task_df["split"] == "val"]
    test_df = task_df[task_df["split"] == "test"]
    logger.info("Task 3 dataset sizes: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    # Per-tag pos_weight (same recipe as Task 1 / Task 2 tuning): tags are sparse
    # (~1-3 positive out of 20 per track), so plain BCE collapses to "predict
    # nothing". Computed from TRAIN labels only, shared across all 4 ablations
    # for a fair comparison.
    train_labels_arr = np.array(train_df["labels"].tolist(), dtype=np.float32)
    num_pos = train_labels_arr.sum(axis=0)
    num_neg = len(train_labels_arr) - num_pos
    pos_weight_arr = np.clip(np.sqrt(num_neg / np.clip(num_pos, 1, None)), None, 10.0)
    logger.info("Task 3 pos_weight (train-only, sqrt-smoothed, clipped): %s", pos_weight_arr.round(2).tolist())

    tokenizer = AutoTokenizer.from_pretrained(config["bert"]["model_name"])
    max_length = config["bert"]["max_length"]
    cache_dir = Path("data/processed/graph_cache")
    common_ds_args = dict(
        audio_root=config["dataset"]["root"],
        cache_dir=cache_dir,
        audio_config=config["audio"],
        graph_config=config["graph"],
        sample_rate=config["dataset"]["sample_rate"],
        tokenizer=tokenizer,
        max_length=max_length,
    )
    train_ds = MultiModalTagDataset(train_df, **common_ds_args)
    val_ds = MultiModalTagDataset(val_df, **common_ds_args)
    test_ds = MultiModalTagDataset(test_df, **common_ds_args)

    batch_size = config["training"]["batch_size"]
    train_loader = TorchDataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=fusion_collate)
    val_loader = TorchDataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=fusion_collate)
    test_loader = TorchDataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=fusion_collate)

    in_dim = 2 * config["audio"]["n_mfcc"] + (24 if config["audio"]["use_chroma"] else 0)
    num_labels = len(label_names)

    def mean_auc_pr(labels: np.ndarray, probs: np.ndarray) -> float:
        scores = [
            average_precision_score(labels[:, k], probs[:, k]) for k in range(labels.shape[1]) if labels[:, k].sum() > 0
        ]
        return float(np.mean(scores)) if scores else 0.0

    def make_model(variant: str) -> torch.nn.Module:
        if variant == "gnn_only":
            return GNNGenreClassifier(
                in_dim=in_dim,
                hidden_dim=config["gnn"]["hidden_dim"],
                num_layers=config["gnn"]["num_layers"],
                num_classes=num_labels,
                dropout=config["gnn"]["dropout"],
                model=config["gnn"]["model"],
            ).to(device)
        if variant == "bert_only":
            return BertTagClassifier(
                config["bert"]["model_name"], num_tags=num_labels, freeze_layers=config["bert"]["freeze_layers"]
            ).to(device)
        mode = "early_concat" if variant == "early_concat" else "cross_attention"
        return FusionModel(
            graph_in_dim=in_dim,
            graph_hidden_dim=config["gnn"]["hidden_dim"],
            graph_num_layers=config["gnn"]["num_layers"],
            graph_dropout=config["gnn"]["dropout"],
            bert_model_name=config["bert"]["model_name"],
            bert_freeze_layers=config["bert"]["freeze_layers"],
            num_labels=num_labels,
            mode=mode,
        ).to(device)

    def forward_pass(model: torch.nn.Module, variant: str, batch: dict[str, Any]) -> torch.Tensor:
        if variant == "gnn_only":
            g = batch["graph_batch"].to(device)
            return model(g.x, g.edge_index, g.batch)
        if variant == "bert_only":
            return model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        g = batch["graph_batch"].to(device)
        return model(g.x, g.edge_index, g.batch, batch["input_ids"].to(device), batch["attention_mask"].to(device))

    # GNN-from-scratch variants (gnn_only, early_concat, cross_attention) need a higher
    # LR than BERT fine-tuning, per the Phase 4 tuning findings; bert_only keeps the
    # config default (BERT-appropriate) learning rate.
    variant_lr = {"gnn_only": 3e-4, "bert_only": config["training"]["learning_rate"], "early_concat": 3e-4, "cross_attention": 3e-4}

    all_results: dict[str, Any] = {}
    embeddings_for_tsne: dict[str, Any] = {}

    for variant in ("gnn_only", "bert_only", "early_concat", "cross_attention"):
        logger.info("=== Task 3 ablation: %s ===", variant)
        model = make_model(variant)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=variant_lr[variant], weight_decay=config["training"]["weight_decay"])
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_arr, dtype=torch.float32).to(device))

        def run_epoch(loader: TorchDataLoader, train: bool) -> tuple[float, np.ndarray, np.ndarray]:
            model.train(train)
            total_loss, n = 0.0, 0
            all_probs, all_labels = [], []
            for batch in loader:
                labels = batch["labels"].to(device)
                with torch.set_grad_enabled(train):
                    logits = forward_pass(model, variant, batch)
                    loss = loss_fn(logits, labels)
                    if train:
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                bs = labels.size(0)
                total_loss += loss.item() * bs
                n += bs
                all_probs += torch.sigmoid(logits).detach().cpu().tolist()
                all_labels += labels.detach().cpu().tolist()
            return total_loss / n, np.array(all_probs), np.array(all_labels)

        history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [], "val_macro_f1": [], "val_micro_f1": [], "val_auc_pr": [],
        }
        best_val_auc_pr = -1.0
        epochs_without_improvement = 0
        best_state = None
        patience = config["training"]["early_stopping_patience"]

        for epoch in range(config["training"]["epochs"]):
            train_loss, _, _ = run_epoch(train_loader, train=True)
            val_loss, val_probs, val_labels = run_epoch(val_loader, train=False)
            val_preds = (val_probs >= 0.5).astype(int)
            val_macro_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
            val_micro_f1 = f1_score(val_labels, val_preds, average="micro", zero_division=0)
            val_auc_pr = mean_auc_pr(val_labels, val_probs)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_macro_f1"].append(val_macro_f1)
            history["val_micro_f1"].append(val_micro_f1)
            history["val_auc_pr"].append(val_auc_pr)
            logger.info(
                "[%s] epoch %d: train_loss=%.4f val_loss=%.4f val_macro_f1=%.4f val_micro_f1=%.4f val_auc_pr=%.4f",
                variant, epoch, train_loss, val_loss, val_macro_f1, val_micro_f1, val_auc_pr,
            )

            if val_auc_pr > best_val_auc_pr:
                best_val_auc_pr = val_auc_pr
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info("[%s] early stopping at epoch %d", variant, epoch)
                    break

        model.load_state_dict(best_state)
        torch.save(best_state, run_dir / f"{variant}_best_model.pt")

        _, best_val_probs, best_val_labels = run_epoch(val_loader, train=False)
        thresholds = tune_per_tag_thresholds(best_val_labels, best_val_probs)

        test_loss, test_probs, test_labels = run_epoch(test_loader, train=False)
        test_preds = (test_probs >= thresholds).astype(int)
        test_macro_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
        test_micro_f1 = f1_score(test_labels, test_preds, average="micro", zero_division=0)
        test_auc_pr = mean_auc_pr(test_labels, test_probs)
        logger.info(
            "[%s] TEST: macro_f1=%.4f micro_f1=%.4f auc_pr=%.4f", variant, test_macro_f1, test_micro_f1, test_auc_pr
        )

        all_results[variant] = {
            "history": history,
            "test": {"macro_f1": test_macro_f1, "micro_f1": test_micro_f1, "auc_pr": test_auc_pr, "loss": test_loss},
            "per_tag_thresholds": thresholds.tolist(),
        }
        embeddings_for_tsne[variant] = {"probs": test_probs.tolist(), "labels": test_labels.tolist()}

    metrics = {
        "top_tags": top_tags,
        "label_names": label_names,
        "dataset_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "ablations": all_results,
    }
    with (run_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for variant, result in all_results.items():
        axes[0].plot(result["history"]["val_loss"], label=variant)
        axes[1].plot(result["history"]["val_macro_f1"], label=variant)
        axes[2].plot(result["history"]["val_auc_pr"], label=variant)
    axes[0].set_title("Task 3 val loss")
    axes[1].set_title("Task 3 val macro-F1")
    axes[2].set_title("Task 3 val AUC-PR")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(run_dir / "ablation_curves.png", dpi=130)
    plt.close()

    logger.info("=== Task 3 ablation comparison (test set) ===")
    for variant, result in all_results.items():
        t = result["test"]
        logger.info("%s: macro_f1=%.4f micro_f1=%.4f auc_pr=%.4f", variant, t["macro_f1"], t["micro_f1"], t["auc_pr"])


def train_task2(config: dict[str, Any], run_dir: Path, logger) -> None:
    """Task 2: GraphSAGE/GAT genre classifier on FMA segment graphs, with early
    stopping on validation loss and comparison against the B1 majority baseline."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device: %s", device)

    subset = config["dataset"]["name"].replace("fma_", "")
    tracks = load_fma_metadata(config["dataset"]["metadata_root"], subset=subset)
    corrupted_ids = load_corrupted_track_ids()
    splits = load_fma_splits(tracks, exclude_track_ids=corrupted_ids)
    label_map = build_genre_label_map(tracks)
    num_classes = len(label_map)

    cache_dir = Path("data/processed/graph_cache")
    common_args = dict(
        tracks_df=tracks,
        audio_root=config["dataset"]["root"],
        cache_dir=cache_dir,
        label_map=label_map,
        audio_config=config["audio"],
        graph_config=config["graph"],
        sample_rate=config["dataset"]["sample_rate"],
    )
    train_ds = SegmentGraphDataset(splits["train"], **common_args)
    val_ds = SegmentGraphDataset(splits["val"], **common_args)
    test_ds = SegmentGraphDataset(splits["test"], **common_args)

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    in_dim = 2 * config["audio"]["n_mfcc"] + (24 if config["audio"]["use_chroma"] else 0)
    model = GNNGenreClassifier(
        in_dim=in_dim,
        hidden_dim=config["gnn"]["hidden_dim"],
        num_layers=config["gnn"]["num_layers"],
        num_classes=num_classes,
        dropout=config["gnn"]["dropout"],
        model=config["gnn"]["model"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"]
    )

    # Inverse-frequency class weights: FMA-medium genres range from ~7100 (Rock) down to
    # ~21 (Easy Listening) train tracks, so unweighted CE lets the model ignore rare
    # classes almost entirely, crushing macro-F1. Weights computed from TRAIN split only.
    # Sqrt-smoothed + clipped to avoid a single ultra-rare class (95x raw weight)
    # dominating gradients and destabilizing training.
    class_weights = compute_class_weights(tracks, splits["train"], label_map, num_classes)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
    logger.info("class weights (train-only, sqrt-smoothed, clipped): %s", class_weights.round(2).tolist())

    def run_epoch(loader: DataLoader, train: bool) -> tuple[float, list[int], list[int]]:
        model.train(train)
        total_loss, n = 0.0, 0
        all_preds, all_labels = [], []
        for batch in loader:
            batch = batch.to(device)
            with torch.set_grad_enabled(train):
                logits = model(batch.x, batch.edge_index, batch.batch)
                loss = loss_fn(logits, batch.y)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            n += batch.num_graphs
            all_preds += logits.argmax(-1).tolist()
            all_labels += batch.y.tolist()
        return total_loss / n, all_preds, all_labels

    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_micro_f1": []}
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(config["training"]["epochs"]):
        train_loss, _, _ = run_epoch(train_loader, train=True)
        val_loss, val_preds, val_labels = run_epoch(val_loader, train=False)
        val_macro_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        val_micro_f1 = f1_score(val_labels, val_preds, average="micro", zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_macro_f1)
        history["val_micro_f1"].append(val_micro_f1)
        logger.info(
            "epoch %d: train_loss=%.4f val_loss=%.4f val_macro_f1=%.4f val_micro_f1=%.4f",
            epoch, train_loss, val_loss, val_macro_f1, val_micro_f1,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["training"]["early_stopping_patience"]:
                logger.info("early stopping at epoch %d", epoch)
                break

    model.load_state_dict(best_state)
    torch.save(best_state, run_dir / "best_model.pt")

    test_loss, test_preds, test_labels = run_epoch(test_loader, train=False)
    test_macro_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
    test_micro_f1 = f1_score(test_labels, test_preds, average="micro", zero_division=0)

    majority_preds = majority_random_baseline(
        [label_map[g] for g in tracks[tracks["track_id"].isin(splits["train"])]["genre_top"]],
        num_samples=len(test_labels),
        mode="majority",
    )
    b1_macro_f1 = f1_score(test_labels, majority_preds, average="macro", zero_division=0)
    b1_micro_f1 = f1_score(test_labels, majority_preds, average="micro", zero_division=0)

    logger.info("TEST: gnn_macro_f1=%.4f gnn_micro_f1=%.4f", test_macro_f1, test_micro_f1)
    logger.info("TEST: b1_majority_macro_f1=%.4f b1_majority_micro_f1=%.4f", b1_macro_f1, b1_micro_f1)

    cnn_results = train_cnn_baseline(config, run_dir, logger, tracks, splits, label_map, device)

    metrics = {
        "gnn": {
            "history": history,
            "test": {
                "gnn_macro_f1": test_macro_f1,
                "gnn_micro_f1": test_micro_f1,
                "gnn_loss": test_loss,
            },
        },
        "cnn_baseline_b2": cnn_results,
        "b1_majority": {"macro_f1": b1_macro_f1, "micro_f1": b1_micro_f1},
        "label_map": label_map,
        "num_epochs_trained": len(history["train_loss"]),
    }
    with (run_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Task 2 GNN loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(history["val_macro_f1"], label="val macro-F1")
    axes[1].plot(history["val_micro_f1"], label="val micro-F1")
    axes[1].set_title("Task 2 GNN validation F1")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(run_dir / "training_curves.png", dpi=130)
    plt.close()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.patience is not None:
        config["training"]["early_stopping_patience"] = args.patience
    if args.freeze_layers is not None:
        config["bert"]["freeze_layers"] = args.freeze_layers
    set_seed(config["training"]["seed"])

    run_dir = make_run_dir(args.results_dir, f"task{args.task}")
    save_run_metadata(run_dir, config, split_name=config["dataset"]["split"])
    logger = get_logger(f"train.task{args.task}", log_file=run_dir / "train.log")
    logger.info("Run directory: %s", run_dir)

    if args.task == 1:
        train_task1(config, run_dir, logger)
        return
    if args.task == 2:
        train_task2(config, run_dir, logger)
        return
    if args.task == 3:
        train_task3(config, run_dir, logger)
        return
    raise NotImplementedError("Task 4 (optional) training implemented in Phase 12.")


if __name__ == "__main__":
    main()
