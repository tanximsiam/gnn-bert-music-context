# GNN-Based BERT for Understanding Context from Music

Course project (Neural Networks — CSE425 / EEE474 / CSE715): a hybrid **BERT + Graph
Neural Network** system for understanding musical context — multi-label tag/genre
classification, structural (GNN) modeling of audio segment graphs, and GNN–BERT fusion —
following the project spec (`Project Spec.pdf` / `Project Spec.txt`).

## Overview

A track is modeled as $T = (X_{audio}, X_{text}, G, y)$: audio segment features, text
(artist bio, leakage-masked), a segment graph $G=(V,E)$ (temporal + cosine-similarity
edges over MFCC/chroma), and multi-label targets $y$ (genre / top tags). Three of the
four spec tasks are implemented and evaluated end-to-end:

| Task | Description | Status |
|------|-------------|--------|
| 1 (Easy) | BERT multi-label tag classifier on (leakage-masked) artist text | Done |
| 2 (Medium) | GraphSAGE on segment graphs vs. CNN mel-spectrogram baseline | Done |
| 3 (Hard) | GNN–BERT fusion (early-concat / cross-attention) + ablations | Done |
| 4 (Advanced, optional) | Contrastive MusicCaps retrieval | Not attempted (bonus, out of scope for this run) |

## Dataset

- **FMA-medium** (25,000 tracks, 16 genres) — audio + genre/tag metadata, official
  artist-disjoint train/val/test split (verified: 0 artist overlap across splits).
- Only ~4,410 tracks have usable free-text tags; Task 1/3 train/val/test = 3,355/523/526.
- 15 corrupted/truncated FMA-medium mp3s are excluded from all splits
  (`data/splits/fma_medium_corrupted_track_ids.json`).
- **DEAM** (valence/arousal auxiliary loss) is wired into the config but disabled by
  default and was not used for the reported results.
- Raw/processed audio and caches are **not** included in this repo (see `.gitignore`) —
  download FMA-medium yourself and point `config.yaml`'s `dataset.root` at it.

## Data leakage safeguards

- Target genre/tag words are masked out of the BERT input text (artist bio) before
  tokenization, per-tag, using word-boundary regex.
- Splits are artist-disjoint; normalizers/thresholds/class weights are fit on the
  train split only.
- Per-tag classification thresholds are tuned on validation probabilities only, then
  frozen for test.

## Repository structure

```
config.yaml           single source of truth for all experiment settings
src/
  audio_features.py   load/resample/segment audio, MFCC+chroma node features
  graph_builder.py     segment graph construction (temporal + similarity edges), caching
  gnn_model.py          GraphSAGE / GAT encoder, mean-pool readout, genre classifier
  cnn_baseline.py       B2 CNN baseline on log-mel spectrograms
  bert_encoder.py       DistilBERT text encoder + multi-label tag classification head
  fusion_model.py        Task 3 early-concat and cross-attention fusion heads
  baselines.py          B1 majority/random baseline (+ optional B4 PCA+MLP)
  contrastive.py         Task 4 (optional) dual-encoder InfoNCE — not run in this project
  datasets.py            FMA/DEAM/MusicCaps loading, splits, Task 1 tag-subset+masking
  train.py / evaluate.py CLI entry points (--task {1,2,3,4})
  utils.py               seeding, config loading, logging, run directories
tests/                 pytest suite (dataset/graph/audio/shape/leakage checks)
notebooks/             eda.ipynb, demo_context.ipynb
data/splits/           precomputed FMA-medium splits, stats, corrupted-track list
results/               per-run metrics.json, training curves, plots (checkpoints excluded)
```

## Setup

```bash
conda create -y -n cse425 python=3.11
conda activate cse425
pip install -r requirements.txt
```

Download FMA-medium + FMA metadata (see `Project Spec.txt` Table 1 for official links)
into `data/raw/fma_medium/` and `data/raw/fma_metadata/` to reproduce results.

## Running

```bash
python -m src.train --task 2 --learning-rate 3e-4 --patience 10   # GNN + CNN baseline
python -m src.train --task 1 --freeze-layers 2                    # BERT tag classifier
python -m src.train --task 3                                       # fusion + ablations
python -m src.evaluate --task {1,2,3} --checkpoint PATH
pytest tests/
```

## Results

**Task 1 — BERT tag classifier** (DistilBERT, top 2 layers unfrozen, test set):

| Model | Macro-F1 | Micro-F1 | AUC-PR |
|---|---|---|---|
| BERT (masked artist bio) | 0.061 | 0.064 | 0.114 |

Masked artist-bio text is a weak signal for tags (bios describe artist backstory, not
musical style, and genre/tag words are stripped for leakage-safety) — a genuine, expected
result rather than a bug.

**Task 2 — GNN on segment graphs vs. CNN baseline** (16-way genre classification, test set):

| Model | Macro-F1 | Micro-F1 |
|---|---|---|
| B1 majority/random | 0.027 | 0.276 |
| GNN (GraphSAGE, segment graph) | 0.312 | 0.534 |
| B2 CNN (log-mel spectrogram) | 0.348 | 0.622 |

**Task 3 — GNN–BERT fusion ablations** (identical Task 1 tagged-subset splits, test set):

| Variant | Macro-F1 | Micro-F1 | AUC-PR |
|---|---|---|---|
| BERT-only | 0.055 | 0.069 | 0.124 |
| GNN-only | 0.121 | 0.108 | 0.133 |
| Early concat | 0.118 | 0.130 | 0.133 |
| Cross-attention | 0.104 | 0.104 | 0.143 |

GNN-only clearly beats BERT-only, consistent with the Task 1 finding that masked bio
text carries little tag signal — the graph/audio modality is the stronger one here.
Fusion (concat/cross-attention) gives the best AUC-PR and micro-F1 but doesn't
uniformly dominate every metric at this model scale/data size.

Full metrics, training curves, t-SNE plots, and case studies are in `results/task{1,2,3}/`.

## Tests

```bash
pytest tests/
```

Covers dataset loading/splits (artist-disjointness, corrupted-track exclusion), audio
feature extraction, graph construction, tensor shapes, and no-label-leakage checks.