# GNN-Based BERT for Understanding Context from Music

Course project (Neural Networks — CSE425 / EEE474 / CSE715): a hybrid **BERT + Graph
Neural Network** system for understanding musical context — multi-label tag/genre
classification, structural (GNN) modeling of audio segment graphs, and GNN–BERT fusion.

## Overview

A track is modeled as $T = (X_{audio}, X_{text}, G, y)$: audio segment features, text
(MusicCaps caption for Task 1, FMA artist bio for Task 3's fusion), a segment graph
$G=(V,E)$ (temporal + cosine-similarity edges over MFCC/chroma), and multi-label targets
$y$ (genre / top tags). Four tasks are implemented and evaluated end-to-end:

| Task | Description | Status |
|------|-------------|--------|
| 1 | BERT multi-label tag classifier (MusicCaps caption → tag proxy, spec 4.1) | Done |
| 2 | GraphSAGE on segment graphs vs. CNN mel-spectrogram baseline | Done |
| 3 | GNN–BERT fusion (early-concat / cross-attention) + ablations | Done |
| 4 | MusicCaps contrastive dual-encoder (InfoNCE, R@K retrieval) | Done |

## Dataset

- **FMA-medium** (25,000 tracks, 16 genres) — audio + genre/tag metadata, official
  artist-disjoint train/val/test split (verified: 0 artist overlap across splits).
- Only ~4,410 tracks have usable free-text tags; Task 3's tagged subset train/val/test =
  3,355/523/526 (Task 1 uses MusicCaps' own split, see below).
- 15 corrupted/truncated FMA-medium mp3s are excluded from all splits
  (`data/splits/fma_medium_corrupted_track_ids.json`).
- **MusicCaps** (5,521 clips, YouTube audio + human captions) — used for Task 1 (caption
  → tag proxy classifier, spec section 4.1) and for Task 4's contrastive retrieval. Audio
  isn't bundled by the dataset itself; 4,814/5,521 clips (87.2%) were successfully fetched
  via `yt-dlp` (`src/musiccaps_prep.py`) — the rest are permanently deleted/private/
  region-blocked YouTube videos, an expected and accepted attrition rate for this kind of
  dataset.
- Raw/processed audio and caches are **not** included in this repo (see `.gitignore`) —
  download FMA-medium/MusicCaps yourself and point `config.yaml`'s `dataset.root` at it.

## Data leakage safeguards

- Splits are artist-disjoint; normalizers/thresholds/class weights are fit on the
  train split only.
- Per-tag classification thresholds are tuned on validation probabilities only, then
  frozen for test.

## Modeling decisions

- Task 2 genre head uses **softmax + CrossEntropyLoss**, not the sigmoid/BCE written in
  the spec's general multi-label formula: FMA `genre_top` is single-label/mutually
  exclusive (one genre per track), so softmax+CE is the mathematically correct loss —
  BCE-per-class would incorrectly treat genres as independent binary decisions.

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
  contrastive.py         Task 4 dual-encoder, InfoNCE loss, R@K retrieval metric
  baselines.py          B1 majority/random baseline (+ optional B4 PCA+MLP)
  datasets.py            FMA + MusicCaps loading, splits, Task 1 tag-subset construction
  musiccaps_prep.py      one-off MusicCaps metadata/audio acquisition (yt-dlp)
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

`ffmpeg` must also be installed and on `PATH` (system package, not pip-installable) —
`librosa`/`soundfile` and `yt-dlp`'s audio extraction/trimming both depend on it
(e.g. `sudo apt install ffmpeg` on Debian/Ubuntu, `brew install ffmpeg` on macOS).

Download FMA-medium + FMA metadata from https://os.unil.cloud.switch.ch/fma/fma_medium.zip
and https://os.unil.cloud.switch.ch/fma/fma_metadata.zip into `data/raw/fma_medium/` and
`data/raw/fma_metadata/` to reproduce results.

## Running

```bash
python -m src.train --task 2 --learning-rate 3e-4 --patience 10   # GNN + CNN baseline
python -m src.musiccaps_prep --fetch-metadata --download-audio     # one-off MusicCaps acquisition
python -m src.train --task 1 --freeze-layers 2                     # BERT tag classifier (MusicCaps caption -> tag)
python -m src.train --task 3 --freeze-layers 2                     # fusion + ablations
python -m src.train --task 4                                       # contrastive dual-encoder
python -m src.evaluate --task {1,2,3,4} --checkpoint PATH
pytest tests/
```

## Results

**Task 1 — BERT tag classifier** (DistilBERT, top 2 layers unfrozen, MusicCaps caption →
top-50 aspect-tag proxy per spec section 4.1, test set):

| Macro-F1 | Micro-F1 | AUC-PR |
|---|---|---|
| 0.444 | 0.511 | 0.453 |

Captions directly describe the audio's musical content, so this is a strong, direct
text→tag signal (see `results/task1/20260906-230137/attention_examples.png` for 5 example
predictions with CLS-token attention visualization).

**Task 2 — GNN on segment graphs vs. CNN baseline** (16-way genre classification, test set):

| Model | Macro-F1 | Micro-F1 |
|---|---|---|
| B1 majority/random | 0.027 | 0.276 |
| GNN (GraphSAGE, segment graph) | 0.307 | 0.532 |
| B2 CNN (log-mel spectrogram) | 0.312 | 0.614 |

**Task 3 — GNN–BERT fusion ablations** (genre + contextual/music tags — including
mood-related tags like `psychedelic`/`dark`/`calm` but also non-mood ones like
`electronic`/`idm` — multi-label target, 36-way, on the FMA tagged subset described
above, test set):

| Variant | Macro-F1 | Micro-F1 | AUC-PR |
|---|---|---|---|
| GNN-only | 0.152 | 0.152 | 0.196 |
| BERT-only | 0.139 | 0.146 | 0.232 |
| Early concat | 0.202 | 0.246 | 0.287 |
| Cross-attention | 0.153 | 0.193 | 0.199 |

Early-concat fusion wins outright on all three metrics — combining graph and text
modalities beats either alone once the target vector combines genre and contextual/music
tag labels together (per spec section 4.3, which calls for "genre + mood tags"; the
actual top-20 tag vocabulary here is broader than mood alone — see Data leakage/Modeling
notes).

**Task 4 — MusicCaps contrastive retrieval** (dual-encoder, InfoNCE, 4,814/5,521 clips
downloaded, test set n=2,514 via `is_audioset_eval` split):

| Direction | R@1 | R@5 | R@10 |
|---|---|---|---|
| Audio → caption | 0.36% | 1.91% | 3.38% |
| Caption → audio | 0.44% | 2.31% | 3.70% |

Random chance at this pool size (2,514 candidates) is ~0.4% for R@10, so the model is
roughly 8-9x better than chance — a modest but real retrieval signal given the small
training set (2,070 pairs). Zero-shot caption→tag classification (nearest-prototype over
top-50 aspect names using the caption's own text embedding, no threshold tuning):
macro_f1=0.074, micro_f1=0.093 (not directly comparable to Task 3's supervised FMA
numbers — different dataset/vocabulary).

Full metrics, training curves, t-SNE plots, and case studies are in `results/task{1,2,3,4}/`.

## Tests

```bash
pytest tests/
```

Covers dataset loading/splits (artist-disjointness, corrupted-track exclusion), audio
feature extraction, graph construction, tensor shapes, and no-label-leakage checks.