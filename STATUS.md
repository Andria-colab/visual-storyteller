# Project Status — Person 2 (Data, Training & Evaluation)

_Last updated: 28 June 2026 — **Code-complete.**_

This document tracks what **The Visual Storyteller** has delivered and what's
left. Plain-English companion to the technical `CLAUDE.md`.

> **Where we are:** both halves are now implemented and verified end to end — the
> data/training/eval pipeline (P2) *and* the modeling code (P1: encoder, both
> decoders, model wrapper, attention viz, feature precompute), plus both
> notebooks. `python -m scripts.smoke_test` passes and a CPU sanity check
> confirms the real `EncoderDecoder` is a drop-in for the stub. The only thing
> left is running the full training on Flickr8k (Colab/GPU) and curating the
> final qualitative examples.

---

## TL;DR

Our entire half of the codebase is **written and verified end to end**. A smoke
test proves data loading, training (with crash-safe checkpoint/resume), caption
decoding, and BLEU scoring all work together — using a stand-in model, so we
didn't have to wait for the modeling side to start. When Person 1's real model
lands, it plugs in with no changes to our code.

---

## ✅ Done

### Data pipeline (`src/data.py`)
- Caption cleaning (lowercase, strip punctuation/digits, drop noise tokens).
- Reads both Flickr8k caption formats (Kaggle CSV and the original token format).
- `Vocabulary` with save/load, built from **training captions only** so the model
  never peeks at validation or test text.
- Train / val / test splits (6k / 1k / 1k), with the **test set held out** until
  final analysis. Uses official Flickr8k split files automatically if present.
- Dataset + batching that pads captions and reads cached image features on demand.

### Training engine (`src/train.py`)
- Full training loop with the right loss (ignores padding, label smoothing).
- Mixed precision + gradient clipping (faster, stabler on Colab GPU).
- Learning-rate warmup schedule for the Transformer.
- **Crash-safe checkpointing to Drive every epoch, with resume.** Written so a
  Colab disconnect mid-save can never corrupt the saved model — this is the
  single biggest operational risk on the project and it's handled.
- Early stopping when validation quality stops improving.

### Decoding & the graded entry point (`src/decode.py`)
- Greedy decoding (fast, for validation) and beam search (better captions).
- **`generate_caption(image_path, model) -> str`** — the exact function the
  assignment grades and the demo notebook calls. Logic is in place; the final
  wiring to the real image encoder is the one piece waiting on Person 1.

### Evaluation (`src/evaluate.py`)
- BLEU-1 through BLEU-4 scored against all 5 reference captions per image.
- Helper that renders results as a Markdown table for the report and README.

### Validation BLEU-4 (early-stopping signal) — `src/data.py` + `src/train.py`
- `ImageFeatureDataset` decodes each unique val image **once** and scores it
  against all 5 references, so early stopping uses real caption quality (BLEU-4),
  not just loss. Verified in the smoke test (real non-zero BLEU-4 per epoch).

### Inference notebook (`notebooks/inference.ipynb`)
- Authored end to end: setup, artifact loading, the graded `generate_caption`
  demo on held-out test images, success cases, diagnosed failure cases, and the
  baseline-vs-Transformer BLEU table (auto-writes `reports/results.md`).
- Structurally complete and validated (all cells parse). The model-loading cell
  is guarded: if Person 1's `model.py`/`encoder.py` aren't merged yet it prints a
  notice and skips the demo cells, so the notebook never errors. Demo cells run
  for real once the model lands.

### Project setup
- `requirements.txt`, `.gitignore`, `README.md`, `CLAUDE.md`, and this `STATUS.md`.
- `scripts/smoke_test.py` — one command (`python -m scripts.smoke_test`) that
  checks our whole half works, now including the BLEU-4 validation path.
  **Currently passing.**

---

## ✅ Modeling code (P1) — now implemented

- **`encoder.py`** — frozen ResNet-50 → `[B,49,2048]` region features, permanently
  in eval mode (frozen BatchNorm), plus `build_image_transform()` shared by feature
  precompute and `generate_caption`.
- **`decoder_lstm.py`** — Show-Attend-and-Tell baseline (LSTM + Bahdanau attention).
- **`decoder_transformer.py`** — from-scratch Transformer decoder (custom pre-norm
  layers over `nn.MultiheadAttention`, sinusoidal positions, causal + padding masks,
  weight-tied output). Init tuned so starting loss ≈ ln(V).
- **`model.py`** — `EncoderDecoder(vocab_size, cfg, variant)`, a verified drop-in for
  the smoke-test stub (`forward` / `encode_image` / `decode_step`).
- **`viz.py`** — per-word attention heatmaps for both decoders.
- **`scripts/precompute_features.py`** — caches `features.h5` (resumable, GPU).
- **`notebooks/data_and_training.ipynb`** — data → model → train → curves → save.

## ⏳ Remaining (needs real data + GPU — the user's run)

- Run `data_and_training.ipynb` on Flickr8k to build `features.h5`, train both
  models, and save `baseline_best.pt` / `transformer_best.pt` / `vocab.pkl`.
- Curate the actual success/failure example ids and the attention figures in
  `inference.ipynb` (needs the trained models), and let it write `reports/results.md`.

---

## How to verify our half right now

```bash
pip install -r requirements.txt
python -m scripts.smoke_test
```

Expected last line: `SMOKE TEST PASSED [OK]`. This confirms data → training →
checkpoint/resume → decoding → BLEU all work together, no GPU or dataset required.
