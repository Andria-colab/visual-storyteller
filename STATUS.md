# Project Status — Person 2 (Data, Training & Evaluation)

_Last updated: 26 June 2026 — **PAUSED at a clean checkpoint.**_

This document tracks what our half of **The Visual Storyteller** has delivered and
what we're waiting on. Plain-English companion to the technical `CLAUDE.md`.

> **Where we paused:** every Person-2 task that can be done without Person 1's
> model is finished and the smoke test passes (`python -m scripts.smoke_test` →
> `SMOKE TEST PASSED [OK]`). Nothing is half-built. The next moves are all blocked
> on Person 1's `model.py` / `features.h5` / encoder transform, or on the Day-2
> interface freeze. Pick back up from the "Blocked / waiting" section below.

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

## ⏳ Blocked / waiting

### On Person 1 (modeling)
| What we need | Why | Impact if late |
|---|---|---|
| Real `model.py` (EncoderDecoder) | Replaces the stub we test against | None yet — stub keeps us unblocked; needed before real training |
| `features.h5` from `precompute_features.py` | Our dataset reads cached image features from it | Can't run real training until it exists |
| Encoder + image transform attached to the model | `generate_caption` needs them to turn a raw image into features | `generate_caption` can't run on real images until then |

### On the Day-2 interface freeze (joint decision with Person 1)
These are quick agreements that prevent rework — none are hard problems:
1. **Step API:** confirm the model exposes `decode_step` (our decoder assumes it);
   if it's a single `forward` instead, we adapt in one spot.
2. **Feature file keys:** confirm `features.h5` is keyed by image filename.
3. **Model attributes:** confirm we attach `.vocab`, `.encoder`, `.image_transform`
   to the model object for `generate_caption`.

### Next on our own plate (not blocked)
- _(done)_ ~~id-aware validation loader for real BLEU-4 early stopping~~
- _(done)_ ~~author `notebooks/inference.ipynb`~~
- Curate the actual success/failure example ids in `inference.ipynb` — needs the
  trained models to run, so effectively waiting on Person 1.
- Draft the training section of `data_and_training.ipynb` (loss curves, artifact
  saving) — can scaffold now against the stub, finalises once the real model lands.

---

## How to verify our half right now

```bash
pip install -r requirements.txt
python -m scripts.smoke_test
```

Expected last line: `SMOKE TEST PASSED [OK]`. This confirms data → training →
checkpoint/resume → decoding → BLEU all work together, no GPU or dataset required.
