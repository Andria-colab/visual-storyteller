# The Visual Storyteller

Image captioning on Flickr8k — a shared frozen **ResNet-50** encoder feeding two
decoders that are compared head to head:

- **Baseline** — LSTM + Bahdanau attention ("Show, Attend and Tell")
- **Main** — from-scratch Transformer decoder (masked self-attention + cross-attention)

See [Visual_Storyteller_Plan.md](Visual_Storyteller_Plan.md) for the full plan.

## Repo layout

```
src/
  config.py     shared contract: paths, dims, hyperparams, special-token indices
  data.py       caption cleaning, Vocabulary, splits, CaptionDataset, collate   (P2)
  train.py      training loop, masked CE + label smoothing, AMP, checkpoint/resume (P2)
  decode.py     greedy + beam search, generate_caption() entry point             (P2)
  evaluate.py   BLEU-1..4 over all 5 references (NLTK)                            (P2)
  encoder.py            frozen ResNet-50 -> 49x2048 regions                      (P1)
  decoder_lstm.py       LSTM + Bahdanau attention                                (P1)
  decoder_transformer.py  from-scratch Transformer decoder                       (P1)
  model.py              EncoderDecoder wrapper (shared forward contract)         (P1)
  viz.py                attention heatmaps                                        (P1)
scripts/
  smoke_test.py            end-to-end Phase-1 test with a stub model (no GPU needed)
  precompute_features.py   cache ResNet features -> features.h5                   (P1)
notebooks/
  data_and_training.ipynb  data + model + training (loss curves) + save artifacts
  inference.ipynb          generate_caption, demo on unseen images, success/failure
reports/
  results.md               metric tables + analysis
```

## Setup

```bash
pip install -r requirements.txt
```

NLTK BLEU needs no extra downloads. On a fresh machine, verify the pipeline:

```bash
python -m scripts.smoke_test
```

This runs data -> train (with checkpoint/resume) -> decode -> BLEU against a tiny
synthetic dataset and a stub model. If it prints `SMOKE TEST PASSED`, our half is
wired correctly and ready for P1's real model to drop in.

## Data

Place Flickr8k under `data/`:

```
data/Images/*.jpg
data/captions.txt          # CSV (image,caption) or the original token format
```

`src/config.py` has `use_local_paths()` (default) and `use_colab_paths()` for the
Drive layout used during training.

## Status

- [x] `config.py`, `data.py`, `train.py`, `decode.py`, `evaluate.py` (P2)
- [x] End-to-end smoke test passing
- [ ] P1: encoder + decoders + model wrapper + feature precompute
- [ ] Notebooks, full training, metric comparison
