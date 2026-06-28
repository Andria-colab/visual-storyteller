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
  encoder.py            frozen ResNet-50 -> 49x2048 regions + image transform    (P1)
  decoder_lstm.py       LSTM + Bahdanau attention                                (P1)
  decoder_transformer.py  from-scratch Transformer decoder                       (P1)
  model.py              EncoderDecoder wrapper (shared forward contract)          (P1)
  viz.py                attention heatmaps                                        (P1)
scripts/
  smoke_test.py            end-to-end Phase-1 test with a stub model (no GPU needed)
  precompute_features.py   cache ResNet features -> features.h5                   (P1)
notebooks/
  eda.ipynb                exploratory data analysis (caption / vocab / image stats)
  data_and_training.ipynb  data + model + training (loss curves) + save artifacts
  inference.ipynb          generate_caption, demo on unseen images, success/failure
reports/
  results.md               metric tables + analysis
```

## Install

```bash
pip install -r requirements.txt
```

NLTK BLEU needs no extra downloads. On a fresh machine, verify the pipeline with
a tiny synthetic dataset + stub model (no GPU or data needed):

```bash
python -m scripts.smoke_test          # expect: SMOKE TEST PASSED [OK]
```

## Data setup

The dataset is **Flickr8k** (8,000 images, 5 captions each — the course
`caption_data.zip`). Unzip it into `data/` so the layout is:

```
data/
  Images/*.jpg               # ~8k JPEGs, e.g. 1000268201_693b08cb0e.jpg
  captions.txt               # CSV (image,caption) OR the original token format
  Flickr_8k.trainImages.txt  # optional — official splits (6k/1k/1k)
  Flickr_8k.devImages.txt    # optional
  Flickr_8k.testImages.txt   # optional
```

- `captions.txt` may be either the Kaggle CSV (`image,caption` header) or the
  original Flickr8k token format (`<img>.jpg#0\tA child ...`) — `data.py` detects both.
- If the three `Flickr_8k.*Images.txt` split files are present they are used;
  otherwise a deterministic 6k/1k/1k random split (seed 42) is made. The **test
  split is held out** until `inference.ipynb`.

Paths come from `src/config.py`: `use_local_paths()` (the default) for the layout
above, or `use_colab_paths()` for the Google Drive layout used when training on
Colab (images under `/content/flickr8k`, artifacts under
`/content/drive/MyDrive/visual_storyteller`).

## How to run (end to end)

```bash
# 0. (optional) explore the dataset
#    run notebooks/eda.ipynb                (caption / vocab / image statistics)

# 1. cache the frozen ResNet-50 features once (GPU recommended; resumable)
python -m scripts.precompute_features            # add --colab on Colab

# 2. build vocab + train both models + save artifacts
#    run notebooks/data_and_training.ipynb  (Restart & Run All)

# 3. demo + success/failure analysis + BLEU comparison table
#    run notebooks/inference.ipynb          (Restart & Run All)
```

Training writes checkpoints atomically every epoch to `cfg.checkpoint_dir` and
keeps the best of each model by validation BLEU-4 at `baseline_best.pt` /
`transformer_best.pt`. `inference.ipynb` loads those plus `vocab.pkl`, runs the
graded `generate_caption(image_path, model) -> str`, visualises attention, and
writes the BLEU table to `reports/results.md`.

## Status

- [x] Data / training / decoding / evaluation pipeline (P2)
- [x] Encoder + LSTM & Transformer decoders + model wrapper + viz (P1)
- [x] Feature precompute (`features.h5`)
- [x] Both notebooks authored; end-to-end smoke test passing
- [ ] Full training run on Flickr8k + final metric comparison (run the notebooks)
