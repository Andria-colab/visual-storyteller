# Experiment log

Track each training/eval run and the config that produced it, so the report can
compare choices and we don't lose numbers between Colab sessions.

## How to log a run
1. Set the knobs in `src/config.py` (and the eval transform in `src/encoder.py`).
2. Run `notebooks/data_and_training.ipynb` → note the **best val BLEU-4** (printed
   per epoch; the best epoch is what gets saved to `*_best.pt`).
3. Run `notebooks/inference.ipynb` → it writes the full BLEU-1..4 **test** table to
   `reports/results.md`; copy the headline **test BLEU-4** into the table below.
4. Add a row. Keep any knob that differs from the defaults explicit.

## Current default config
| knob | value |
|---|---|
| variants | `lstm` (baseline), `transformer` (main) |
| `min_word_freq` | 3 |
| `max_caption_len` | 22 |
| Transformer `dropout` | 0.25 |
| Transformer | 4 layers, 8 heads, d=512, ffn=2048, weight-tied |
| LSTM | hidden 512, dropout 0.5, Bahdanau attention |
| image transform | resize → 224×224 (squash, no crop) |
| optimizer | AdamW lr 1e-4, wd 1e-4, label smoothing 0.1, grad clip 5.0 |
| Transformer LR | warmup 4000 + inverse-sqrt |
| decoding | beam size 3 (greedy used for the per-epoch val signal) |

## Results
| date | variant | min_freq | dropout | transform | decode | best epoch | val BLEU-4 | test BLEU-4 | notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-06-28 | lstm | 3 | 0.5 | square | beam | 14 | 0.1908 | TBD | baseline; early-stopped ep 19 |
| 2026-06-29 | transformer | 3 | 0.25 | square | beam | 11 | 0.2024 | TBD | mild overfit ep 12-16; best ckpt saved |
|  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |

_Realistic Flickr8k target: BLEU-4 ≈ 0.18–0.24._

## Planned ablations (EDA-driven)
- [ ] **transform** — square (224×224) vs `Resize(256)+CenterCrop(224)`: does keeping
      edge content help? (must rebuild `features.h5` between the two)
- [ ] **Transformer dropout** — 0.25 vs 0.1: overfitting check (watch the train/val gap)
- [ ] **min_word_freq** — 3 vs 5 vs 2: vocab coverage vs learning rare words
- [ ] **Transformer capacity** — 4L/d=512 vs 3L/d=384: only if the train/val gap shows
      the Transformer overfitting
- [ ] **decoding** — greedy vs beam (size 3): already produced side-by-side by
      `inference.ipynb`
