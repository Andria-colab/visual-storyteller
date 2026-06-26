# The Visual Storyteller
### Deep Learning — Final Project Plan

**Team size:** 2 &nbsp;·&nbsp; **Due:** 3 July, 23:59 &nbsp;·&nbsp; **Working window:** 17 days
**Stack:** PyTorch · Google Colab (GPU) · Google Drive (checkpoints) · GitHub
**Submission:** GitHub repository (mandatory) — two notebooks + README + run instructions, with a *distributed* commit history.

---

## 1. What we are building

A system that takes one image and generates a sentence describing it — bridging two modalities: a vision model reads the pixels (encoder) and a sequence model writes the words (decoder).

The brief requires **one** working captioning model plus the two notebooks. To put the project comfortably above that bar, we ship **two models and compare them**:

- **Baseline — "Show, Attend and Tell":** a pretrained CNN encoder + an LSTM decoder with Bahdanau (additive) attention.
- **Main — Transformer captioner:** the same CNN encoder feeding a Transformer decoder with masked self-attention and cross-attention over the image regions.

The head-to-head comparison (metrics + qualitative analysis + attention maps) is what lifts a working project into a top one — and the baseline doubles as a safety net: even if the Transformer underperforms on a small dataset, we always have a complete, submittable result.

## 2. Locked decisions

| Decision | Choice | Why |
|---|---|---|
| **Models** | Baseline (CNN + LSTM + attention) **and** main (CNN + Transformer decoder), compared head to head | The comparison maximizes the grade; the baseline is the guaranteed fallback |
| **Data** | Flickr8k (provided), with CNN features **precomputed and cached** once | The single biggest Colab time-saver — training never re-runs the CNN forward pass |
| **Encoder** | Frozen pretrained **ResNet-50**, spatial map → 49 region vectors (7×7×2048) | Transfer learning on a small dataset; regions are what attention attends to |
| **Code structure** | A small `src/` package with **two thin notebooks** on top | Parallelizable, reviewable diffs; satisfies the "two notebooks" requirement while showing real, distributed contribution |

## 3. Requirements coverage — every mark accounted for

| # | Requirement (from the brief) | Owner | How it is satisfied |
|---|---|---|---|
| 1 | Work with the provided Flickr8k data (8,000 images, 5 captions each, raw images + text) | Both (P2 leads) | Cleaning, vocab, splits, feature caching in `data.py` |
| 2 | A model that takes an image and generates a sentence | Both | Shared encoder + decoder; two variants |
| 3 | **Notebook 1** `data_and_training.ipynb` — data loading | P2 | Ingest images + text, show samples, build vocab, splits |
| 4 | Notebook 1 — model definition | P1 | Encoder + both decoders, imported from `src/` |
| 5 | Notebook 1 — training showing loss progression | P2 | Printed + plotted loss / metric curves for both models |
| 6 | Notebook 1 — save trained model artifacts | P2 | `baseline.pt`, `transformer.pt`, `vocab.pkl` to Drive |
| 7 | **Notebook 2** `inference.ipynb` — exact `generate_caption(image_path: str, model: any) -> str` | P2 | The public entry point used in the demo |
| 8 | Notebook 2 — demonstration on unseen test images | P2 | Held-out test set, untouched during development |
| 9 | Notebook 2 — analysis: success **and** failure cases | P1 + P2 | Strong examples + diagnosed failures + attention maps |
| 10 | GitHub repo + README + run instructions | Both | Set up in Phase 0 |
| 11 | Distributed commit history (everyone contributes) | Both | Feature-branch → PR → review workflow (§7) |
| 12 | Delivered by 3 July 23:59 | Both | Schedule targets 2 July (§6) |

Everything the brief *mandates* is one working model and these two notebooks. The **second model, the metric comparison, and the attention visualizations are deliberate upside** layered on top — which is what takes this from "complete" to "top of the class."

## 4. Architecture

```
Image
  │
  ▼
[Encoder]  frozen ResNet-50 -> 49 region vectors (2048-d), linear -> d=512   (shared)
  │
  ├──────────────► [Baseline decoder]  LSTM + Bahdanau attention over regions
  │
  └──────────────► [Main decoder]      Transformer: masked self-attention
                                        + cross-attention over regions + FFN
  │
  ▼
[Inference]  greedy + beam search (beam 3-5) -> generate_caption() -> caption
```

- **Encoder (shared).** Drop the CNN classifier head; take `layer4`'s 7×7×2048 map = 49 regions; freeze and precompute; a linear layer projects 2048 → 512.
- **Baseline decoder.** Embedding → LSTM; additive attention builds a context over the 49 regions each step; dropout 0.5; teacher forcing. The attention weights give interpretable "where it looked" heatmaps.
- **Main decoder.** Token embeddings + positional encoding → N = 4 layers (tune 3–6), h = 8 heads, d = 512, FFN 2048; each layer = masked (causal) self-attention → cross-attention over the region features → feed-forward, with residuals + LayerNorm.
- **Training (both).** Cross-entropy with a padding mask + label smoothing 0.1; AdamW; mixed precision; batch 32–64; gradient clipping 5.0; 15–30 epochs with early stopping on val CIDEr / BLEU-4; checkpoint to Drive every epoch + resume.

## 5. Task distribution (2 people)

Days 1–3 we **pair** on the foundation (repo, environment, data, frozen interfaces) so both of us understand the whole system. Then we split into two balanced verticals and **review each other's pull requests**, which keeps both names in the commit graph throughout.

### Person 1 — Andria — Modeling lead &nbsp; ⭐ HARDEST

Owns the network architecture and the part that earns the headline result.

- **Encoder** (`encoder.py`) + `scripts/precompute_features.py` — extract and cache the 49×2048 features to `features.h5`.
- **Baseline decoder** (`decoder_lstm.py`) — LSTM + Bahdanau attention.
- **Main decoder** (`decoder_transformer.py`) — the from-scratch Transformer decoder: masked self-attention, cross-attention over the image regions, positional encoding, the causal mask. This is the hardest, most bug-prone code in the project.
- **Model wrapper** (`model.py`) — one `EncoderDecoder` exposing both variants behind a shared `forward`.
- **Attention visualization** (`viz.py`) — LSTM attention heatmaps + Transformer cross-attention maps.
- Leads the **model + training sections** of `data_and_training.ipynb`.
- Reviews P2's data + evaluation PRs.

### Person 2 — Data, Training & Evaluation lead

Owns the pipeline, the training machinery, and measurement.

- **Data** (`data.py`) — caption cleaning, vocab (freq ≥ 5, `<unk>` / `<pad>`, save `vocab.pkl`), `CaptionDataset`, collate / padding, the 6k / 1k / 1k splits (test untouched until final analysis).
- **Trainer** (`train.py`) — the loop, loss with mask + label smoothing, AMP, LR schedule (warmup + inverse-sqrt for the Transformer), checkpoint-to-Drive, resume.
- **Decoding** (`decode.py`) — greedy + beam search (beam 3–5), and the `generate_caption` entry point.
- **Evaluation** (`evaluate.py`) — BLEU-1..4, METEOR, CIDEr over all 5 references; the score tables.
- Runs the **full training** of both models; tunes LR / batch; confirms convergence and checkpoint / resume.
- Leads `inference.ipynb` — `generate_caption`, the demo on unseen images, success + failure analysis.
- Reviews P1's model PRs.

### Shared (both)

`README.md`, `config.py` (paths + hyperparameters + frozen interfaces), the integration and training runs, `reports/results.md`, and the final write-up. The owner of a file writes it; the partner reviews and merges. Swap a small task or two mid-project so each person has touched both halves.

## 6. Why Modeling is the hardest task

- The **Transformer decoder from scratch** — masked self-attention, cross-attention over image regions, positional encoding, the causal mask — is the most conceptually demanding and most bug-prone code here; subtle masking or shape errors fail *silently* and quietly wreck the captions.
- It is the **"main model"** whose result is the headline comparison: whether the Transformer beats the baseline is decided in this code.
- It owns **two decoders plus the shared encoder**, so it integrates the most moving parts.
- The **attention visualizations** are a second, separate piece of non-trivial work (extracting and overlaying attention weights correctly per word).
- The data and training side is demanding too, and carries the real *operational* risk (convergence, Colab disconnects) — but it follows well-trodden patterns. The from-scratch Transformer is the genuinely open, high-skill part, which is why it's yours.

## 7. Interfaces (freeze on Day 2)

Agree all of these before splitting, so neither person blocks the other:

- Paths (data, features, checkpoints), `d = 512`, vocab params (min freq, max length ~22 tokens), batch size — all in `config.py`.
- The **`vocab` object** (stoi / itos) contract.
- The **`model.forward`** signature and the tensor shapes it expects (features `[B, 49, 2048]`, captions `[B, T]`).
- The **`generate_caption(image_path: str, model: any) -> str`** signature — the public entry point used in `inference.ipynb`.

## 8. Timeline (17 days → 3 July)

| Phase | Dates | What happens | Milestone |
|---|---|---|---|
| **0 — Setup & alignment** | 17–18 Jun | Both: repo + collaborators + `.gitignore` + `requirements.txt`; Colab + Drive; unzip data; eyeball images + captions; freeze interfaces in `config.py`; open issues + a board | Repo cloneable, Colab opens it, roles & interfaces agreed |
| **1 — Data + encoder + skeletons** | 19–22 Jun | P2: `data.py` + vocab + Dataset. P1: encoder + feature caching, decoder stub. P2: Trainer skeleton (loss, AMP, checkpoint / resume) | One tiny batch flows features → stub decoder → loss with no errors |
| **2 — Baseline trains end-to-end** | 23–26 Jun | P1: finish LSTM + attention + `model.py`. P2: full baseline training, tune, confirm checkpoints. P1: start the Transformer. P2: greedy decode + first BLEU | **Midpoint:** baseline produces real captions on val; BLEU measured |
| **3 — Transformer + full evaluation** | 27–30 Jun | Both: integrate + train the Transformer (warmup); compare curves. P2: full metrics + beam search. P1: attention heatmaps | Both models trained; baseline-vs-Transformer table exists |
| **4 — Notebooks, analysis, README** | 1–2 Jul | P1: finalize `data_and_training.ipynb`. P2: finalize `inference.ipynb` (demo + success / failure). Both: README + `results.md` | Both notebooks run top-to-bottom from a clean runtime |
| **5 — Freeze, QA, submit** | 3 Jul | Both: "Restart & Run All" on a fresh Colab; verify artifacts load and captions reproduce; proofread; tag a release; **submit well before 23:59** | Submitted |

**Built-in buffer:** the baseline is fully working by Day 10 (26 Jun), so the last week is upside (Transformer, beam search, analysis), not core risk. If we slip, we still submit a complete project.

## 9. Evaluation & analysis

- **Quantitative.** BLEU-1..4, METEOR, CIDEr (optionally ROUGE-L) on the held-out test set against all 5 references (via `pycocoevalcap`, or NLTK for BLEU). Report greedy vs beam and baseline vs Transformer in one table in `reports/results.md` and the README.
- **Qualitative.** ~8 strong examples and ~5 diagnosed failures (object hallucination, miscounting, colour / scene errors), plus attention heatmaps showing where each model looks per word. This analysis is high-value for the grade.
- **Ballpark targets (Flickr8k).** BLEU-4 ≈ 0.18–0.24, CIDEr ≈ 0.5–0.7 — realistic student targets, not guarantees; report whatever we measure honestly.

> **Stretch (only once the core is done):** fine-tune the top CNN block, CIDEr-oriented beam tuning, or a ViT / CLIP encoder. A web demo (Gradio / Streamlit) is **out of scope** — the `inference.ipynb` notebook *is* the demonstration the brief asks for.

## 10. Risks & mitigations

- **Colab disconnects / GPU limits** → precompute and cache features; checkpoint to Drive every epoch + resume; short epochs; keep the best model.
- **Overfitting (8k images is small)** → frozen pretrained encoder, dropout, weight decay, label smoothing, early stopping, light augmentation.
- **Transformer underperforms on small data** → smaller model (N = 3, d = 256), stronger regularization, warmup tuning — and the baseline is the guaranteed fallback.
- **Notebook merge conflicts** → keep logic in `src/`, thin notebooks, clear outputs before committing.
- **Time slip** → baseline done by Day 10; everything after is layered, independently submittable upside.
- **Last-minute submission failure** → hard internal freeze midday 3 Jul; submit hours before 23:59.

## 11. Deliverables & submission checklist

- [ ] `data_and_training.ipynb` — data loading, model definition, training with visible loss progression, artifact saving
- [ ] `inference.ipynb` — exact `generate_caption(image_path, model) -> str`, demo on unseen images, success + failure analysis
- [ ] Model artifacts (`baseline.pt`, `transformer.pt`, `vocab.pkl`) saved and loadable from a clean runtime
- [ ] `README.md` — overview, results table, exact run instructions
- [ ] Both notebooks pass "Restart & Run All" on a fresh Colab
- [ ] Repo: `requirements.txt`, `.gitignore`, tidy `src/`
- [ ] GitHub: both members contributing across the timeline (not one dump)
- [ ] Submitted before **23:59 on 3 July 2026**
