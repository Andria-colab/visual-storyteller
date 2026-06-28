# CLAUDE.md

Guidance for Claude Code when working in this repo.

## Project

Image captioning on **Flickr8k**. A shared frozen **ResNet-50** encoder feeds two
decoders compared head to head: an **LSTM + Bahdanau attention** baseline and a
**from-scratch Transformer** decoder. Full plan in
[Visual_Storyteller_Plan.md](Visual_Storyteller_Plan.md).

Two-person project. **We are Person 2 — Data, Training & Evaluation lead.**
Person 1 (Andria) owns the network internals (encoder, both decoders, model
wrapper, attention viz). Do not implement P1's files unless explicitly asked.

## Ownership split

| Ours (P2) | Theirs (P1) |
|---|---|
| `config.py` (shared, we drafted it) | `encoder.py` |
| `data.py` | `decoder_lstm.py` |
| `train.py` | `decoder_transformer.py` |
| `decode.py` (incl. `generate_caption`) | `model.py` (EncoderDecoder wrapper) |
| `evaluate.py` | `viz.py` |
| `inference.ipynb` | `scripts/precompute_features.py` |
| training/eval sections of `data_and_training.ipynb` | model section of `data_and_training.ipynb` |

## The frozen contract (do not change without agreeing with P1)

- Special-token indices: `<pad>`=0, `<start>`=1, `<end>`=2, `<unk>`=3 (see `config.py`).
  `<pad>`=0 doubles as `padding_idx` and the CE `ignore_index`.
- Features per image: `[num_regions, feature_dim]` = `[49, 2048]`.
- `model.forward(features, captions)` returns logits `[B, T-1, V]`, aligned to
  predict `captions[:, 1:]` (teacher forcing).
- Inference contract used by `decode.py`:
  `model.encode_image(features) -> memory` and
  `model.decode_step(memory, tokens[B,t]) -> logits[B, V]`.
- Graded entry point: `generate_caption(image_path: str, model: any) -> str`.
  `model` carries `.vocab`, `.device`, `.encoder`, `.image_transform`, optional
  `.beam_size` / `.max_len`.
- `features.h5` is keyed by image id (the `.jpg` filename); our `CaptionDataset`
  reads it lazily.

## Commands

```bash
pip install -r requirements.txt
python -m scripts.smoke_test     # end-to-end check, no GPU/data needed
```

The smoke test exercises data -> train (with checkpoint/resume) -> decode -> BLEU
using a stub model that honours the contract above. Run it after touching any
P2 module; if it prints `SMOKE TEST PASSED [OK]`, our half is intact.

## Conventions

- Keep logic in `src/`; notebooks stay thin and import from it. Clear notebook
  outputs before committing (avoids merge conflicts — see `.gitignore`).
- BLEU uses NLTK only (METEOR/CIDEr deliberately out of scope).
- Checkpoints are written atomically (write `.tmp` then `os.replace`) so a Colab
  crash mid-save never corrupts the good checkpoint. Preserve this.
- Paths come from `config.py`: `use_local_paths()` for dev, `use_colab_paths()`
  on Colab. Never hardcode paths in modules or notebooks.
- Vocab is built from **train ids only** (`build_vocab(..., image_ids=train)`) and
  the **test split stays untouched until final analysis**. Do not leak.

## Status

**Code-complete (28 June 2026).** Both halves are implemented and verified end to
end; `python -m scripts.smoke_test` passes. The only thing left is the user
running the full training on Flickr8k (Colab/GPU) and curating final example ids.

See [STATUS.md](STATUS.md) for the human-readable breakdown.

- **Done (P2):** `config.py`, `data.py` (incl. `ImageFeatureDataset` for val BLEU),
  `train.py` (real BLEU-4 early stopping), `decode.py`, `evaluate.py`,
  `notebooks/inference.ipynb` (incl. attention-viz section), smoke test passing,
  repo scaffolding (`requirements.txt`, `.gitignore`, README).
- **Done (P1, implemented in this repo):** `encoder.py` (frozen ResNet-50 +
  `build_image_transform`), `decoder_lstm.py` (LSTM + Bahdanau), `decoder_transformer.py`
  (from-scratch Transformer via custom layers over `nn.MultiheadAttention`),
  `model.py` (`EncoderDecoder`, drop-in for the stub), `viz.py` (attention heatmaps),
  `scripts/precompute_features.py` (`features.h5`), `notebooks/data_and_training.ipynb`.
- **Interface freeze (resolved):** model exposes `decode_step` (stateless, recompute
  over prefix); `features.h5` keyed by `.jpg` filename; `EncoderDecoder` holds
  `.encoder`, notebook attaches `.vocab/.device/.image_transform/.beam_size/.max_len`.
- **Remaining (user):** run `data_and_training.ipynb` on real data to produce
  `features.h5` + checkpoints, then curate success/failure ids in `inference.ipynb`.
