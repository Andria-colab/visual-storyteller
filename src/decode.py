"""
Decoding & inference (Person 2).

Turns a trained model + image into a sentence:
  - greedy_decode   : argmax at each step (fast, used during validation)
  - beam_search     : beam 3-5 (better captions, used for final eval / demo)
  - generate_caption: the GRADED public entry point (exact signature below)

These functions depend only on the shared model contract, not on which decoder
variant is inside `model`:
  - model.encode_image(features[B,49,2048]) -> memory  (P1 provides; or model
    accepts raw features in step())
  - model.decode_step(memory, tokens[B,t]) -> logits[B, vocab]  for the next token

NOTE: the exact step API is part of the Day-2 interface freeze. The two helpers
below are written against `model.decode_step`; if P1's wrapper exposes a single
`forward` instead, we adapt here in one place so the notebooks never change.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from .config import START_IDX, END_IDX
from .data import Vocabulary


@torch.no_grad()
def greedy_decode(model, features: torch.Tensor, vocab: Vocabulary, max_len: int = 22, device: str = "cpu") -> list[list[int]]:
    """Greedy decode a batch of features -> list of token-id lists (per sample)."""
    model.eval()
    features = features.to(device)
    b = features.size(0)
    memory = model.encode_image(features)
    tokens = torch.full((b, 1), START_IDX, dtype=torch.long, device=device)
    finished = torch.zeros(b, dtype=torch.bool, device=device)

    for _ in range(max_len):
        logits = model.decode_step(memory, tokens)        # [B, vocab]
        nxt = logits.argmax(dim=-1, keepdim=True)          # [B, 1]
        tokens = torch.cat([tokens, nxt], dim=1)
        finished |= nxt.squeeze(1) == END_IDX
        if bool(finished.all()):
            break

    return [seq[1:].tolist() for seq in tokens]            # drop <start>


@torch.no_grad()
def beam_search(model, feature: torch.Tensor, vocab: Vocabulary, beam_size: int = 3, max_len: int = 22, device: str = "cpu") -> list[int]:
    """Beam search for a SINGLE image. `feature` is [49, 2048] or [1,49,2048].

    Returns the best token-id sequence (without <start>, truncated at <end>).
    Length-normalized score to avoid favouring short captions.
    """
    model.eval()
    if feature.dim() == 2:
        feature = feature.unsqueeze(0)
    feature = feature.to(device)
    memory = model.encode_image(feature)                   # [1, ...]

    # Each beam: (tokens tensor [1, t], cumulative logprob).
    beams = [(torch.full((1, 1), START_IDX, dtype=torch.long, device=device), 0.0)]
    completed: list[tuple[list[int], float]] = []

    for _ in range(max_len):
        candidates = []
        for tokens, score in beams:
            if tokens[0, -1].item() == END_IDX:
                completed.append((tokens[0].tolist(), score))
                continue
            logits = model.decode_step(memory, tokens)     # [1, vocab]
            logp = torch.log_softmax(logits[0], dim=-1)
            topv, topi = logp.topk(beam_size)
            for v, i in zip(topv.tolist(), topi.tolist()):
                new_tokens = torch.cat([tokens, torch.tensor([[i]], device=device)], dim=1)
                candidates.append((new_tokens, score + v))

        if not candidates:
            break
        # Keep top beams by length-normalized score.
        candidates.sort(key=lambda x: x[1] / (x[0].size(1) - 1), reverse=True)
        beams = candidates[:beam_size]

    for tokens, score in beams:
        completed.append((tokens[0].tolist(), score))

    completed.sort(key=lambda x: x[1] / max(len(x[0]) - 1, 1), reverse=True)
    best = completed[0][0][1:]                              # drop <start>
    if END_IDX in best:
        best = best[: best.index(END_IDX)]
    return best


# --------------------------------------------------------------------------- #
# GRADED public entry point — exact signature from the brief. Do not change.
# --------------------------------------------------------------------------- #
def generate_caption(image_path: str, model: Any) -> str:
    """Generate a caption for the image at `image_path` using `model`.

    This is the function the brief grades and the demo in inference.ipynb calls.
    It must be fully self-contained: take a raw image path, return a string.

    Implementation plan (wired up once P1's encoder/model wrapper lands):
      1. load the image (PIL), apply the encoder's eval transform
      2. run the frozen ResNet-50 encoder -> [49, 2048] features
      3. beam_search (or greedy) to token ids
      4. vocab.decode -> join into a sentence

    `model` is expected to carry what it needs (its vocab, device, and encoder)
    so the signature stays exactly (str, any) -> str. We attach those as
    attributes when we build the model in inference.ipynb:
        model.vocab, model.device, model.image_transform, model.encoder
    """
    vocab: Vocabulary = getattr(model, "vocab")
    device: str = getattr(model, "device", "cpu")
    max_len: int = getattr(model, "max_len", 22)
    beam_size: int = getattr(model, "beam_size", 3)

    features = _image_to_features(image_path, model, device)   # [49, 2048]
    ids = beam_search(model, features, vocab, beam_size=beam_size, max_len=max_len, device=device)
    return " ".join(vocab.decode(ids))


def _image_to_features(image_path: str, model: Any, device: str) -> torch.Tensor:
    """Load an image and run it through the (frozen) encoder -> [49, 2048].

    Relies on P1's encoder + transform being attached to `model`. Kept separate
    so the feature-extraction detail lives in one place.
    """
    from PIL import Image

    transform = getattr(model, "image_transform")
    encoder = getattr(model, "encoder")
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)             # [1, 3, H, W]
    with torch.no_grad():
        feats = encoder(x)                                 # [1, 49, 2048]
    return feats.squeeze(0)
