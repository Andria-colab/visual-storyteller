"""
Attention visualisation (Person 1).

Renders what each decoder "looks at" while generating a caption — the classic
Show-Attend-and-Tell figure for the LSTM baseline, and cross-attention maps for
the Transformer. Used only in the analysis notebook (inference.ipynb); nothing in
the training/decoding path imports this, so it is free to drive its own greedy
decode loop and read the attention maps the decoders stash after each step:

  - LSTMDecoder._last_alpha        : [B, 49] Bahdanau weights
  - TransformerDecoder._last_cross_attn : [B, L, 49] last-layer cross-attention

Both functions take an already-built `model` (with `.encoder`, `.image_transform`,
`.vocab`, `.device` attached, exactly as inference.ipynb builds it).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from .config import START_IDX, END_IDX

# Plain resize/crop (no normalisation) so we can show the same 224x224 frame the
# 7x7 attention grid aligns to.
_DISPLAY_TF = T.Compose([T.Resize(256), T.CenterCrop(224)])


@torch.no_grad()
def _decode_with_attention(model, image_path: str):
    """Greedy-decode one image, collecting (word, attn_map[49]) per step.

    Returns (display_image PIL, list[(word, attn[49] tensor)]).
    """
    model.eval()
    device = getattr(model, "device", "cpu")
    vocab = getattr(model, "vocab")
    transform = getattr(model, "image_transform")
    max_len = getattr(model, "max_len", 22)
    variant = getattr(model, "variant", "lstm")

    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)             # [1, 3, H, W]
    feats = model.encoder(x)                               # [1, 49, 2048]
    memory = model.encode_image(feats)

    tokens = torch.full((1, 1), START_IDX, dtype=torch.long, device=device)
    steps = []
    for _ in range(max_len):
        logits = model.decode_step(memory, tokens)         # [1, V]
        nxt = int(logits.argmax(dim=-1).item())
        if nxt == END_IDX:
            break
        if variant == "transformer":
            attn = model.decoder._last_cross_attn[0, -1]   # [49] (last query pos)
        else:
            attn = model.decoder._last_alpha[0]            # [49]
        word = vocab.itos.get(nxt, "<unk>")
        steps.append((word, attn.detach().float().cpu()))
        tokens = torch.cat([tokens, torch.tensor([[nxt]], device=device)], dim=1)

    return _DISPLAY_TF(img), steps


def _overlay(ax, image, attn_49, title: str, grid: int = 7, alpha: float = 0.5):
    """Show `image` with the [49]-vector attention map upsampled and overlaid."""
    side = int(round(image.size[0]))                       # 224
    heat = attn_49.reshape(1, 1, grid, grid)
    heat = F.interpolate(heat, size=(side, side), mode="bilinear", align_corners=False)
    heat = heat.squeeze().numpy()
    ax.imshow(image)
    ax.imshow(heat, cmap="jet", alpha=alpha, extent=(0, side, side, 0))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def visualize_attention(model, image_path: str, max_cols: int = 5):
    """Plot the generated caption with one attention heatmap per word.

    Works for both variants (LSTM Bahdanau / Transformer cross-attention).
    Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    image, steps = _decode_with_attention(model, image_path)
    if not steps:
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.imshow(image)
        ax.set_title("(no caption generated)")
        ax.axis("off")
        return fig

    n = len(steps) + 1                                     # +1 for the raw image
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.6 * rows))
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    axes[0].imshow(image)
    axes[0].set_title(" ".join(w for w, _ in steps), fontsize=9)
    axes[0].axis("off")
    for i, (word, attn) in enumerate(steps, start=1):
        _overlay(axes[i], image, attn, word)
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    return fig


# Convenience aliases — the notebook can call either name.
def visualize_lstm_attention(model, image_path: str):
    return visualize_attention(model, image_path)


def visualize_transformer_attention(model, image_path: str):
    return visualize_attention(model, image_path)
