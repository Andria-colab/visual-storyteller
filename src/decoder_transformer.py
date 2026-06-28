"""
Main decoder (Person 1): a from-scratch Transformer decoder.

Built from custom decoder layers that use `nn.MultiheadAttention` as the
attention primitive (the layer assembly, masking, positional encoding and
cross-attention wiring are all our own). Each layer does, pre-norm style:

    x = x + SelfAttn(LN(x))        # causal-masked self-attention over tokens
    x = x + CrossAttn(LN(x), mem)  # attention over the 49 image regions
    x = x + FFN(LN(x))             # 512 -> 2048 -> 512

Pre-norm is used deliberately: it is markedly more stable than post-norm with
the warmup + inverse-sqrt ("Noam") schedule already in train.py, especially on a
small dataset like Flickr8k.

Shapes / contract (see config.py and model.py):
  - region features come in RAW as [B, 49, 2048]; `project_features` maps them to
    [B, 49, 512] = the opaque `memory` the wrapper passes to `decode_tokens`.
  - forward(features, captions) -> logits [B, T-1, V], teacher-forced.
  - decode_tokens(memory, tokens[B, t]) -> logits [B, V], STATELESS (recompute
    over the prefix each call), matching the decode loop in decode.py.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import Config, PAD_IDX


class SinusoidalPositionalEncoding(nn.Module):
    """Standard fixed sin/cos positional encoding, added to token embeddings."""

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # Not a parameter and kept out of the checkpoint (persistent=False).
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, L, D] -> x + positional encoding for the first L positions."""
        return x + self.pe[: x.size(1)].unsqueeze(0)


class TransformerDecoderLayer(nn.Module):
    """One pre-norm decoder layer: masked self-attn -> cross-attn -> FFN."""

    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.embed_dim
        self.self_attn = nn.MultiheadAttention(d, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, cfg.num_heads, dropout=cfg.dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d, cfg.ffn_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_dim, d),
        )
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)
        self.drop1 = nn.Dropout(cfg.dropout)
        self.drop2 = nn.Dropout(cfg.dropout)
        self.drop3 = nn.Dropout(cfg.dropout)

    def forward(self, x, memory, attn_mask, key_padding_mask):
        """Returns (x, cross_attn_weights [B, L_tgt, 49])."""
        h = self.norm1(x)
        sa, _ = self.self_attn(
            h, h, h, attn_mask=attn_mask,
            key_padding_mask=key_padding_mask, need_weights=False,
        )
        x = x + self.drop1(sa)

        h = self.norm2(x)
        ca, attn_w = self.cross_attn(
            h, memory, memory, need_weights=True, average_attn_weights=True,
        )
        x = x + self.drop2(ca)

        h = self.norm3(x)
        x = x + self.drop3(self.ffn(h))
        return x, attn_w


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, cfg: Config, tie_weights: bool = True):
        super().__init__()
        self.cfg = cfg
        d = cfg.embed_dim

        # 2048 -> 512 region projection (memory side) lives here.
        self.mem_proj = nn.Linear(cfg.feature_dim, d)
        self.embed = nn.Embedding(vocab_size, d, padding_idx=0)
        self.scale = math.sqrt(d)
        self.pos = SinusoidalPositionalEncoding(d, max_len=64)
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([TransformerDecoderLayer(cfg) for _ in range(cfg.num_layers)])
        self.norm = nn.LayerNorm(d)
        self.fc = nn.Linear(d, vocab_size)
        if tie_weights:
            # Tie output projection to the embedding matrix (fewer params, helps
            # generalisation on a small dataset). Shapes match: both [V, d].
            self.fc.weight = self.embed.weight

        # Most recent last-layer cross-attention [B, L, 49], stashed for viz.
        self._last_cross_attn: torch.Tensor | None = None
        self._reset_parameters()

    def _reset_parameters(self):
        """Init for stable starting loss (~log V).

        The embedding is scaled by sqrt(d) on input and (when tied) reused as the
        output projection, so it must start at std = 1/sqrt(d): that makes the
        scaled input ~unit-variance AND keeps the tied output logits ~unit-scale.
        Leaving the default N(0,1) makes the initial loss explode.
        """
        d = self.cfg.embed_dim
        nn.init.normal_(self.embed.weight, mean=0.0, std=d ** -0.5)
        with torch.no_grad():
            self.embed.weight[0].zero_()             # keep <pad> row at zero
        nn.init.xavier_uniform_(self.mem_proj.weight)
        nn.init.zeros_(self.mem_proj.bias)
        for layer in self.layers:
            for module in layer.ffn:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)

    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        """[B, 49, 2048] -> [B, 49, 512]  (the opaque `memory`)."""
        return self.mem_proj(features)

    def _causal_mask(self, length: int, device) -> torch.Tensor:
        """Boolean mask [L, L]: True above the diagonal = not allowed to attend.

        Boolean (not float) so it matches the boolean key-padding mask — mixing a
        float attn_mask with a bool key_padding_mask is deprecated in PyTorch.
        """
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )

    def _run(self, x, memory, key_padding_mask):
        causal = self._causal_mask(x.size(1), x.device)
        attn_w = None
        for layer in self.layers:
            x, attn_w = layer(x, memory, causal, key_padding_mask)
        self._last_cross_attn = attn_w                     # [B, L, 49]
        return self.fc(self.norm(x))                       # [B, L, V]

    # -- training: teacher forcing -------------------------------------------#
    def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """features [B, 49, 2048], captions [B, T] -> logits [B, T-1, V]."""
        memory = self.project_features(features)           # [B, 49, 512]
        inp = captions[:, :-1]                             # [B, T-1]
        x = self.drop(self.pos(self.embed(inp) * self.scale))
        # Mask padded query positions so they cannot leak via self-attention.
        pad_mask = inp == PAD_IDX                          # [B, T-1] (True = ignore)
        return self._run(x, memory, pad_mask)              # [B, T-1, V]

    # -- inference: stateless single step over the full prefix ----------------#
    def decode_tokens(self, memory: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """memory = projected feats [B, 49, 512], tokens [B, t] -> logits [B, V]."""
        x = self.drop(self.pos(self.embed(tokens) * self.scale))
        # No padding in the decode prefix, so no key-padding mask.
        logits = self._run(x, memory, None)                # [B, t, V]
        return logits[:, -1]                               # [B, V]
