"""
Baseline decoder (Person 1): LSTM + Bahdanau attention ("Show, Attend and Tell").

At every step the decoder attends over the 49 image regions to form a context
vector, concatenates it with the current word embedding, and steps an LSTM cell.
The attention weights are interpretable per word and are exposed for the heatmap
visualisations in `viz.py`.

Shapes / contract (see config.py and model.py):
  - region features come in RAW as [B, 49, 2048]; `project_features` maps them to
    [B, 49, 512] (embed_dim). The wrapper calls this once and treats the result
    as the opaque `memory` it passes back into `decode_tokens`.
  - forward(features, captions) -> logits [B, T-1, V], teacher-forced to predict
    captions[:, 1:].
  - decode_tokens(memory, tokens[B, t]) -> logits [B, V] for the NEXT token. It is
    STATELESS: it re-rolls the LSTM over the whole prefix each call (cheap at
    t <= 22), matching the stateless decode loop in decode.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import Config


class BahdanauAttention(nn.Module):
    """Additive attention over the 49 region features given the LSTM hidden state."""

    def __init__(self, feat_dim: int, hidden_dim: int, attn_dim: int):
        super().__init__()
        self.W_feat = nn.Linear(feat_dim, attn_dim)
        self.W_hid = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1)

    def forward(self, proj_feats: torch.Tensor, h: torch.Tensor):
        """proj_feats [B, 49, D], h [B, H] -> (context [B, D], alpha [B, 49])."""
        # e_i = v^T tanh(W_feat f_i + W_hid h)
        e = self.v(torch.tanh(self.W_feat(proj_feats) + self.W_hid(h).unsqueeze(1)))
        e = e.squeeze(-1)                                  # [B, 49]
        alpha = torch.softmax(e, dim=1)                    # [B, 49]
        context = (alpha.unsqueeze(-1) * proj_feats).sum(dim=1)   # [B, D]
        return context, alpha


class LSTMDecoder(nn.Module):
    def __init__(self, vocab_size: int, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.embed_dim
        h = cfg.lstm_hidden

        # 2048 -> 512 region projection lives here (raw features in).
        self.feat_proj = nn.Linear(cfg.feature_dim, d)
        self.embed = nn.Embedding(vocab_size, d, padding_idx=0)
        self.attention = BahdanauAttention(feat_dim=d, hidden_dim=h, attn_dim=d)
        # LSTM input = [word embedding ; attention context] = 512 + 512.
        self.lstm = nn.LSTMCell(d + d, h)
        # Initial (h, c) derived from the mean of the projected region features.
        self.init_h = nn.Linear(d, h)
        self.init_c = nn.Linear(d, h)
        self.dropout = nn.Dropout(cfg.lstm_dropout)
        self.fc = nn.Linear(h, vocab_size)

        # Most recent per-step attention map [B, 49], stashed for viz.
        self._last_alpha: torch.Tensor | None = None

    # -- shared with the wrapper's encode_image -------------------------------#
    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        """[B, 49, 2048] -> [B, 49, 512]  (this is the opaque `memory`)."""
        return self.feat_proj(features)

    def _init_state(self, proj_feats: torch.Tensor):
        mean = proj_feats.mean(dim=1)                      # [B, 512]
        return torch.tanh(self.init_h(mean)), torch.tanh(self.init_c(mean))

    # -- training: teacher forcing -------------------------------------------#
    def forward(self, features: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """features [B, 49, 2048], captions [B, T] -> logits [B, T-1, V]."""
        proj = self.project_features(features)             # [B, 49, 512]
        h, c = self._init_state(proj)
        emb = self.embed(captions[:, :-1])                 # [B, T-1, 512]

        outputs = []
        for t in range(emb.size(1)):
            context, _ = self.attention(proj, h)           # [B, 512]
            lstm_in = torch.cat([emb[:, t], context], dim=1)   # [B, 1024]
            h, c = self.lstm(lstm_in, (h, c))
            outputs.append(self.fc(self.dropout(h)))       # [B, V]
        return torch.stack(outputs, dim=1)                 # [B, T-1, V]

    # -- inference: stateless single step over the full prefix ----------------#
    def decode_tokens(self, memory: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """memory = projected feats [B, 49, 512], tokens [B, t] -> logits [B, V].

        Re-rolls the LSTM over the whole prefix and returns logits for the token
        that follows `tokens`. The final attention map is saved to `_last_alpha`.
        """
        proj = memory
        h, c = self._init_state(proj)
        emb = self.embed(tokens)                           # [B, t, 512]
        alpha = None
        for i in range(emb.size(1)):
            context, alpha = self.attention(proj, h)
            lstm_in = torch.cat([emb[:, i], context], dim=1)
            h, c = self.lstm(lstm_in, (h, c))
        self._last_alpha = alpha                           # [B, 49]
        return self.fc(self.dropout(h))                    # [B, V]
