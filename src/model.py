"""
Model wrapper (Person 1): EncoderDecoder.

The single integration surface between the modeling code and Person 2's
training / decoding / evaluation code. It bundles the frozen ResNet-50 encoder
with one of the two decoders and exposes exactly the three methods the rest of
the codebase calls (a drop-in for the StubModel in scripts/smoke_test.py):

    EncoderDecoder(vocab_size, cfg, variant='lstm' | 'transformer')

    forward(features[B,49,2048], captions[B,T]) -> logits [B, T-1, V]   (training)
    encode_image(features[B,49,2048])           -> memory                (inference)
    decode_step(memory, tokens[B,t])            -> logits [B, V]         (inference)

`memory` is the projected region features [B, 49, 512] for BOTH variants, so the
decode loop in decode.py can treat it as an opaque object. The 2048->512
projection lives inside each decoder.

The ResNet encoder is held on the model (decode.py / generate_caption reach it
via `model.encoder`) and is frozen + permanently in eval mode. Because its
params have requires_grad=False, the AdamW filter in train.py excludes them and
the grad-clip skips their (absent) gradients automatically.
"""

from __future__ import annotations

import torch.nn as nn

from .config import Config
from .encoder import ResNetEncoder
from .decoder_lstm import LSTMDecoder
from .decoder_transformer import TransformerDecoder


class EncoderDecoder(nn.Module):
    def __init__(self, vocab_size: int, cfg: Config, variant: str = "lstm",
                 pretrained_encoder: bool = True):
        super().__init__()
        if variant not in ("lstm", "transformer"):
            raise ValueError(f"variant must be 'lstm' or 'transformer', got {variant!r}")
        self.cfg = cfg
        self.variant = variant

        self.encoder = ResNetEncoder(pretrained=pretrained_encoder)
        if variant == "lstm":
            self.decoder = LSTMDecoder(vocab_size, cfg)
        else:
            self.decoder = TransformerDecoder(vocab_size, cfg)

    # -- training -------------------------------------------------------------#
    def forward(self, features, captions):
        """features [B, 49, 2048], captions [B, T] -> logits [B, T-1, V]."""
        return self.decoder(features, captions)

    # -- inference (used by decode.py) ---------------------------------------#
    def encode_image(self, features):
        """features [B, 49, 2048] -> memory [B, 49, 512]."""
        return self.decoder.project_features(features)

    def decode_step(self, memory, tokens):
        """memory, tokens [B, t] -> next-token logits [B, V]."""
        return self.decoder.decode_tokens(memory, tokens)

    # -- keep the frozen encoder in eval even when the model is in train mode -#
    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.train(False)
        return self
