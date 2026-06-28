"""
Shared configuration — the frozen contract between Person 1 (modeling) and
Person 2 (data / training / eval).

Everything that both halves of the code must agree on lives here so neither
person blocks the other. Freeze these values on Day 2 (see plan §7).

Paths default to a local layout for development and are overridden for Colab
(see `use_colab_paths`). Nothing here should import torch-heavy modules so it
stays cheap to import everywhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict


# --------------------------------------------------------------------------- #
# Special tokens — index order is part of the contract. Do not reorder.
# --------------------------------------------------------------------------- #
PAD_TOKEN = "<pad>"      # index 0  — must be 0 so it doubles as the padding_idx
START_TOKEN = "<start>"  # index 1
END_TOKEN = "<end>"      # index 2
UNK_TOKEN = "<unk>"      # index 3
SPECIAL_TOKENS = (PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN)

PAD_IDX = 0
START_IDX = 1
END_IDX = 2
UNK_IDX = 3


@dataclass
class Config:
    # ---- paths (local defaults; call use_colab_paths() on Colab) --------- #
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir: str = field(default="")          # raw Flickr8k images + captions
    images_dir: str = field(default="")        # the Flickr8k_Dataset images
    captions_file: str = field(default="")     # captions.txt (Flickr8k.token.txt)
    features_file: str = field(default="")     # features.h5 produced by P1's precompute
    vocab_file: str = field(default="")        # vocab.pkl
    checkpoint_dir: str = field(default="")    # baseline.pt / transformer.pt go here

    # ---- vocab ----------------------------------------------------------- #
    min_word_freq: int = 3          # words rarer than this become <unk>
    max_caption_len: int = 22       # tokens between <start> and <end> (incl.)

    # ---- shared model dims (the forward() contract) ---------------------- #
    feature_dim: int = 2048         # ResNet-50 layer4 channel dim
    num_regions: int = 49           # 7x7 spatial map
    embed_dim: int = 512            # d — projection + token embedding size

    # ---- transformer decoder (P1 owns internals, dims shared) ----------- #
    num_layers: int = 4             # tune 3-6
    num_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1

    # ---- baseline (lstm) ------------------------------------------------- #
    lstm_hidden: int = 512
    lstm_dropout: float = 0.5

    # ---- training -------------------------------------------------------- #
    batch_size: int = 64
    num_epochs: int = 30
    lr: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    grad_clip: float = 5.0
    warmup_steps: int = 4000        # transformer warmup + inverse-sqrt schedule
    early_stop_patience: int = 5    # epochs without val-metric improvement
    use_amp: bool = True
    num_workers: int = 2
    seed: int = 42

    # ---- decoding -------------------------------------------------------- #
    beam_size: int = 3              # 3-5

    def use_local_paths(self) -> "Config":
        d = os.path.join(self.project_root, "data")
        self.data_dir = d
        self.images_dir = os.path.join(d, "Images")
        self.captions_file = os.path.join(d, "captions.txt")
        self.features_file = os.path.join(d, "features.h5")
        self.vocab_file = os.path.join(d, "vocab.pkl")
        self.checkpoint_dir = os.path.join(self.project_root, "checkpoints")
        return self

    def use_colab_paths(self, drive_root: str = "/content/drive/MyDrive/visual_storyteller") -> "Config":
        self.data_dir = "/content/flickr8k"
        self.images_dir = "/content/flickr8k/Images"
        self.captions_file = "/content/flickr8k/captions.txt"
        self.features_file = os.path.join(drive_root, "features.h5")
        self.vocab_file = os.path.join(drive_root, "vocab.pkl")
        self.checkpoint_dir = os.path.join(drive_root, "checkpoints")
        return self

    def to_dict(self) -> dict:
        return asdict(self)


def default_config() -> Config:
    """Local-path config used for development and tests."""
    return Config().use_local_paths()
