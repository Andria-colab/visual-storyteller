"""
Data pipeline (Person 2).

Owns everything between the raw Flickr8k files and a batch of tensors the model
can consume:

    raw captions.txt  ->  cleaned tokens  ->  Vocabulary  ->  CaptionDataset  ->  collate

The CNN features themselves are produced and cached to `features.h5` by P1's
`scripts/precompute_features.py`; this module only *reads* that cache, keyed by
image id. To allow development before features exist, `CaptionDataset` can run
in `return_image_ids=True` mode (no feature lookup) and the trainer skeleton can
feed random features.

Contract (see config.py):
  - features per image: [num_regions, feature_dim]  == [49, 2048]
  - a caption tensor:   [T] of token indices, starting with <start>, ending <end>
  - collate pads captions to the batch max with PAD_IDX (== 0)
"""

from __future__ import annotations

import os
import re
import pickle
from collections import Counter
from typing import Optional

import h5py
import torch
from torch.utils.data import Dataset

from .config import (
    Config,
    SPECIAL_TOKENS,
    PAD_IDX,
    START_IDX,
    END_IDX,
    UNK_IDX,
    START_TOKEN,
    END_TOKEN,
    UNK_TOKEN,
)


# --------------------------------------------------------------------------- #
# Caption cleaning
# --------------------------------------------------------------------------- #
_PUNCT_RE = re.compile(r"[^a-z\s]")        # keep letters + whitespace only
_MULTISPACE_RE = re.compile(r"\s+")


def clean_caption(text: str) -> list[str]:
    """Lowercase, strip punctuation/digits, collapse whitespace -> token list.

    Single-character tokens other than 'a' are dropped (they are almost always
    OCR/typo noise in Flickr8k captions, e.g. stray 's').
    """
    text = text.lower().strip()
    text = _PUNCT_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return [t for t in text.split(" ") if t and (len(t) > 1 or t == "a")]


# --------------------------------------------------------------------------- #
# Raw caption loading
# --------------------------------------------------------------------------- #
def load_captions(captions_file: str) -> dict[str, list[list[str]]]:
    """Parse the Flickr8k caption file into {image_id: [tokens, tokens, ...]}.

    Supports both common formats:
      - CSV (Kaggle):  `image,caption`  with a header line
      - token  (orig): `1000268201_693b08cb0e.jpg#0<TAB>A child ...`
    image_id is the filename without the `#n` suffix.
    """
    captions: dict[str, list[list[str]]] = {}
    with open(captions_file, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    if not lines:
        return captions

    # Skip a CSV header if present.
    start = 0
    first = lines[0].lower()
    if first.startswith("image,") or first.strip() in ("image,caption", "image, caption"):
        start = 1

    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        if "\t" in line:                       # token format
            left, caption = line.split("\t", 1)
            image_id = left.split("#")[0]
        else:                                  # CSV format
            parts = line.split(",", 1)
            if len(parts) != 2:
                continue
            image_id, caption = parts
            image_id = image_id.split("#")[0].strip()

        tokens = clean_caption(caption)
        if tokens:
            captions.setdefault(image_id.strip(), []).append(tokens)

    return captions


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
class Vocabulary:
    """Token <-> index mapping. Special tokens occupy indices 0..3 (see config)."""

    def __init__(self, stoi: dict[str, int], itos: dict[int, str], min_freq: int):
        self.stoi = stoi
        self.itos = itos
        self.min_freq = min_freq

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str], add_special: bool = True) -> list[int]:
        ids = [self.stoi.get(t, UNK_IDX) for t in tokens]
        if add_special:
            ids = [START_IDX] + ids + [END_IDX]
        return ids

    def decode(self, ids: list[int], strip_special: bool = True) -> list[str]:
        out = []
        for i in ids:
            tok = self.itos.get(int(i), UNK_TOKEN)
            if strip_special:
                if i == END_IDX:
                    break
                if i in (PAD_IDX, START_IDX):
                    continue
            out.append(tok)
        return out

    # -- persistence ------------------------------------------------------- #
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"stoi": self.stoi, "itos": self.itos, "min_freq": self.min_freq}, f)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path, "rb") as f:
            d = pickle.load(f)
        return cls(d["stoi"], d["itos"], d["min_freq"])


def build_vocab(
    captions: dict[str, list[list[str]]],
    min_freq: int = 5,
    image_ids: Optional[set[str]] = None,
) -> Vocabulary:
    """Build a Vocabulary from token frequencies.

    If `image_ids` is given, only those images' captions count toward the
    frequencies — pass the TRAIN ids so the vocab never sees val/test text.
    """
    counter: Counter[str] = Counter()
    for img_id, caps in captions.items():
        if image_ids is not None and img_id not in image_ids:
            continue
        for tokens in caps:
            counter.update(tokens)

    itos: dict[int, str] = {i: tok for i, tok in enumerate(SPECIAL_TOKENS)}
    idx = len(SPECIAL_TOKENS)
    for word, freq in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
        if freq >= min_freq:
            itos[idx] = word
            idx += 1
    stoi = {tok: i for i, tok in itos.items()}
    return Vocabulary(stoi, itos, min_freq)


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def make_splits(
    image_ids: list[str],
    sizes: tuple[int, int, int] = (6000, 1000, 1000),
    seed: int = 42,
) -> dict[str, list[str]]:
    """Deterministic train/val/test split over image ids.

    The test split is held out and must stay untouched until final analysis.
    If standard Flickr8k split files are present, prefer those instead (see
    `load_official_splits`).
    """
    ids = sorted(set(image_ids))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(ids), generator=g).tolist()
    ids = [ids[i] for i in perm]

    n_train, n_val, n_test = sizes
    train = ids[:n_train]
    val = ids[n_train : n_train + n_val]
    test = ids[n_train + n_val : n_train + n_val + n_test]
    return {"train": train, "val": val, "test": test}


def load_official_splits(data_dir: str) -> Optional[dict[str, list[str]]]:
    """Use Flickr_8k.{train,dev,test}Images.txt if they exist, else None."""
    names = {
        "train": "Flickr_8k.trainImages.txt",
        "val": "Flickr_8k.devImages.txt",
        "test": "Flickr_8k.testImages.txt",
    }
    out: dict[str, list[str]] = {}
    for split, fname in names.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            out[split] = [ln.strip() for ln in f if ln.strip()]
    return out


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class CaptionDataset(Dataset):
    """One (image_features, caption) pair per sample.

    Each (image, caption) combination is its own training example, so an image
    with 5 captions yields 5 samples. Features are read lazily from `features.h5`
    keyed by image id; the file handle is opened per worker (h5py is not
    fork-safe across the initial handle).

    Set `return_image_ids=True` to skip feature lookup entirely — used by the
    Phase-1 skeleton test before features.h5 exists.
    """

    def __init__(
        self,
        image_ids: list[str],
        captions: dict[str, list[list[str]]],
        vocab: Vocabulary,
        features_file: Optional[str] = None,
        max_len: int = 22,
        return_image_ids: bool = False,
    ):
        self.vocab = vocab
        self.features_file = features_file
        self.max_len = max_len
        self.return_image_ids = return_image_ids or features_file is None
        self._h5: Optional[h5py.File] = None

        # Flatten into (image_id, caption_tokens) pairs.
        self.samples: list[tuple[str, list[str]]] = []
        for img_id in image_ids:
            for tokens in captions.get(img_id, []):
                self.samples.append((img_id, tokens[:max_len]))

    def __len__(self) -> int:
        return len(self.samples)

    def _features(self, image_id: str) -> torch.Tensor:
        if self._h5 is None:
            self._h5 = h5py.File(self.features_file, "r")
        arr = self._h5[image_id][:]          # [num_regions, feature_dim]
        return torch.from_numpy(arr).float()

    def __getitem__(self, idx: int):
        image_id, tokens = self.samples[idx]
        caption = torch.tensor(self.vocab.encode(tokens), dtype=torch.long)
        if self.return_image_ids:
            return image_id, caption
        return self._features(image_id), caption


class ImageFeatureDataset(Dataset):
    """One entry per UNIQUE image (not per caption) — for validation/test BLEU.

    Captions are flattened to (image, caption) pairs for training, but metric
    scoring decodes each image ONCE and compares the single hypothesis against
    all 5 references. This dataset yields (image_id, features) so the trainer can
    decode per image and map results back to references by id.
    """

    def __init__(self, image_ids: list[str], features_file: str):
        # de-dupe while preserving order
        seen: set[str] = set()
        self.image_ids = [i for i in image_ids if not (i in seen or seen.add(i))]
        self.features_file = features_file
        self._h5: Optional[h5py.File] = None

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        if self._h5 is None:
            self._h5 = h5py.File(self.features_file, "r")
        image_id = self.image_ids[idx]
        feats = torch.from_numpy(self._h5[image_id][:]).float()
        return image_id, feats


def image_collate_fn(batch):
    """Collate for ImageFeatureDataset -> (list[image_id], features[B,49,2048])."""
    ids, feats = zip(*batch)
    return list(ids), torch.stack(feats, dim=0)


def collate_fn(batch):
    """Pad captions to the batch max with PAD_IDX.

    Returns (features_or_ids, captions[B,T], lengths[B]). `features` is either a
    stacked float tensor [B, num_regions, feature_dim] or a tuple of image ids
    when the dataset is in return_image_ids mode.
    """
    firsts, captions = zip(*batch)
    lengths = torch.tensor([c.size(0) for c in captions], dtype=torch.long)
    max_t = int(lengths.max())
    padded = torch.full((len(captions), max_t), PAD_IDX, dtype=torch.long)
    for i, c in enumerate(captions):
        padded[i, : c.size(0)] = c

    if torch.is_tensor(firsts[0]):
        features = torch.stack(firsts, dim=0)
    else:
        features = firsts                    # tuple of image ids (skeleton mode)
    return features, padded, lengths


# --------------------------------------------------------------------------- #
# Convenience: references for BLEU (all 5 captions per test image)
# --------------------------------------------------------------------------- #
def references_for(
    image_ids: list[str],
    captions: dict[str, list[list[str]]],
) -> dict[str, list[list[str]]]:
    """{image_id: [ref_tokens, ...]} for metric scoring against all refs."""
    return {img: captions.get(img, []) for img in image_ids}
