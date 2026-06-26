"""
Phase-1 smoke test (Person 2).

Proves our half wires together end to end WITHOUT P1's real model:
  data.py (vocab + dataset + collate) -> train.py (loss + AMP + checkpoint +
  resume) -> decode.py (greedy + beam) -> evaluate.py (BLEU).

Uses synthetic captions, a synthetic features.h5, and a tiny stub decoder that
honours the frozen model contract (forward / encode_image / decode_step). If
this runs clean, the milestone "one tiny batch flows features -> decoder -> loss
with no errors" is met and we can drop P1's real model in unchanged.

Run:  python -m scripts.smoke_test
"""

from __future__ import annotations

import os
import sys
import tempfile

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import default_config
from src import data as D
from src import evaluate as E
from src import decode as DEC
from src.train import Trainer


# --------------------------------------------------------------------------- #
# A tiny stand-in for P1's EncoderDecoder. Honours the contract only.
# --------------------------------------------------------------------------- #
class StubModel(nn.Module):
    def __init__(self, vocab_size: int, cfg):
        super().__init__()
        self.proj = nn.Linear(cfg.feature_dim, cfg.embed_dim)
        self.embed = nn.Embedding(vocab_size, cfg.embed_dim, padding_idx=0)
        self.rnn = nn.GRU(cfg.embed_dim, cfg.embed_dim, batch_first=True)
        self.out = nn.Linear(cfg.embed_dim, vocab_size)

    def _memory(self, features):                       # [B,49,2048] -> [B,512]
        return self.proj(features).mean(dim=1)

    def forward(self, features, captions):             # teacher forcing
        h0 = self._memory(features).unsqueeze(0)       # [1,B,512]
        emb = self.embed(captions[:, :-1])             # predict 1:
        seq, _ = self.rnn(emb, h0)
        return self.out(seq)                           # [B, T-1, V]

    # -- inference contract used by decode.py ------------------------------ #
    def encode_image(self, features):
        return self._memory(features).unsqueeze(0)     # [1,B,512] as 'memory'

    def decode_step(self, memory, tokens):             # next-token logits
        emb = self.embed(tokens)
        seq, _ = self.rnn(emb, memory)
        return self.out(seq[:, -1])                    # [B, V]


def make_synthetic_captions(n_images=40):
    words = "a dog runs on the grass child plays with ball man rides bike".split()
    caps = {}
    rng = np.random.default_rng(0)
    for i in range(n_images):
        img = f"img_{i:03d}.jpg"
        caps[img] = []
        for _ in range(5):
            k = int(rng.integers(4, 9))
            caps[img].append(list(rng.choice(words, size=k)))
    return caps


def main():
    cfg = default_config()
    cfg.batch_size = 8
    cfg.num_epochs = 2
    cfg.min_word_freq = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    captions = make_synthetic_captions()
    image_ids = list(captions.keys())
    splits = D.make_splits(image_ids, sizes=(28, 6, 6), seed=1)
    vocab = D.build_vocab(captions, min_freq=cfg.min_word_freq, image_ids=set(splits["train"]))
    print(f"vocab size: {len(vocab)}")

    tmp = tempfile.mkdtemp()
    feats_path = os.path.join(tmp, "features.h5")
    with h5py.File(feats_path, "w") as f:
        for img in image_ids:
            f.create_dataset(img, data=np.random.randn(cfg.num_regions, cfg.feature_dim).astype("float32"))
    print(f"synthetic features.h5 -> {feats_path}")

    def loader(split):
        ds = D.CaptionDataset(splits[split], captions, vocab, features_file=feats_path,
                              max_len=cfg.max_caption_len)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=(split == "train"),
                          collate_fn=D.collate_fn)

    train_loader, val_loader = loader("train"), loader("val")

    # image-level val loader + references -> real BLEU-4 early-stopping signal
    val_img_ds = D.ImageFeatureDataset(splits["val"], feats_path)
    val_img_loader = DataLoader(val_img_ds, batch_size=cfg.batch_size,
                                shuffle=False, collate_fn=D.image_collate_fn)
    val_refs = D.references_for(splits["val"], captions)

    model = StubModel(len(vocab), cfg)
    ckpt = os.path.join(tmp, "stub.pt")
    trainer = Trainer(model, vocab, cfg, device, ckpt, is_transformer=False,
                      references=val_refs, val_image_loader=val_img_loader)

    print("\n-- fit (with BLEU-4 validation) --")
    trainer.fit(train_loader, val_loader, resume=False)
    assert trainer.state.history["val_bleu4"], "BLEU-4 history should be populated"
    print("val BLEU-4 history:", [round(x, 4) for x in trainer.state.history["val_bleu4"]])

    print("\n-- resume (should pick up at saved epoch) --")
    model2 = StubModel(len(vocab), cfg)
    Trainer(model2, vocab, cfg, device, ckpt).fit(train_loader, val_loader, resume=True)

    print("\n-- decode + BLEU --")
    feats, caps, _ = next(iter(val_loader))
    greedy = DEC.greedy_decode(model, feats, vocab, max_len=cfg.max_caption_len, device=device)
    print("greedy[0]:", " ".join(vocab.decode(greedy[0])) or "(empty)")
    beam = DEC.beam_search(model, feats[0], vocab, beam_size=3, max_len=cfg.max_caption_len, device=device)
    print("beam[0]:  ", " ".join(vocab.decode(beam)) or "(empty)")

    hyps = {img: vocab.decode(g) for img, g in zip(splits["val"], greedy)}
    refs = D.references_for(splits["val"], captions)
    scores = E.score_bleu(hyps, refs)
    print("BLEU:", {k: round(v, 3) for k, v in scores.items()})
    print("\ntable:\n" + E.format_score_table({"stub-greedy": scores}))

    print("\nSMOKE TEST PASSED [OK]")


if __name__ == "__main__":
    main()
