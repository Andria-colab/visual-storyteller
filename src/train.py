"""
Training engine (Person 2).

Owns the loop that makes both models converge on Colab, plus the operational
resilience that is OUR risk to carry: checkpoint-to-Drive every epoch and
resume after a disconnect.

Design notes:
  - Loss is masked cross-entropy with label smoothing 0.1. PAD_IDX is ignored.
  - The model is expected to expose `forward(features, captions)` returning
    logits [B, T-1, vocab] aligned to predict captions[:, 1:] (teacher forcing).
    This is the frozen forward contract (plan §7).
  - AMP + grad clipping + AdamW. The transformer uses a warmup + inverse-sqrt
    LR schedule; the baseline uses a flat LR (set warmup_steps=0).
  - Validation tracks BLEU-4 (via greedy decode) for early stopping.

This file is written to run unchanged against P1's real model AND against the
Phase-1 stub decoder, so we can prove the loop end-to-end before the real
decoders land.
"""

from __future__ import annotations

import os
import math
import time
from dataclasses import dataclass
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config, PAD_IDX
from .data import Vocabulary
from . import decode as decode_mod
from . import evaluate as eval_mod


# --------------------------------------------------------------------------- #
# LR schedule
# --------------------------------------------------------------------------- #
def inverse_sqrt_lr(step: int, warmup_steps: int, d_model: int) -> float:
    """Transformer 'Noam' schedule scale factor. step is 1-indexed."""
    step = max(step, 1)
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))


# --------------------------------------------------------------------------- #
# Checkpointing — the disconnect-resilience that is ours to own
# --------------------------------------------------------------------------- #
def save_checkpoint(path: str, model, optimizer, epoch: int, best_metric: float, extra: Optional[dict] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
    }
    if extra:
        payload.update(extra)
    tmp = path + ".tmp"
    torch.save(payload, tmp)          # write-then-rename so a crash mid-save
    os.replace(tmp, path)             # never corrupts the good checkpoint


def load_checkpoint(path: str, model, optimizer=None, map_location="cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
@dataclass
class TrainState:
    epoch: int = 0
    best_metric: float = -1.0
    epochs_no_improve: int = 0
    history: dict = None

    def __post_init__(self):
        if self.history is None:
            self.history = {"train_loss": [], "val_loss": [], "val_bleu4": []}


class Trainer:
    def __init__(
        self,
        model,
        vocab: Vocabulary,
        cfg: Config,
        device: str,
        ckpt_path: str,
        is_transformer: bool = False,
        references: Optional[dict] = None,
        val_image_loader: Optional[DataLoader] = None,
    ):
        self.model = model.to(device)
        self.vocab = vocab
        self.cfg = cfg
        self.device = device
        self.ckpt_path = ckpt_path
        self.is_transformer = is_transformer
        self.references = references          # {image_id: [ref_tokens,...]} for val BLEU
        # image-level loader (one entry per unique image) for caption-quality BLEU
        self.val_image_loader = val_image_loader

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=PAD_IDX, label_smoothing=cfg.label_smoothing
        )
        self.optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=cfg.lr, weight_decay=cfg.weight_decay,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and device.startswith("cuda"))
        self.state = TrainState()
        self._step = 0

    # -- one optimization step worth of loss --------------------------------#
    def _compute_loss(self, features, captions) -> torch.Tensor:
        # logits: [B, T-1, V] predicting captions[:, 1:]
        logits = self.model(features, captions)
        targets = captions[:, 1:]
        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        )
        return loss

    def _set_lr(self):
        if self.is_transformer and self.cfg.warmup_steps > 0:
            scale = inverse_sqrt_lr(self._step, self.cfg.warmup_steps, self.cfg.embed_dim)
            for g in self.optimizer.param_groups:
                g["lr"] = scale

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total, n = 0.0, 0
        for features, captions, _ in loader:
            if torch.is_tensor(features):
                features = features.to(self.device)
            captions = captions.to(self.device)
            self._step += 1
            self._set_lr()

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
                loss = self._compute_loss(features, captions)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total += loss.item() * captions.size(0)
            n += captions.size(0)
        return total / max(n, 1)

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> tuple[float, float]:
        """Returns (val_loss, val_bleu4)."""
        self.model.eval()
        total, n = 0.0, 0
        for features, captions, _ in loader:
            if torch.is_tensor(features):
                features = features.to(self.device)
            captions = captions.to(self.device)
            loss = self._compute_loss(features, captions)
            total += loss.item() * captions.size(0)
            n += captions.size(0)
        val_loss = total / max(n, 1)

        bleu4 = 0.0
        if self.references is not None and self.val_image_loader is not None:
            bleu4 = self._val_bleu4()
        return val_loss, bleu4

    @torch.no_grad()
    def _val_bleu4(self) -> float:
        """Greedy-decode each unique val image once, corpus-BLEU-4 vs all refs.

        Uses the image-level loader (ImageFeatureDataset / image_collate_fn) so
        each image maps cleanly to its references by id — giving early stopping a
        real caption-quality signal, not just loss.
        """
        self.model.eval()
        hyps: dict[str, list[str]] = {}
        for image_ids, features in self.val_image_loader:
            features = features.to(self.device)
            token_lists = decode_mod.greedy_decode(
                self.model, features, self.vocab,
                max_len=self.cfg.max_caption_len, device=self.device,
            )
            for img_id, ids in zip(image_ids, token_lists):
                hyps[img_id] = self.vocab.decode(ids)
        if not hyps:
            return 0.0
        scores = eval_mod.score_bleu(hyps, self.references)
        return scores.get("BLEU-4", 0.0)

    # -- full training with resume + early stopping -------------------------#
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, resume: bool = True) -> TrainState:
        if resume and os.path.exists(self.ckpt_path):
            ckpt = load_checkpoint(self.ckpt_path, self.model, self.optimizer, self.device)
            self.state.epoch = ckpt.get("epoch", 0)
            self.state.best_metric = ckpt.get("best_metric", -1.0)
            print(f"[resume] from epoch {self.state.epoch}, best={self.state.best_metric:.4f}")

        for epoch in range(self.state.epoch + 1, self.cfg.num_epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss, val_bleu4 = self.validate(val_loader)
            dt = time.time() - t0

            self.state.epoch = epoch
            self.state.history["train_loss"].append(train_loss)
            self.state.history["val_loss"].append(val_loss)
            self.state.history["val_bleu4"].append(val_bleu4)
            print(f"epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f} "
                  f"| BLEU-4 {val_bleu4:.4f} | {dt:.0f}s")

            # checkpoint every epoch (Drive); track best on the chosen metric
            metric = val_bleu4 if self.references is not None else -val_loss
            save_checkpoint(self.ckpt_path, self.model, self.optimizer, epoch, self.state.best_metric)
            if metric > self.state.best_metric:
                self.state.best_metric = metric
                self.state.epochs_no_improve = 0
                save_checkpoint(self.ckpt_path.replace(".pt", "_best.pt"),
                                self.model, self.optimizer, epoch, metric)
            else:
                self.state.epochs_no_improve += 1
                if self.state.epochs_no_improve >= self.cfg.early_stop_patience:
                    print(f"[early-stop] no improvement for {self.cfg.early_stop_patience} epochs")
                    break

        return self.state
