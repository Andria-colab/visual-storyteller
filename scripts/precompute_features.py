"""
Precompute ResNet-50 features (Person 1).

Runs the frozen encoder over every image once and caches the [49, 2048] region
features to `features.h5`, keyed by image filename (e.g.
`1000268201_693b08cb0e.jpg`). Training reads from this cache instead of running
the CNN, which is what makes 30 epochs on Colab feasible.

This must use the SAME transform + encoder as `generate_caption` (both import
from src.encoder) so cached features and inference-time features match exactly.

Run (local):  python -m scripts.precompute_features
Run (Colab):  python -m scripts.precompute_features --colab

Resumable: re-running skips images already present in features.h5, so a Colab
disconnect just means you re-run and it continues where it left off.
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import default_config, Config
from src.encoder import ResNetEncoder, build_image_transform


_IMAGE_EXTS = (".jpg", ".jpeg", ".png")


class _ImageDataset(Dataset):
    """Yields (filename, image_tensor) for each pending image; None on failure."""

    def __init__(self, images_dir: str, filenames: list[str], transform):
        self.images_dir = images_dir
        self.filenames = filenames
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        name = self.filenames[idx]
        try:
            img = Image.open(os.path.join(self.images_dir, name)).convert("RGB")
            return name, self.transform(img)
        except Exception as e:                       # corrupt / unreadable image
            print(f"[skip] {name}: {e}")
            return name, None


def _collate(batch):
    names, tensors = [], []
    for name, t in batch:
        if t is not None:
            names.append(name)
            tensors.append(t)
    if not tensors:
        return [], None
    return names, torch.stack(tensors, dim=0)


def precompute(cfg: Config, batch_size: int = 64, num_workers: int = 2,
               flush_every: int = 20) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    print(f"images_dir : {cfg.images_dir}")
    print(f"features_h5: {cfg.features_file}")

    if not os.path.isdir(cfg.images_dir):
        raise FileNotFoundError(f"images_dir not found: {cfg.images_dir}")

    encoder = ResNetEncoder().to(device)
    encoder.eval()
    transform = build_image_transform()

    all_imgs = sorted(
        f for f in os.listdir(cfg.images_dir) if f.lower().endswith(_IMAGE_EXTS)
    )
    if not all_imgs:
        raise FileNotFoundError(f"no images found in {cfg.images_dir}")

    os.makedirs(os.path.dirname(os.path.abspath(cfg.features_file)), exist_ok=True)
    # 'a': create if missing, otherwise append — this is what makes it resumable.
    h5 = h5py.File(cfg.features_file, "a")
    done = set(h5.keys())
    pending = [f for f in all_imgs if f not in done]
    print(f"{len(all_imgs)} images total | {len(done)} already cached | {len(pending)} to do")

    if not pending:
        h5.close()
        print("nothing to do — features.h5 is complete.")
        return

    ds = _ImageDataset(cfg.images_dir, pending, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=_collate)

    written = 0
    try:
        for bi, (names, images) in enumerate(tqdm(loader, desc="extract")):
            if images is None:
                continue
            images = images.to(device)
            with torch.no_grad():
                feats = encoder(images)              # [B, 49, 2048]
            feats = feats.float().cpu().numpy()
            for name, arr in zip(names, feats):
                h5.create_dataset(name, data=arr)    # arr [49, 2048] float32
                written += 1
            if (bi + 1) % flush_every == 0:
                h5.flush()                           # durable partial progress
    finally:
        h5.flush()
        h5.close()
    print(f"done — wrote {written} new feature sets to {cfg.features_file}")


def main():
    ap = argparse.ArgumentParser(description="Precompute ResNet-50 features -> features.h5")
    ap.add_argument("--colab", action="store_true", help="use Colab paths from config")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    cfg = Config().use_colab_paths() if args.colab else default_config()
    precompute(cfg, batch_size=args.batch_size, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
