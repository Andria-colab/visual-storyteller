"""
Image encoder (Person 1).

The shared, FROZEN ResNet-50 backbone that turns a raw image into a grid of
region features both decoders attend over:

    image [B, 3, 224, 224]  ->  region features [B, 49, 2048]

49 = the 7x7 spatial map from ResNet-50's last conv block (layer4); 2048 is that
block's channel dim. We drop ResNet's avgpool + fc head and keep the spatial map,
flattened to 49 region vectors.

Contract (see config.py):
  - output is RAW [B, 49, 2048] (feature_dim=2048). The 2048->512 projection
    lives inside each decoder, NOT here, so `features.h5` stores raw 2048-d
    features and either decoder can project them its own way.
  - the encoder is frozen (requires_grad=False) and ALWAYS runs in eval mode so
    its BatchNorm layers keep their pretrained running statistics. This is the
    single most important correctness point: a frozen pretrained CNN run in
    train mode lets BN stats drift and silently corrupts every feature.

`build_image_transform()` is the eval-time PIL->tensor transform. It is the SAME
transform used by `scripts/precompute_features.py` (to build features.h5) and by
`generate_caption` (to caption a raw image), so train-time and inference-time
features can never diverge.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models

# ImageNet statistics ResNet-50 was trained with.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ResNetEncoder(nn.Module):
    """Frozen ResNet-50 -> [B, 49, 2048] region features.

    Args:
        pretrained: load ImageNet weights (default True). Pass False for fast,
            offline construction in tests where the features are not used.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = models.resnet50(weights=weights)
        # Everything except avgpool (-2) and fc (-1): conv1..layer4.
        # Output for a 224x224 input is [B, 2048, 7, 7].
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Freeze: these params must never receive gradients, and the AdamW filter
        # in train.py (p.requires_grad) relies on this to exclude them.
        for p in self.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

    def train(self, mode: bool = True):
        """Keep the encoder permanently in eval mode.

        The backbone is frozen, so BatchNorm must always use its pretrained
        running stats and dropout must be off — regardless of whether the parent
        module is put in train mode. We therefore ignore `mode` and force eval.
        """
        return super().train(False)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images [B, 3, H, W] -> region features [B, 49, 2048]."""
        feats = self.backbone(images)              # [B, 2048, 7, 7]
        b, c, h, w = feats.shape
        # [B, C, H, W] -> [B, H*W, C] = [B, 49, 2048]
        feats = feats.flatten(2).transpose(1, 2).contiguous()
        return feats


def build_image_transform() -> T.Compose:
    """Eval-time transform: PIL.Image -> normalized [3, 224, 224] tensor.

    Canonical ImageNet eval pipeline (resize the short side to 256, center-crop
    224) so aspect ratio is preserved. No augmentation: the encoder is frozen and
    features are cached, so train-time augmentation would have no effect.
    """
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
