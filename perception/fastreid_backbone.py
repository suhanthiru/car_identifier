"""Standalone loader for FastReID's VeRi-776 SBS(R50-ibn) checkpoint.

perception/embedder.py's docstring flagged this as a possible future swap:
JD AI's FastReID (github.com/JDAI-CV/fast-reid, Apache 2.0) ships a
vehicle-ReID-FINETUNED checkpoint (97.0% Rank-1 / 81.9% mAP on VeRi-776),
unlike this project's default OSNet backbone (ImageNet-pretrained only,
never trained to distinguish vehicle identity). Investigating why real
CityFlow cross-camera recognition wasn't firing found the default
embedder's raw similarity is actually ANTI-discriminative on real footage
(AUC 0.10 vs hard negatives — different cars of the same color score
HIGHER than the same car does with itself across cameras); this is the
fix being tried.

The architecture (ResNet50 + IBN blocks in layers 1-3 only + 2 non-local
blocks in layer2 + 3 in layer3 + a learnable GeM pool + a BNNeck) is
reimplemented here to exactly match the checkpoint's state_dict, following
FastReID's own source (fastreid/modeling/backbones/resnet.py,
fastreid/layers/{non_local,pooling,batch_norm}.py,
fastreid/modeling/heads/embedding_head.py — verified against the actual
checkpoint's key names/shapes, not reconstructed from memory) WITHOUT
installing the full training framework (config system, data loaders,
losses) this project doesn't need. Not a training implementation: no
losses, no classifier head, inference (eval-mode forward) only.

Checkpoint: JDAI-CV/fast-reid release v0.1.1, veri_sbs_R50-ibn.pth
(https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth).
Config: configs/VeRi/sbs_R50-ibn.yml (WITH_IBN, WITH_NL, GeneralizedMeanPoolingP,
NECK_FEAT=after, INPUT.SIZE_TEST=[256,256]) + inherited Base-SBS/Base-bagtricks
defaults (LAST_STRIDE=1, FEAT_DIM=2048, WITH_BNNECK=True).
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import torch
from torch import nn

DEFAULT_CHECKPOINT = Path("models/veri_sbs_R50-ibn.pth")
INPUT_SIZE = 256  # SIZE_TEST: [256, 256] in configs/VeRi/sbs_R50-ibn.yml
# This backbone (ResNet50 + non-local attention blocks at 256x256) is far
# heavier than OSNet-x0_25's 128x256 -- non-local blocks materialize an
# H*W x H*W attention matrix per stage, so batching the whole crop list at
# once (fine for OSNet) exhausts memory on CPU. Chunk instead.
BATCH_CHUNK = 16


class IBN(nn.Module):
    """Half the channels get InstanceNorm, half get BatchNorm, concatenated
    back — IBN-Net's identity-preserving normalization split. Applied only
    to the first 1x1 conv's output in layers 1-3 (never layer4, matching
    FastReID's resnet.py, which omits with_ibn on the layer4 call)."""

    def __init__(self, planes: int):
        super().__init__()
        self.half = planes // 2
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        a, b = torch.split(x, self.half, 1)
        return torch.cat((self.IN(a.contiguous()), self.BN(b.contiguous())), 1)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, with_ibn, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = IBN(planes) if with_ibn else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class NonLocalBlock(nn.Module):
    """FastReID's embedded-Gaussian non-local block. `inter_channels` is
    hardcoded to 1 to match the checkpoint exactly -- FastReID's own source
    computes it as `reduc_ratio // reduc_ratio`, which is always 1
    regardless of reduc_ratio (verified against the checkpoint's theta/phi/g
    conv shapes, all (1, C, 1, 1); reproduced faithfully, not "fixed", since
    the goal is loading their exact trained weights)."""

    def __init__(self, channels: int):
        super().__init__()
        self.g = nn.Conv2d(channels, 1, 1)
        self.theta = nn.Conv2d(channels, 1, 1)
        self.phi = nn.Conv2d(channels, 1, 1)
        self.W = nn.Sequential(nn.Conv2d(1, channels, 1), nn.BatchNorm2d(channels))

    def forward(self, x):
        b, c, h, w = x.shape
        g_x = self.g(x).view(b, 1, -1).permute(0, 2, 1)
        theta_x = self.theta(x).view(b, 1, -1).permute(0, 2, 1)
        phi_x = self.phi(x).view(b, 1, -1)
        f = torch.matmul(theta_x, phi_x) / (h * w)
        y = torch.matmul(f, g_x).permute(0, 2, 1).contiguous().view(b, 1, h, w)
        return self.W(y) + x


class GeneralizedMeanPoolingP(nn.Module):
    """Learnable-p GeM pooling: p=1 is average pooling, p=inf is max
    pooling; p is a trained parameter here (heads.pool_layer.p)."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * 3.0)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return nn.functional.adaptive_avg_pool2d(x, 1).pow(1.0 / self.p)


def _make_layer(inplanes, planes, blocks, stride, with_ibn):
    downsample = None
    if stride != 1 or inplanes != planes * Bottleneck.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * Bottleneck.expansion, 1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * Bottleneck.expansion))
    layers = [Bottleneck(inplanes, planes, with_ibn, stride, downsample)]
    inplanes = planes * Bottleneck.expansion
    for _ in range(1, blocks):
        layers.append(Bottleneck(inplanes, planes, with_ibn))
    return nn.Sequential(*layers), inplanes


class ResNet50IbnNL(nn.Module):
    """LAST_STRIDE=1 (not 2): layer4 keeps a higher spatial resolution than
    a classification ResNet50, standard for ReID backbones."""

    NL_LAYERS = (0, 2, 3, 0)   # non-local blocks per stage, depth=50x
    BLOCKS = (3, 4, 6, 3)      # ResNet50 bottleneck counts per stage

    def __init__(self, last_stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        inplanes = 64
        self.layer1, inplanes = _make_layer(inplanes, 64, self.BLOCKS[0], 1, with_ibn=True)
        self.layer2, inplanes = _make_layer(inplanes, 128, self.BLOCKS[1], 2, with_ibn=True)
        self.layer3, inplanes = _make_layer(inplanes, 256, self.BLOCKS[2], 2, with_ibn=True)
        self.layer4, inplanes = _make_layer(inplanes, 512, self.BLOCKS[3], last_stride, with_ibn=False)

        # Inserted after the LAST `n` blocks of each stage (see FastReID's
        # NL_k_idx = sorted(blocks - (i+1) for i in range(n)) -- verified
        # against the checkpoint: NL_2 after blocks {2,3}, NL_3 after {3,4,5}.
        self.NL_2 = nn.ModuleList(NonLocalBlock(512) for _ in range(self.NL_LAYERS[1]))
        self.NL_2_idx = list(range(self.BLOCKS[1] - self.NL_LAYERS[1], self.BLOCKS[1]))
        self.NL_3 = nn.ModuleList(NonLocalBlock(1024) for _ in range(self.NL_LAYERS[2]))
        self.NL_3_idx = list(range(self.BLOCKS[2] - self.NL_LAYERS[2], self.BLOCKS[2]))

    def _run_stage(self, stage, x, nl_blocks=None, nl_idx=None):
        nl_blocks = nl_blocks or []
        nl_idx = nl_idx or []
        counter = 0
        for i, block in enumerate(stage):
            x = block(x)
            if counter < len(nl_idx) and i == nl_idx[counter]:
                x = nl_blocks[counter](x)
                counter += 1
        return x

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self._run_stage(self.layer1, x)
        x = self._run_stage(self.layer2, x, self.NL_2, self.NL_2_idx)
        x = self._run_stage(self.layer3, x, self.NL_3, self.NL_3_idx)
        x = self._run_stage(self.layer4, x)
        return x


class _Head(nn.Module):
    """GeM pool -> BNNeck. Eval-mode FastReID always returns the
    post-bottleneck feature regardless of NECK_FEAT (that config only
    matters for choosing the triplet-loss input during training)."""

    def __init__(self, feat_dim: int = 2048):
        super().__init__()
        self.pool_layer = GeneralizedMeanPoolingP()
        self.bottleneck = nn.Sequential(nn.BatchNorm2d(feat_dim))

    def forward(self, x):
        return self.bottleneck(self.pool_layer(x))[..., 0, 0]


class FastReidVeriModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("pixel_mean", torch.zeros(1, 3, 1, 1))
        self.register_buffer("pixel_std", torch.ones(1, 3, 1, 1))
        self.backbone = ResNet50IbnNL(last_stride=1)
        self.heads = _Head(feat_dim=2048)

    def forward(self, x):
        x = (x - self.pixel_mean) / self.pixel_std
        return self.heads(self.backbone(x))


# Keys present in the released checkpoint but not needed for inference:
# `heads.classifier.weight` (the training-time ID classifier) and
# `heads.bnneck.num_batches_tracked` (a stray buffer from an older internal
# head naming in the checkpoint's training run, unrelated to the current
# `heads.bottleneck` module this class defines).
_EXPECTED_UNUSED_KEYS = {"heads.classifier.weight", "heads.bnneck.num_batches_tracked"}


def load_fastreid_veri_model(checkpoint_path: Path = DEFAULT_CHECKPOINT) -> FastReidVeriModel:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]
    model = FastReidVeriModel()
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        raise RuntimeError(
            f"fastreid checkpoint load is missing keys the architecture "
            f"needs: {result.missing_keys}")
    unexpected = set(result.unexpected_keys) - _EXPECTED_UNUSED_KEYS
    if unexpected:
        raise RuntimeError(
            f"fastreid checkpoint has unrecognized keys not accounted for "
            f"by the reimplemented architecture: {sorted(unexpected)}")
    model.eval()
    return model


class FastReidEmbedder:
    """Drop-in replacement for perception.embedder.ReidEmbedder's interface
    (.embed / .embed_batch), backed by the VeRi-776-finetuned FastReID
    checkpoint instead of the ImageNet-pretrained OSNet default."""

    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT):
        self._checkpoint_path = checkpoint_path
        self._model: FastReidVeriModel | None = None
        self._load_lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            self._model = load_fastreid_veri_model(self._checkpoint_path)

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        return self.embed_batch([crop_bgr])[0]

    def embed_batch(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        import cv2

        self._load()
        prepped = []
        for crop in crops_bgr:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32)
            prepped.append(rgb.transpose(2, 0, 1))   # HWC -> CHW, stays 0-255

        chunks = []
        with torch.no_grad():
            for i in range(0, len(prepped), BATCH_CHUNK):
                x = torch.from_numpy(np.stack(prepped[i:i + BATCH_CHUNK]))
                chunks.append(self._model(x).cpu().numpy())
        feats = np.concatenate(chunks, axis=0).astype(np.float32)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        return feats / np.maximum(norms, 1e-12)
