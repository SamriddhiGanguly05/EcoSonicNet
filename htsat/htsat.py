# ====================================================
# htsat.py — Hierarchical Token-Semantic Audio Transformer
# ✅ Clean version with absolute imports
# ====================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from itertools import repeat
from typing import List

from .layers import PatchEmbed, Mlp, DropPath, trunc_normal_, to_2tuple
from .utils import do_mixup, interpolate
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

# ====================================================
# Swin Transformer Block for HTSAT
# ====================================================

class SwinTransformerBlock(nn.Module):
    # Your block implementation...
    pass  # Replace with actual Swin Transformer block if needed

# ====================================================
# Main HTSAT Model
# ====================================================

class HTSAT_Swin_Transformer(nn.Module):
    def __init__(
        self,
        spec_size=224,
        patch_size=4,
        in_chans=1,
        num_classes=527,
        window_size=7,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        ffn_expansion_factor=4,
        qkv_bias=True,
        config=None
    ):
        super(HTSAT_Swin_Transformer, self).__init__()
        self.spec_size = spec_size
        self.num_classes = num_classes
        self.config = config  # optional config for torchlibrosa

        # Patch embedding layer
        self.patch_embed = PatchEmbed(
            img_size=spec_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution

        # Dropout layer
        self.pos_drop = nn.Dropout(p=0.0)

        # Add Swin Transformer stages here if needed
        # For example:
        # self.stage1 = SwinTransformerBlock(...)

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)

        # SpecAug and Logmel if needed
        if config:
            self.spectrogram_extractor = Spectrogram(
                n_fft=config.window_size,
                hop_length=config.hop_size,
                win_length=config.window_size,
                window='hann'
            )
            self.logmel_extractor = LogmelFilterBank(
                sr=config.sample_rate,
                n_fft=config.window_size,
                n_mels=config.mel_bins,
                fmin=config.fmin,
                fmax=config.fmax
            )
            self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                                   freq_drop_width=8, freq_stripes_num=2)
        else:
            self.spectrogram_extractor = None
            self.logmel_extractor = None
            self.spec_augmenter = None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: (B, 1, mel_bins, time_steps)
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        # Add Swin Transformer stages here if needed
        # x = self.stage1(x)

        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        logits = self.head(x)
        return logits
