import torch
import torch.nn.functional as F
from torch import nn, Tensor


class SelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.view(B, C, H * W).transpose(1, 2)   # [B, H*W, C]
        h, _ = self.attn(h, h, h)
        h = h.transpose(1, 2).view(B, C, H, W)    # [B, C, H, W]
        return x + h                                # Residual


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_classes: int = 43,
        t_emb_dim: int = 256,      # war 128
    ):
        super().__init__()
        self.num_classes = num_classes

        self.t_emb = nn.Sequential(
            nn.Linear(1, t_emb_dim), nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim)
        )
        self.y_emb = nn.Embedding(num_classes + 1, t_emb_dim)
        self.null_token = num_classes

        # Encoder — doppelte Kanalzahl
        self.enc1 = self._block(in_channels, 64)    # war 32
        self.enc2 = self._block(64, 128)            # war 64
        self.enc3 = self._block(128, 256)           # neu: dritte Ebene
        self.pool = nn.MaxPool2d(2)                 # 32→16→8→4

        # Bottleneck + Attention
        self.bottleneck = self._block(256 + t_emb_dim, 512)  # war 64+, 128
        self.attn = SelfAttention(512, num_heads=8)           # war 128, 4

        # Decoder
        self.up2  = nn.ConvTranspose2d(512, 256, 2, stride=2)  # 4→8
        self.dec2 = self._block(512, 256)
        self.attn_dec2 = SelfAttention(256, num_heads=4)        # Attention auch hier

        self.up1  = nn.ConvTranspose2d(256, 128, 2, stride=2)  # 8→16
        self.dec1 = self._block(256, 128)

        self.up0  = nn.ConvTranspose2d(128, 64, 2, stride=2)   # 16→32
        self.dec0 = self._block(128, 64)

        self.out  = nn.Conv2d(64, out_channels, 1)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c), nn.SiLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c), nn.SiLU()
        )

    def forward(self, x: Tensor, t: Tensor, y: Tensor = None, **kwargs) -> Tensor:
        B = x.shape[0]

        t = t.view(-1).float()
        if t.numel() == 1:
            t = t.expand(B)
        emb = self.t_emb(t.view(B, 1))
        if y is None:
            y = torch.full((B,), self.null_token, device=x.device, dtype=torch.long)
        emb = emb + self.y_emb(y)

        # Encoder
        e1 = self.enc1(x)               # [B, 64,  32, 32]
        e2 = self.enc2(self.pool(e1))   # [B, 128, 16, 16]
        e3 = self.enc3(self.pool(e2))   # [B, 256,  8,  8]
        e4 = self.pool(e3)              # [B, 256,  4,  4]

        # Bottleneck
        h, w = e4.shape[-2:]
        emb_map = emb.view(B, -1, 1, 1).expand(B, -1, h, w)
        bn = self.bottleneck(torch.cat([e4, emb_map], dim=1))  # [B, 512, 4, 4]
        bn = self.attn(bn)

        # Decoder
        d2 = self.dec2(torch.cat([self.up2(bn), e3], dim=1))  # [B, 256, 8, 8]
        d2 = self.attn_dec2(d2)

        d1 = self.dec1(torch.cat([self.up1(d2), e2], dim=1))  # [B, 128, 16, 16]
        d0 = self.dec0(torch.cat([self.up0(d1), e1], dim=1))  # [B, 64,  32, 32]

        return self.out(d0)