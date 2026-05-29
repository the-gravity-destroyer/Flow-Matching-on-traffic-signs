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
        t_emb_dim: int = 128,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Zeit-Embedding
        self.t_emb = nn.Sequential(
            nn.Linear(1, t_emb_dim), nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim)
        )
        # Klassen-Embedding
        self.y_emb = nn.Embedding(num_classes + 1, t_emb_dim)
        self.null_token = num_classes

        # Encoder
        self.enc1 = self._block(in_channels, 32)
        self.enc2 = self._block(32, 64)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck + Attention
        self.bottleneck = self._block(64 + t_emb_dim, 128)
        self.attn = SelfAttention(128, num_heads=4)   # neu

        # Decoder
        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self._block(128, 64)
        self.up0  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec0 = self._block(64, 32)

        self.out  = nn.Conv2d(32, out_channels, 1)

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

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.pool(e2)

        h, w = e3.shape[-2:]
        emb_map = emb.view(B, -1, 1, 1).expand(B, -1, h, w)
        bn = self.bottleneck(torch.cat([e3, emb_map], dim=1))
        bn = self.attn(bn)                             # neu

        d1 = self.dec1(torch.cat([self.up1(bn), e2], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), e1], dim=1))
        return self.out(d0)