import torch
from torch import nn, Tensor

class UNet(nn.Module):
    def __init__(self, t_emb_dim: int = 128):
        super().__init__()
        # Zeit-Embedding
        self.t_emb = nn.Sequential(
            nn.Linear(1, t_emb_dim), nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim)
        )
        # Encoder
        self.enc1 = self._block(1, 32)
        self.enc2 = self._block(32, 64)
        self.pool = nn.MaxPool2d(2)  # 28→14→7

        # Bottleneck
        self.bottleneck = self._block(64 + t_emb_dim, 128)

        # Decoder
        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)  # 7→14
        self.dec1 = self._block(128, 64)   # 64 skip + 64 up
        self.up0  = nn.ConvTranspose2d(64, 32, 2, stride=2)   # 14→28
        self.dec0 = self._block(64, 32)    # 32 skip + 32 up

        self.out  = nn.Conv2d(32, 1, 1)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c), nn.SiLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c), nn.SiLU()
        )

    def forward(self, x: Tensor, t: Tensor, **kwargs) -> Tensor:
        B = x.shape[0]
        t = t.view(1).expand(B).view(B, 1).float()
        t_emb = self.t_emb(t)  # [B, t_emb_dim]
        e1 = self.enc1(x)             # [B, 32, 28, 28]
        e2 = self.enc2(self.pool(e1)) # [B, 64, 14, 14]
        e3 = self.pool(e2)            # [B, 64,  7,  7]

        t_map = t_emb.view(B, -1, 1, 1).expand(B, -1, 7, 7)
        bn = self.bottleneck(torch.cat([e3, t_map], dim=1))  # [B, 128, 7, 7]

        d1 = self.dec1(torch.cat([self.up1(bn), e2], dim=1)) # [B, 64, 14, 14]
        d0 = self.dec0(torch.cat([self.up0(d1), e1], dim=1)) # [B, 32, 28, 28]

        return self.out(d0)  # [B, 1, 28, 28]