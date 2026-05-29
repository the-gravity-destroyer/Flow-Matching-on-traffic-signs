import torch
from torch import Tensor


class CPDPrior:
    def __init__(self, stats_path: str, device: torch.device = None):
        stats = torch.load(stats_path, map_location=device or "cpu")
        self.means = stats["means"]   # [num_classes, 3, 32, 32]
        self.stds  = stats["stds"]    # [num_classes, 3, 32, 32]

    def to(self, device):
        self.means = self.means.to(device)
        self.stds  = self.stds.to(device)
        return self

    def sample(self, y: Tensor) -> Tensor:
        """
        y: [B] Klassenindizes
        gibt x0: [B, 3, 32, 32] zurück — klassenspezifisches Rauschen
        """
        mu  = self.means[y]                      # [B, 3, 32, 32]
        sig = self.stds[y]                       # [B, 3, 32, 32]
        eps = torch.randn_like(mu)               # zufällige Variation
        return mu + sig * eps                    # N(µ_c, σ_c²·I)