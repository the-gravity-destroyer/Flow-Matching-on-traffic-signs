import lightning as pl
import torch
import torchvision
from torch import Tensor
from torchvision.utils import make_grid

from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver

from u_net import UNet


class Flow(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        in_channels: int = 3,
        image_size: int = 32,
        num_classes: int = 43,
        cond_drop_prob: float = 0.1,   # für CFG; 0.0 = reines Conditioning
    ):
        super().__init__()
        self.save_hyperparameters()
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.net = UNet(
            in_channels=in_channels,
            out_channels=in_channels,
            num_classes=num_classes,
        )

    def forward(self, x: Tensor, t: Tensor, y: Tensor = None, **kwargs) -> Tensor:
        return self.net(x, t, y)

    def training_step(self, batch, batch_idx):
        x1, y = batch                         
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.size(0), device=self.device)
        sample = self.path.sample(t=t, x_0=x0, x_1=x1)

        # Conditioning-Dropout: manche Samples auf Null-Token setzen (CFG-Training)
        if self.hparams.cond_drop_prob > 0:
            drop = torch.rand(y.size(0), device=self.device) < self.hparams.cond_drop_prob
            y = y.clone()
            y[drop] = self.net.null_token

        pred = self(sample.x_t, sample.t, y)
        loss = (pred - sample.dx_t).pow(2).mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def sample(self, y=None, n: int = 64, filename: str = "samples.png"):
        solver = ODESolver(velocity_model=self)
        x0 = torch.randn(
            n, self.hparams.in_channels,
            self.hparams.image_size, self.hparams.image_size,
            device=self.device,
        )
        extras = {} if y is None else {"y": y}
        x1 = solver.sample(x_init=x0, method="midpoint", step_size=1 / 100, **extras)
        grid = make_grid(x1.clamp(0, 1), nrow=8, padding=2)
        torchvision.utils.save_image(grid, filename)
        return x1

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)