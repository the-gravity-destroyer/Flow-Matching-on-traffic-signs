import lightning as pl
import torch
import torchvision
from torch import Tensor
from torchvision.utils import make_grid

from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.solver import ODESolver

from u_net import UNet
from cpd_prior import CPDPrior


class Flow(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        in_channels: int = 3,
        image_size: int = 32,
        num_classes: int = 43,
        cond_drop_prob: float = 0.1,
        stats_path: str = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.net = UNet(
            in_channels=in_channels,
            out_channels=in_channels,
            num_classes=num_classes,
        )
        self.prior = CPDPrior(stats_path) if stats_path else None

    def on_fit_start(self):
        if self.prior is not None:
            self.prior.to(self.device)

    def forward(self, x: Tensor, t: Tensor, y: Tensor = None, **kwargs) -> Tensor:
        return self.net(x, t, y)

    def training_step(self, batch, batch_idx):
        x1, y = batch

        if self.prior is not None:
            x0 = self.prior.sample(y)
        else:
            x0 = torch.randn_like(x1)

        t = torch.rand(x1.size(0), device=self.device)

        y_in = y.clone()
        if self.hparams.cond_drop_prob > 0:
            drop = torch.rand(y.size(0), device=self.device) < self.hparams.cond_drop_prob
            y_in[drop] = self.net.null_token

        sample = self.path.sample(t=t, x_0=x0, x_1=x1)
        pred = self(sample.x_t, sample.t, y_in)
        loss = (pred - sample.dx_t).pow(2).mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def sample(self, y=None, n: int = 64, filename: str = "samples.png"):
        solver = ODESolver(velocity_model=self)

        # CPD-Prior beim Sampling nutzen wenn vorhanden UND y gegeben
        if self.prior is not None and y is not None:
            self.prior.to(self.device)
            x0 = self.prior.sample(y)
        else:
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
            }
        }