import os
import sys, argparse
from flow_module import Flow

# Workaround für Hydra + Python 3.14: deaktiviert die neue argparse-Help-Validierung,
# an der Hydras LazyCompletionHelp scheitert. Sicher, weil dessen __repr__()
# einen sauberen String ohne %-Formatierung liefert.
if sys.version_info >= (3, 14):
    argparse.ArgumentParser._check_help = lambda self, action: None
import hydra
import lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.model)
    trainer = instantiate(cfg.trainer)

    trainer.fit(model, datamodule=datamodule)

    # Nach dem Training: ein Grid pro Dataset, nicht überschreibbar
    if trainer.is_global_zero:
        out_dir = hydra.utils.get_original_cwd()
        fname = os.path.join(out_dir, f"samples_{cfg.dataset_name}.png")

        # Bestes Checkpoint laden
        best_ckpt = trainer.checkpoint_callback.best_model_path
        model = Flow.load_from_checkpoint(best_ckpt)
        model.eval()
        model.to("cuda" if torch.cuda.is_available() else "cpu")

        y = torch.randint(0, model.hparams.num_classes, (64,), device=model.device)
        model.sample(y=y, n=64, filename=fname)
        print(f"Saved grid to {fname}")

if __name__ == "__main__":
    main()