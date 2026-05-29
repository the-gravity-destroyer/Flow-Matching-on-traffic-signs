import hydra
import lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)
    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.model)
    trainer = instantiate(cfg.trainer)
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()