import torch
from pathlib import Path
from tqdm import tqdm
import sys, argparse

if sys.version_info >= (3, 14):
    argparse.ArgumentParser._check_help = lambda self, action: None
from hydra import compose, initialize
from hydra.utils import instantiate


def compute_stats(dataset_name: str, num_classes: int, out_path: str):
    with initialize(version_base=None, config_path="conf"):
        cfg = compose(config_name="config", overrides=[f"data={dataset_name}"])

    dm = instantiate(cfg.data)
    dm.prepare_data()
    dm.setup(stage="fit")

    loader = dm.train_dataloader()
    buckets = [[] for _ in range(num_classes)]

    print("Sammle Bilder pro Klasse...")
    for x, y in tqdm(loader):
        for img, label in zip(x, y):
            buckets[label.item()].append(img)   # [3, 32, 32]

    means, stds = [], []
    print("Berechne Statistiken...")
    for c, imgs in enumerate(buckets):
        stack = torch.stack(imgs)               # [N_c, 3, 32, 32]
        means.append(stack.mean(dim=0))         # [3, 32, 32]
        stds.append(stack.std(dim=0) + 1e-5)   # [3, 32, 32], +eps gegen 0
        print(f"  Klasse {c:3d}: {len(imgs)} Bilder")

    means = torch.stack(means)   # [num_classes, 3, 32, 32]
    stds  = torch.stack(stds)    # [num_classes, 3, 32, 32]

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"means": means, "stds": stds}, out_path)
    print(f"Gespeichert: {out_path}")


if __name__ == "__main__":
    compute_stats("gtsrb", num_classes=43, out_path="class_stats/gtsrb.pt")
    #compute_stats("lisa",  num_classes=47, out_path="class_stats/lisa.pt")