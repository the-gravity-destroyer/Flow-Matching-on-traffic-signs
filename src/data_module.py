import torch
import lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import os
import torch
import lightning as pl
from torchvision import transforms

class BaseDataModule(pl.LightningDataModule):    
    def __init__(self, data_dir, batch_size, num_workers, variant, num_classes):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        # Wir weisen die Transformation erst hier zu, damit der variant-String genutzt werden kann
        self.transform = self._get_transforms(variant)


    def _get_transforms(self, variant):
        base_transforms = [transforms.Resize((32, 32)), transforms.ToTensor()]
        return transforms.Compose(base_transforms)
            


    def _collate_onehot(self, batch):
        images = torch.stack([item[0] for item in batch], dim=0)
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).float()
        return images, labels_onehot

    def _get_dataloader(self, dataset, shuffle=False):
        use_collate = self.hparams.variant == "variatonal_autoencoder" 
        return DataLoader(
            dataset, 
            batch_size=self.hparams.batch_size, 
            shuffle=shuffle, 
            num_workers=self.hparams.num_workers,
            pin_memory=True,                    
            persistent_workers=self.hparams.num_workers > 0, 
            collate_fn=self._collate_onehot if use_collate else None
        )

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, shuffle=False)
        
    def predict_dataloader(self):
        return self.test_dataloader()

class GTSRBDataModule(BaseDataModule):
    def __init__(self, data_dir="/home/leonis/project_datasets", batch_size=64, num_workers=4, variant="classifier"):
        super().__init__(data_dir, batch_size, num_workers, variant, num_classes=43)

    def prepare_data(self):
        datasets.GTSRB(root=self.hparams.data_dir, split="train", download=True)
        datasets.GTSRB(root=self.hparams.data_dir, split="test", download=True)

    def setup(self, stage=None):
        if stage in (None, "fit", "validate"):
            full_data = datasets.GTSRB(root=self.hparams.data_dir, split="train", transform=self.transform)
            train_len = int(len(full_data) * 0.9)
            val_len = len(full_data) - train_len
            self.train_dataset, self.val_dataset = random_split(
                full_data, [train_len, val_len], generator=torch.Generator().manual_seed(42)
            )
        if stage in (None, "test", "predict"):
            self.test_dataset = datasets.GTSRB(root=self.hparams.data_dir, split="test", transform=self.transform)


class LISADataModule(BaseDataModule):
    def __init__(self, data_dir="/home/leonis/project_datasets", batch_size=64, num_workers=4, variant="classifier"):
        super().__init__(data_dir, batch_size, num_workers, variant, num_classes=47)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        lisa_path = os.path.join(self.hparams.data_dir, "LISA")
        
        if stage in (None, "fit", "validate"):
            full_data = datasets.ImageFolder(root=os.path.join(lisa_path, "train"), transform=self.transform)
            train_len = int(len(full_data) * 0.9)
            val_len = len(full_data) - train_len
            self.train_dataset, self.val_dataset = random_split(
                full_data, [train_len, val_len], generator=torch.Generator().manual_seed(42)
            )
            
        if stage in (None, "test", "predict"):
            self.test_dataset = datasets.ImageFolder(root=os.path.join(lisa_path, "test"), transform=self.transform)


class DataModuleFactory:
    _REGISTRY = {
        "gtsrb": GTSRBDataModule,
        "lisa": LISADataModule 
    }

    @staticmethod
    def create(dataset_name, variant, data_dir="/home/leonis/project_datasets", batch_size=64, num_workers=4):
        dataset_name = dataset_name.lower()

        if dataset_name not in DataModuleFactory._REGISTRY:
            raise ValueError(f"Dataset '{dataset_name}' not found. Available: {list(DataModuleFactory._REGISTRY.keys())}")
            
        DataModuleClass = DataModuleFactory._REGISTRY[dataset_name]
        return DataModuleClass(data_dir=data_dir, batch_size=batch_size, num_workers=num_workers, variant=variant)