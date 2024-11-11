import os
import yaml
from dataclasses import dataclass
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms as T
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from dataset import collate_fn
from trainner import WireModel
from utils import save_model


@dataclass
class Config:
    batch_size: int
    resize_x: int
    resize_y: int
    learning_rate: float
    train_set_split: float
    epoch: int
    encoder_type: str
    transforms_type: str
    use_patches: bool
    model_name: str
    in_channels: int
    out_classes: int
    data_path: str


def split_datasets(config: Config, dataset):
    train_size = int(config.train_set_split * len(dataset))
    test_and_val_size = len(dataset) - train_size
    test_size = test_and_val_size // 2
    val_size = test_and_val_size - test_size

    train_dataset, _ = random_split(dataset, [train_size, test_and_val_size])
    test_dataset, val_dataset = random_split(_, [test_size, val_size])

    return train_dataset, val_dataset, test_dataset


def get_dataloaders(config: Config, dataset):
    train_dataset, val_dataset, test_dataset = split_datasets(config, dataset)

    collate_fn_func = collate_fn if config.use_patches else None
    n_cpu = round(os.cpu_count() * 0.7)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn_func,
        num_workers=n_cpu,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn_func,
        num_workers=n_cpu,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn_func,
        num_workers=n_cpu,
    )

    return train_loader, val_loader, test_loader


def train_model(config: Config, train_loader, val_loader, test_loader):
    model = WireModel(
        config.model_name,
        config.encoder_type,
        in_channels=config.in_channels,
        out_classes=config.out_classes,
        tmax=config.epoch * len(train_loader),
    )
    trainer = pl.Trainer(max_epochs=config.epoch, log_every_n_steps=1)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    valid_metrics = trainer.validate(model, dataloaders=test_loader, verbose=False)
    print(valid_metrics)
    model.save_metrics()
    save_model(model.model)


def build_transforms(config: Config):

    if config.transforms_type == "full-image-resize":
        return A.Compose(
            [
                A.Resize(256, 256),
                A.Lambda(image=build_encoder(config)),
                ToTensorV2(),
                A.Lambda(image=lambda x, **kwargs: x.float()),
            ]
        )
    elif config.transforms_type == "patches":
        return A.Compose(
            [
                A.Lambda(image=build_encoder(config)),
                ToTensorV2(),
                A.Lambda(image=lambda x, **kwargs: x.float()),
            ]
        )
    else:
        raise ValueError(f"Unknown transforms_type: {transf_type}")


def build_encoder(config: Config):
    if config.encoder_type:
        return get_preprocessing_fn(config.encoder_type, pretrained="imagenet")
    return None


def load_config(yaml_file):
    with open(yaml_file, "r") as file:
        config = yaml.safe_load(file)
    return Config(**config)
