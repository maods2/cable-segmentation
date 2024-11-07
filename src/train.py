from segmentation_models_pytorch.encoders import get_preprocessing_fn
import segmentation_models_pytorch as smp

import os

import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from dataset import ImageMaskDataset, collate_fn
import gc
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms as T
from trainner import WireModel
from utils import save_model
torch.set_float32_matmul_precision('medium')

preprocess_input = get_preprocessing_fn('resnet34', pretrained='imagenet')


# init train, val, test sets
path = './data_original_size'

def tensor_to_float(x, **kwargs):
    return x.float()

transforms = A.Compose([
    A.Resize(256, 256),
    A.Lambda(image=preprocess_input),
    ToTensorV2(),
    A.Lambda(image=tensor_to_float),
])

dataset = ImageMaskDataset(image_dir=path, transform=transforms)


train_size = int(0.8 * len(dataset))  
test_size = len(dataset) - train_size  
print(f"Train size: {train_size}")
print(f"Test size: {test_size}")

# Divida o dataset entre treino e teste
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Configure os DataLoaders 
n_cpu =  round(os.cpu_count() * 0.7)

batch_size=2

train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=n_cpu)
test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=n_cpu)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=n_cpu)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=n_cpu)


n_cpu = os.cpu_count()

EPOCHS = 10
T_MAX = EPOCHS * len(train_loader)
T_MAX = 1
OUT_CLASSES = 1


model = WireModel("Unet", "resnet34", in_channels=3, out_classes=1, tmax=T_MAX)
trainer = pl.Trainer(max_epochs=EPOCHS, log_every_n_steps=1)
trainer.fit(
    model, 
    train_dataloaders=train_loader,
    val_dataloaders=test_loader,
)

valid_metrics = trainer.validate(model, dataloaders=test_loader, verbose=False)
print(valid_metrics)

smp_model = model.model


save_model(smp_model)