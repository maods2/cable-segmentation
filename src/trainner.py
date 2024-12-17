from datetime import datetime
import os

import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
from dataset import ImageMaskDataset
import gc
import albumentations as A
import numpy as np
from collections import defaultdict

from utils import plot_loss_curve, save_loss_iou_plot

def model_builder(
    architecture: str = "Unet",
    encoder_name: str = "resnet34", 
    in_channels: int = 3,
    out_classes: int = 1
    ):
    if architecture == 'Unet':
        model = smp.Unet(
            encoder_name=encoder_name, 
            encoder_weights='imagenet', 
            in_channels=in_channels, 
            classes=out_classes)
    elif architecture == 'FPN':
        model = smp.FPN(
            encoder_name=encoder_name, 
            encoder_weights='imagenet', 
            in_channels=in_channels, 
            classes=out_classes)
    elif architecture == 'DeepLabV3':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name, 
            encoder_weights='imagenet', 
            in_channels=in_channels, 
            classes=out_classes)
    else:
        raise ValueError('architecture not supported')
    return model

class WireModel(pl.LightningModule):
    def __init__(
        self,
        architecture,
        encoder_name,
        in_channels,
        out_classes,
        tmax,
        pipeline_name,
        **kwargs,
    ):
        super().__init__()
        self.t_max = tmax
        self.pipeline_name = pipeline_name
        
        
        # self.model = smp.Unet(
        #     encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        #     encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        #     in_channels=in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        #     classes=out_classes,  # model output channels (number of classes in your dataset)
        # )

        self.model = model_builder(
            architecture=architecture,
            encoder_name=encoder_name,
            in_channels=in_channels,
            out_classes=out_classes
        )
        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        # initialize step metics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.metrics = defaultdict(list)

    def save_metrics(self):

        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # model_path = f"./checkpoints/model_metrics_{timestamp}.npz"
        # np.savez(model_path, **self.metrics)
        loss_train = self.metrics['loss_train']
        iou_train = self.metrics['iou_train']
        
        loss_valid = self.metrics['loss_valid']
        iou_valid = self.metrics['iou_valid']
        # plot_loss_curve(self.metrics, f"{self.pipeline_name}_loss_curve.png")
        save_loss_iou_plot(loss_train, iou_train, len(loss_train),f"{self.pipeline_name}_loss_iou_curve.png")

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert (
            h % 32 == 0 and w % 32 == 0
        ), f"image size should be divisible by 32, got {h}x{w}"

        mask = batch["mask"]
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert (
            mask.max() <= 1.0 and mask.min() >= 0
        ), f"mask values should be in between 0 and 1, got max={mask.max()} min={ mask.min()}"

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "_loss": loss.cpu().detach().numpy(),
        }

    def shared_epoch_end(self, outputs, stage):
        losses = [x["_loss"] for x in outputs]
        loss = sum(losses) / len(losses)
        self.metrics[f'loss_{stage}'].append(loss)

        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        
        self.metrics[f'iou_{stage}'].append(dataset_iou)
        

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")

        # append the metics of each step to the
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        # empty set output list
        self.training_step_outputs.clear()
        return

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")

        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()
        return

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")

        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        # empty set output list
        self.test_step_outputs.clear()
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.t_max, eta_min=1e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }



if __name__ == "__main__":

    # init train, val, test sets
    path = "./data_original_size"
    transforms = A.Compose(
        [
            A.PadIfNeeded(min_height=250, min_width=250),
            A.pytorch.transforms.ToTensorV2(),
        ]
    )

    dataset = ImageMaskDataset(image_dir=path, transform=transforms)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    print(f"Train size: {train_size}")
    print(f"Test size: {test_size}")

    # Divida o dataset entre treino e teste
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Configure os DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    n_cpu = os.cpu_count()

    EPOCHS = 10
    T_MAX = EPOCHS * len(train_loader)
    OUT_CLASSES = 1

    progress_bar = tqdm(train_loader, desc=f"Epoch {0+1}/{10}", leave=True)
    for batch in progress_bar:
        images_patches = batch["image"]
        masks_patches = batch["mask"]
        del batch
        del images_patches
        del masks_patches
        gc.collect()
