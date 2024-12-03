import sys
import argparse
from inference import load_model_checkpoint
from utils import save_model
sys.path.append('/workspaces/cable-segmentation/src')

from transformers import SamModel, SamProcessor
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import monai
from dataset import SAMDataset
from statistics import mean
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description="SAM finetune.")
parser.add_argument("--num_epochs", type = int, default = 1)
parser.add_argument("--batch_size", type = int, default = 2)
parser.add_argument("--max_points", type = int, default = 8)

args = parser.parse_args()

# Configuration dictionary
config = {
    "data_path": "./data_original_size",
    "batch_size": args.batch_size,
    "shuffle": True,
    "model_name": "facebook/sam-vit-base",
    "learning_rate": 1e-5,
    "weight_decay": 0,
    "loss_function": monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean'),
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "max_points": args.max_points,
    "num_epochs": args.num_epochs,
}

def save_loss_iou_plot(epoch_loss_scores, epoch_iou_scores, num_epochs, filename="training_loss_iou_curve.png"):
    """
    Function to plot and save the loss and IoU curves.

    Args:
    - epoch_loss_scores (list): List of loss values for each epoch.
    - epoch_iou_scores (list): List of IoU values for each epoch.
    - num_epochs (int): Number of epochs in the training.
    - filename (str): The filename to save the plot image (default is "training_loss_iou_curve.png").
    """
    epochs = np.arange(0, num_epochs + 1)

    # Create a figure with two subplots
    plt.figure(figsize=(12, 6))

    # Plotting Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, epoch_loss_scores, label='Loss', color='blue', marker='o')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Plotting IoU
    plt.subplot(1, 2, 2)
    plt.plot(epochs, epoch_iou_scores, label='IoU', color='green', marker='o')
    plt.title('IoU Curve')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.grid(True)
    plt.legend()

    # Save the plot to a file
    plt.tight_layout()
    plt.savefig(filename)


# Function to initialize dataset and dataloader
def initialize_data(config):
    dataset = SAMDataset(image_dir=config["data_path"])
    dataloader = DataLoader(dataset=dataset, batch_size=config["batch_size"], shuffle=config["shuffle"])
    return dataloader


# Function to initialize model and optimizer
def initialize_model_and_optimizer(config):
    model = SamModel.from_pretrained(config["model_name"]).to(config["device"])
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    
    optimizer = Adam(model.mask_decoder.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    return model, optimizer


# Function to prepare batch points for the processor
def prepare_batch_points(masks, max_points):
    batch_points = []
    for i in range(masks.shape[0]):
        coords = torch.nonzero(masks[i])
        selected_indices = coords[torch.randperm(coords.size(0))[:max_points]]
        points = selected_indices[:, -2:].tolist()
        if len(points)==0:
            # points = [[np.nan, np.nan] for _ in range(max_points)]
            points = [[0, 0] for _ in range(max_points)]
        batch_points.append(points)
    return batch_points


# Function to compute IoU
def compute_iou(pred_masks, true_masks):
    intersection = (pred_masks & true_masks).float().sum((1, 2))
    union = (pred_masks | true_masks).float().sum((1, 2))
    iou = intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero
    return iou.mean().item()


def train_model(config):
    dataloader = initialize_data(config)
    model, optimizer = initialize_model_and_optimizer(config)
    processor = SamProcessor.from_pretrained(config["model_name"])
    loss_function = config["loss_function"]

    model.to(config["device"])
    
    num_epochs = config['num_epochs']
    epoch_iou_scores = []  # To track IoU for each epoch
    epoch_loss_scores = []  # To track loss for each epoch

    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_ious = []  # To track IoU for this epoch

        # Wrapping the dataloader with tqdm to display progress
        with tqdm(dataloader, desc=f"Epoch {epoch + 1}", dynamic_ncols=True) as pbar:
            for i, batch in enumerate(pbar):
                images = batch["image"]
                masks = batch["mask"]

                batch_points = prepare_batch_points(masks, config["max_points"])
                inputs = processor(images, input_points=batch_points, return_tensors="pt")

                outputs = model(
                    pixel_values=inputs["pixel_values"].to(config["device"]),
                    input_points=inputs["input_points"].to(config["device"]),
                    multimask_output=False
                )

                predicted_masks = outputs.pred_masks.squeeze(1)
                loss = loss_function(predicted_masks, masks.to(config["device"]))

                optimizer.zero_grad()
                loss.backward()

                # Optimize
                optimizer.step()
                epoch_losses.append(loss.item())

                # Calculate IoU for the current batch
                pred_masks_bin = (predicted_masks > 0.5).int()  # Convert logits to binary masks
                batch_iou = compute_iou(pred_masks_bin, masks.int().to(config["device"]))
                epoch_ious.append(batch_iou)
                if i == 100:
                    break

                # Update tqdm progress bar with current loss and IoU
                pbar.set_postfix(loss=mean(epoch_losses), iou=mean(epoch_ious))

        # Calculate mean loss and IoU for the epoch
        mean_loss = mean(epoch_losses)
        mean_iou = mean(epoch_ious)

        epoch_iou_scores.append(mean_iou)
        epoch_loss_scores.append(mean_loss)

        print(f'EPOCH {epoch + 1}:')
        print(f'Mean Loss: {mean_loss}')
        print(f'Mean IoU: {mean_iou:.4f}')
        
        save_model(model, "./checkpoints/cable_seg_model_sam.pth")
        
        # save_loss_iou_plot(epoch_losses, epoch_ious, 100)
        save_loss_iou_plot(epoch_loss_scores, epoch_iou_scores, num_epochs)
        
        

if __name__ == "__main__":
    train_model(config)
    
    
    