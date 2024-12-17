from datetime import datetime
import os
from pathlib import Path
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt


def save_model(model, model_path=None):
    
    if not model_path:
        Path(f"./checkpoints/").mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"./checkpoints/cable_seg_model_{timestamp}.pth"
    
    torch.save(model.state_dict(), model_path)


def load_checkpoint(path, model):
    print("=> Loading checkpoint")
    model.load_state_dict(torch.load(path, weights_only=True))


def merge_image(image, mask, output_dir):
    # Resize mask if its shape doesn't match the original image dimensions
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # Create an empty image of the same size as the original image
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 255] = [0, 0, 255]  # Fill the mask area with red (BGR)

    # Apply transparency
    alpha = 0.5  # Transparency level of the mask
    result = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

    # Convert the image from BGR to RGB
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    sufix = "_predicted_mask_merged.jpg"
    output_path = output_dir.replace(".jpg", sufix).replace(".png", sufix)
    cv2.imwrite(output_path, result)


def plot_loss_curve(data, filename):
    """
    Plots the loss curve with training and validation data and saves the image to a file.

    Parameters:
    - data: Dictionary containing arrays for training and validation loss.
    - filename: The name of the file where the plot will be saved.
    """
    # Extracting the data
    train_loss = data["train"]
    val_loss = data["valid"]

    # Generating the number of epochs (assuming both arrays have the same length)
    epochs_t = np.arange(1, len(train_loss) + 1)
    epochs_v = np.arange(1, len(val_loss) + 1)

    # Creating the plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_t, train_loss, label="Train Loss", color="b")
    plt.plot(epochs_v, val_loss, label="Validation Loss", color="r")

    # Adding title and axis labels
    plt.title("Loss Curve", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)

    # Adding the legend
    plt.legend(loc="upper right")

    # Saving the plot to the file
    plt.savefig(filename)
    plt.close()

def save_loss_iou_plot(epoch_loss_scores, epoch_iou_scores, num_epochs, filename="training_loss_iou_curve.png"):
    """
    Function to plot and save the loss and IoU curves.

    Args:
    - epoch_loss_scores (list): List of loss values for each epoch.
    - epoch_iou_scores (list): List of IoU values for each epoch.
    - num_epochs (int): Number of epochs in the training.
    - filename (str): The filename to save the plot image (default is "training_loss_iou_curve.png").
    """
    epochs = np.arange(1, num_epochs + 1)

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