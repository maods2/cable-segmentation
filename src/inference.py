from segmentation_models_pytorch.encoders import get_preprocessing_fn
import segmentation_models_pytorch as smp
from PIL import Image
import albumentations as A
from utils import load_checkpoint, merge_image
import torch
import cv2
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np



# Set the device to GPU if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: ", device)

# Load the preprocessing function for ResNet18
preprocess_input = get_preprocessing_fn("resnet18", pretrained="imagenet")

# Define the segmentation model
model = smp.Unet(
    encoder_name="resnet34",        # choose encoder (e.g., mobilenet_v2 or efficientnet-b7)
    encoder_weights="imagenet",     # use pre-trained weights from ImageNet
    in_channels=3,                  # input channels (1 for grayscale, 3 for RGB, etc.)
    classes=1,                      # output channels (number of classes in your dataset)
)

# Load the model checkpoint
check_point = "checkpoints/cable_seg_model_20241110_182300.pth"
load_checkpoint(check_point, model)

# Load the image and convert from BGR to RGB
path = "/workspaces/cable-segmentation/data_original_size/1_00186.jpg"
image = cv2.imread(path, cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Define padding and preprocessing transformations
pad_transform = A.Compose(
    [
        A.PadIfNeeded(min_height=2176, min_width=3840),   # Pad to required dimensions
        A.Resize(height=2176 // 2, width=3840 // 2),      # Resize the image because My personal GPU has only 8GB of RAM
        A.Lambda(image=preprocess_input),                 # Apply preprocessing for the model
        ToTensorV2(),                                     # Convert to tensor with (H, W, C) format
    ]
)

# Apply transformations and add batch dimension
padded_image = pad_transform(image=image)["image"].unsqueeze(0).float()

# Set the model to evaluation mode and move to the appropriate device
model.eval()
model.to(device)
with torch.no_grad():
    predicted_mask = model(padded_image.to(device))
    predicted_mask = torch.sigmoid(predicted_mask)  # Apply sigmoid to get values between 0 and 1

# Convert to numpy and apply threshold to create a binary mask
binary_mask = (predicted_mask.squeeze().cpu().numpy() > 0.5).astype("uint8") * 255  # Threshold at 0.5 for binarization

# Save the binary mask as an image
output_path = "predicted_mask.png"
cv2.imwrite(output_path, binary_mask)
merge_image(image, binary_mask)

print(f"Binary mask saved at: {output_path}")



