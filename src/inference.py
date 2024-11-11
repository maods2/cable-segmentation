from segmentation_models_pytorch.encoders import get_preprocessing_fn
import segmentation_models_pytorch as smp
from PIL import Image
import albumentations as A
from utils import load_checkpoint, merge_image
import torch
import cv2
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np

# Set the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: ", device)

# COMMON FUNCTIONS
def load_preprocessing_fn(encoder_name="resnet18", pretrained="imagenet"):
    """Load preprocessing function for the specified encoder."""
    return get_preprocessing_fn(encoder_name, pretrained=pretrained)

def initialize_model(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1):
    """Initialize the segmentation model."""
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
    return model.to(device)

def load_model_checkpoint(checkpoint_path, model):
    """Load model weights from a checkpoint."""
    load_checkpoint(checkpoint_path, model)
    return model

# FULL IMAGE PREDCTION FUNCTIONS
def preprocess_image(image_path, preprocess_fn):
    """Load and preprocess an image for the model."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    pad_transform = A.Compose(
        [
            A.PadIfNeeded(min_height=2176, min_width=3840),
            A.Resize(height=2176 // 2, width=3840 // 2),
            A.Lambda(image=preprocess_fn),
            ToTensorV2(),
        ]
    )
    return pad_transform(image=image)["image"].unsqueeze(0).float().to(device)

def predict_mask(model, preprocessed_image):
    """Generate a binary mask from the model's prediction."""
    model.eval()
    with torch.no_grad():
        predicted_mask = model(preprocessed_image)
        predicted_mask = torch.sigmoid(predicted_mask)
    return (predicted_mask.squeeze().cpu().numpy() > 0.5).astype("uint8") * 255

def save_mask(binary_mask, output_path):
    """Save the binary mask as an image."""
    cv2.imwrite(output_path, binary_mask)
    print(f"Binary mask saved at: {output_path}")

def segment_image(image_path, checkpoint_path, output_path="predicted_mask.png"):
    """Main function to execute the segmentation pipeline."""
    preprocess_fn = load_preprocessing_fn()
    model = initialize_model()
    model = load_model_checkpoint(checkpoint_path, model)
    
    preprocessed_image = preprocess_image(image_path, preprocess_fn)
    binary_mask = predict_mask(model, preprocessed_image)
    
    save_mask(binary_mask, output_path)
    merge_image(cv2.imread(image_path), binary_mask)

# PATCH PRECTION FUNCTIONS
def preprocess_patch(patch, preprocess_fn):
    patch_transform = A.Compose(
        [
            A.Lambda(image=preprocess_fn),
            ToTensorV2(),
        ]
    )
    return patch_transform(image=patch)["image"].unsqueeze(0).float().to(device)

def split_image_into_patches(image, patch_size=512, stride=256):
    patches = []
    h, w = image.shape[:2]
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y+patch_size, x:x+patch_size]
            patches.append((patch, x, y))
    return patches

def predict_patch_mask(model, preprocessed_patch):
    model.eval()
    with torch.no_grad():
        predicted_mask = model(preprocessed_patch)
        predicted_mask = torch.sigmoid(predicted_mask)
    return (predicted_mask.squeeze().cpu().numpy() > 0.5).astype("uint8") * 255

def reconstruct_full_mask(patches_masks, image_shape, patch_size=512, stride=256):
    full_mask = np.zeros(image_shape[:2], dtype="uint8")
    for mask, x, y in patches_masks:
        full_mask[y:y+patch_size, x:x+patch_size] = np.maximum(full_mask[y:y+patch_size, x:x+patch_size], mask)
    return full_mask

def segment_patches_image(image_path, checkpoint_path, output_path="predicted_full_mask.png"):
    preprocess_fn = load_preprocessing_fn()
    model = initialize_model()
    model = load_model_checkpoint(checkpoint_path, model)
    
    # Load and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Split image into patches
    patches = split_image_into_patches(image)
    
    # Predict mask for each patch
    patches_masks = []
    for patch, x, y in patches:
        preprocessed_patch = preprocess_patch(patch, preprocess_fn)
        patch_mask = predict_patch_mask(model, preprocessed_patch)
        patches_masks.append((patch_mask, x, y))
    
    # Reconstruct the full mask from patch masks
    full_mask = reconstruct_full_mask(patches_masks, image.shape)
    
    # Save and optionally merge the result
    cv2.imwrite(output_path, full_mask)
    print(f"Full binary mask saved at: {output_path}")
    merge_image(image, full_mask)

if __name__ == "__main__":
    # Execute the pipeline
    segment_image(
        image_path="/workspaces/cable-segmentation/data_original_size/1_00186.jpg",
        checkpoint_path="/workspaces/cable-segmentation/checkpoints/cable_seg_model_20241111_005256.pth"
    )
    
    segment_patches_image(
    image_path="/workspaces/cable-segmentation/data_original_size/1_00186.jpg",
    checkpoint_path="/workspaces/cable-segmentation/checkpoints/cable_seg_model_20241111_005256.pth"
    )
