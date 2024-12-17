# import cv2
# from pathlib import Path
# import albumentations as A
# from albumentations.pytorch.transforms import ToTensorV2
# import sys

# from inference import load_model_checkpoint
# from utils import save_model
# sys.path.append('/workspaces/cable-segmentation/src')

# from transformers import SamModel, SamProcessor
# import torch


# import numpy as np

# import matplotlib.pyplot as plt


# device = "cuda" if torch.cuda.is_available() else "cpu"
# checkpoint_path = "./checkpoints/cable_seg_model_sam.pth"
# processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
# model = SamModel.from_pretrained("facebook/sam-vit-base").to(config["device"])
# model = load_model_checkpoint(checkpoint_path, model)

# transform = A.Compose(
#         [
#             # A.Resize(256, 256),
#             A.Resize(1024, 1024),
#             ToTensorV2(),
#             A.Lambda(image=lambda x, **kwargs: x.float()),
#         ]
#     )

# img_path = "./assets/tree"

# imgs = [file.as_posix() for file in Path(img_path).glob('*.jpg') if 'segmented' not in Path(file).name]
# # print(imgs)
# for img in imgs:
    
#     image = cv2.imread(img, cv2.IMREAD_COLOR)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    
#     single_patch = transform(image=image)["image"]
#     inputs = processor(single_patch.to(device), return_tensors="pt")        
#     inputs = {k: v.to(device) for k, v in inputs.items()}
#     model.eval()

#     # forward pass
#     with torch.no_grad():
#         outputs = model(**inputs, multimask_output=False)
    
#     single_patch_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
#     # convert soft mask to hard mask
#     single_patch_prob = single_patch_prob.cpu().numpy().squeeze()
#     single_patch_prediction = (single_patch_prob > 0.5).astype(np.uint8)
    
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))

#     # Plot the first image on the left
#     axes[0].imshow(np.array(image), cmap='gray')  # Assuming the first image is grayscale
#     axes[0].set_title("Image")

#     # Plot the second image on the right
#     axes[1].imshow(single_patch_prob)  # Assuming the second image is grayscale
#     axes[1].set_title("Probability Map")

#     # Plot the second image on the right
#     axes[2].imshow(single_patch_prediction, cmap='gray')  # Assuming the second image is grayscale
#     axes[2].set_title("Prediction")
    
#     plt.savefig('filename.png')
#     break
    
    

import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from transformers import SamProcessor, SamModel
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


from inference import reconstruct_full_mask, split_image_into_patches
from utils import load_checkpoint, merge_image


device = "cuda" if torch.cuda.is_available() else "cpu"


# ===============================
# Image Segmentation Functions
# ===============================

def sam_segment_image(image_path, checkpoint_path, output_path="predicted_mask_sam.png"):
    """Segment a full image using SAM."""
    image = load_image(image_path)
    processor, model = load_sam_model_and_processor(checkpoint_path, device)
    
    # Apply the transformation and preprocess the image
    transformed_image = transform(image=image)["image"]
    inputs = processor(transformed_image.to(device), return_tensors="pt")
    
    # Perform inference and get the predicted mask
    mask = predict_mask(model, inputs)
    
    # Save the predicted mask
    save_predicted_mask(mask, output_path)


def sam_segment_patches(image_path, checkpoint_path, output_path="predicted_full_mask_sam.png", patch_size=512, stride=256):
    """Segment an image by processing it in patches using SAM."""
    image = load_image(image_path)
    processor, model = load_sam_model_and_processor(checkpoint_path, device)
    
    # Split the image into patches
    patches = split_image_into_patches(image, patch_size, stride)
    
    patches_masks = []
    
    for patch, x, y in patches:
        transformed_patch = transform(image=patch)["image"]
        inputs = processor(transformed_patch.to(device), return_tensors="pt")
        
        # Perform inference for each patch
        patch_mask = predict_mask(model, inputs)
        patch_mask = cv2.resize(patch_mask, (patch_size, patch_size), interpolation=cv2.INTER_NEAREST)
        patches_masks.append((patch_mask, x, y))
    
    # Reconstruct the full mask
    full_mask = reconstruct_full_mask(patches_masks, image.shape, patch_size=patch_size, stride=stride)
    
    # Save and optionally merge the result

    cv2.imwrite(output_path, full_mask)
    print(f"Full binary mask saved at: {output_path}")
    merge_image(image, full_mask, output_path)


# ===============================
# Helper Functions
# ===============================

def load_image(image_path):
    """Load an image from file."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)




def predict_mask(model, inputs):
    """Run inference and predict mask from model."""
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    
    mask_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
    mask_prob = mask_prob.cpu().numpy().squeeze()
    return (mask_prob > 0.5).astype(np.uint8) * 255



def save_predicted_mask(mask, output_path):
    """Save the predicted binary mask to a file."""
    cv2.imwrite(output_path, mask * 255)
    print(f"Predicted mask saved at: {output_path}")


# ===============================
# Model and Processor Loading Functions
# ===============================

def load_sam_model_and_processor(checkpoint_path: str, device: str = "cuda"):
    """
    Load the SAM model and processor from the pretrained configuration and checkpoint.
    
    Args:
        checkpoint_path (str): Path to the checkpoint for loading model weights.
        device (str): Device to load the model on (default is 'cuda').
        
    Returns:
        processor (SamProcessor): The SAM processor.
        model (SamModel): The SAM model.
    """
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    # load_checkpoint(checkpoint_path, model)
    return processor, model


def load_image_transform():
    """
    Define and return the image transformation pipeline for segmentation.
    
    Returns:
        transform (A.Compose): The albumentations transformation pipeline.
    """
    return A.Compose(
        [
            A.Resize(1024, 1024),
            ToTensorV2(),
            A.Lambda(image=lambda x, **kwargs: x.float()),
        ]
    )





# ===============================
# Initialize Processor, Model, and Transformation
# ===============================
if __name__ == "__main__":
    # Load SAM model and processor
    checkpoint_path = "./checkpoints/cable_seg_model_sam_no_inputs.pth"
    

    # Define the transformation
    transform = load_image_transform()

    img_path = "./assets/tree"
    out_path = "./assets/output"
    out_path = str(Path(out_path).resolve())
    checkpoint_path = "./checkpoints/cable_seg_model_sam_no_inputs.pth"
    

    # Define the transformation
    # transform = load_image_transform()

    # img_path = "./assets/tree"
    # out_path = str(Path(out_path).resolve())

    # sam_segment_patches(img_path, checkpoint_path, output_path="predicted_full_mask_sam.png")


    imgs = [file.as_posix() for file in Path(img_path).glob('*.jpg') if 'segmented' not in Path(file).name]

    for img in imgs:

        # # Segment the full image
        sam_segment_image(
            img, 
            checkpoint_path, 
            img.replace(".jpg","_segmented.jpg").replace("tree","sam_output")     
            )

        # Segment an image by processing it in patches
        # sam_segment_patches(
        #     img, 
        #     checkpoint_path, 
        #     img.replace(".jpg","_patch_segmented.jpg"),   
        #     patch_size=1024, 
        #     stride=512
        #     )