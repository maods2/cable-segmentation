import sys
sys.path.append('/workspaces/cable-segmentation/src')

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from dataset import ImageMaskDataset, collate_fn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def plot_tensor_patch(tensor_patch, file_name):
    # Remove batch dimension if present, convert to numpy
    if tensor_patch.dim() == 4:
        tensor_patch = tensor_patch.squeeze(0)
    elif tensor_patch.dim() != 3:
        raise ValueError("Tensor must have 3 or 4 dimensions (C, H, W or B, C, H, W)")

    # Move channel dimension to the last position and convert to numpy
    patch = tensor_patch.permute(1, 2, 0).cpu().numpy()
    
    # If values are in range [0, 1], scale to [0, 255]
    if patch.max() <= 1.0:
        patch = (patch * 255).astype(np.uint8)
        patch = np.squeeze(patch, axis=-1)
    elif patch.max() > 255:
        # Clip values above 255 for safety
        patch = np.clip(patch, 0, 255).astype(np.uint8)
    else:
        patch = patch.astype(np.uint8)


    # Save the patch as an image using PIL
    output_path = f"{file_name}.png"
    Image.fromarray(patch).save(output_path)
    print(f"Patch image saved at: {output_path}")
    
    
path = './data_original_size'
transforms = A.Compose([
    # A.PadIfNeeded(min_height=2176, min_width=3840),
    ToTensorV2(),
])

batch_size=2
dataset = ImageMaskDataset(image_dir=path, transform=transforms)
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
sub_batch_size=32

progress_bar = tqdm(dataloader, desc=f'Epoch {0+1}/{10}', leave=True)
for batch in progress_bar:
    print(batch["image"].shape)
    print(batch["mask"].shape)
    
    plot_tensor_patch(batch["image"][0], 'debug_image')
    plot_tensor_patch(batch["mask"][0], 'debug_mask')
    break
    

