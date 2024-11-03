import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from pathlib import Path


# https://stackoverflow.com/questions/78104467/how-to-load-a-batch-of-images-of-and-split-them-into-patches-on-the-fly-with-pyt
def make_paches(
    img : torch.Tensor,
    patch_width : int,
    patch_height : int
) -> List[torch.Tensor]:

    patches = img \
        .unfold(1,patch_width,patch_width) \
        .unfold(2,patch_height,patch_height) \
        .flatten(1,2) \
        .permute(1,0,2,3)

    patches = list(patches)
    return patches

def collate_fn(batch : List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    
    new_x = []
    new_mask = []
    
    for b in batch:
        
        patches = make_paches(b['image'], 224, 224)
        new_x.extend(patches)
        mask_patches = make_paches(b['mask'], 224, 224)
        new_mask.extend(mask_patches)

    new_x = torch.stack(new_x)
    new_mask = torch.stack(new_mask)
    
    return {"image": new_x, "mask": new_mask}

class ImageMaskDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        """
        Builds the dataset to load images and masks.

        Args:
            image_dir (str): Directory containing the images.
            transform (callable, optional): Transformation function to be applied to images and masks.
        """
        self.image_dir = image_dir
        self.transform = transform
        self.images = [Path(f).name  for f in Path(image_dir).glob('*.jpg') if 'mask' not in Path(f).name]

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """Loads an image and its corresponding mask."""
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.image_dir, f'{self.images[idx].split(".")[0]}_mask.jpg')
        
        # Load the image and the mask
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformations if provided
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return {"image": torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0,  # Normalize and change order to (C, H, W)
                "mask": torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0}  # Adds channel dimension



if __name__ == '__main__':
    # testing 
    path = './data_original_size'
    dataset = ImageMaskDataset(image_dir=path)
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    for batch in dataloader:
        images_patches = batch['image']
        masks_patches = batch['mask']
        
        num_patches = patches.size(0)
        for i in range(0, num_patches, sub_batch_size):
            # Selecionar o sub-batch
            sub_batch = patches[i:i + sub_batch_size]