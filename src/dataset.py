import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from pathlib import Path
from tqdm import tqdm



# https://stackoverflow.com/questions/78104467/how-to-load-a-batch-of-images-of-and-split-them-into-patches-on-the-fly-with-pyt
def make_patches(
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

def collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    valid_images = [] # List to store valid image patches
    valid_masks = []  # List to store valid mask patches

    # Iterating over each item in the batch
    for i, b in enumerate(batch):
        image, mask = b['image'], b['mask']


        # Assuming `make_patches` splits the image and mask into patches
        patches = make_patches(image, 256, 256)
        mask_patches = make_patches(mask, 256, 256)

        # Iterating over the mask patches
        for j, mask_patch in enumerate(mask_patches):
            # Check if the mask patch contains any non-zero value
            if mask_patch.max() != 0:
                valid_images.append(patches[j])  # Add valid image patch
                valid_masks.append(mask_patch)   # Add valid mask patch

    # Stack the valid patches into tensors
    filtered_images = torch.stack(valid_images)
    filtered_masks = torch.stack(valid_masks)

    # Return the dictionary with valid image and mask patches
    return {"image": filtered_images, "mask": filtered_masks}

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
            mask = augmented['mask'].unsqueeze(0)
            mask = mask / 255.0
            
            # # print(f'Augmented image shape: {augmented["image"].shape}')
            # print(f'Augmented mask shape: {augmented["mask"].shape}')
            # # print(mask.unique())
            # mask_np = np.squeeze(mask.numpy())
            # import matplotlib.pyplot as plt
            # print(f'Augmented numpy shape: {mask_np.shape}')
            # plt.imsave('mask_binaria.png', mask_np, cmap='gray', format='png')
            # raise
            
        return {"image": image,  # Normalize and change order to (C, H, W)
                "mask": mask  } # Adds channel dimension


if __name__ == '__main__':
    # testing 
    path = './data_original_size'
    dataset = ImageMaskDataset(image_dir=path)
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    sub_batch_size=32
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {0+1}/{10}', leave=True)
    for batch in progress_bar:
        images_patches = batch['image'].to('cuda')
        masks_patches = batch['mask'].to('cuda')
        
        num_patches = images_patches.size(0)
        for i in range(0, num_patches, sub_batch_size):
            # Selecionar o sub-batch
            img_sub_batch = images_patches[i:i + sub_batch_size]
            mask_sub_batch = masks_patches[i:i + sub_batch_size]