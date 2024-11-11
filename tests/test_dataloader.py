import sys
sys.path.append('/workspaces/cable-segmentation/src')

from dataset import ImageMaskDataset


from torch.utils.data import IterableDataset
from torch.utils.data import Dataset, DataLoader
import torch
from typing import List, Tuple
import numpy as np


def make_patches(img: torch.Tensor, patch_width: int, patch_height: int) -> List[torch.Tensor]:
    patches = (
        img.unfold(1, patch_width, patch_width)
        .unfold(2, patch_height, patch_height)
        .flatten(1, 2)
        .permute(1, 0, 2, 3)
    )
    return list(patches)

class PatchIterableDataset(IterableDataset):
    def __init__(self, dataset, patch_width=256, patch_height=256):
        self.dataset = dataset
        self.patch_width = patch_width
        self.patch_height = patch_height
    
    def __iter__(self):
        for data in self.dataset:
            image = torch.from_numpy(np.permute(data["image"], (2, 0, 1)))
            mask = torch.from_numpy(np.expand_dims(data["mask"], axis=-1))
            print(f"Image shape {image.shape}")
            print(f"Mask shape {mask.shape}")
            image_patches = make_patches(image, self.patch_width, self.patch_height)
            mask_patches = make_patches(mask, self.patch_width, self.patch_height)
            for img_patch, mask_patch in zip(image_patches, mask_patches):
                if mask_patch.max() != 0:
                    yield {"image": img_patch, "mask": mask_patch}


if __name__ == "__main__":
    # testing
    path = "./data_original_size"
    dataset = ImageMaskDataset(image_dir=path)
    
    patch_iterable_dataset = PatchIterableDataset(dataset)

    train_loader = DataLoader(
        patch_iterable_dataset,
        batch_size=32,  # Número fixo de patches por batch
        shuffle=False,  # Shuffle pode ser complexo em IterableDataset
        num_workers=7,
    )

    for batch in train_loader:
        images_patches = batch["image"].to("cuda")
        masks_patches = batch["mask"].to("cuda")
        print(images_patches.shape)
        break
