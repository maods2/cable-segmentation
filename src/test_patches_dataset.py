import sys
sys.path.append('/workspaces/cable-segmentation/src')

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from dataset import ImageMaskDataset, collate_fn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

path = './data_original_size'
transforms = A.Compose([
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
    break