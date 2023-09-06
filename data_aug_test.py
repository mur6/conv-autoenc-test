from pathlib import Path
import random

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision

from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T
from torchvision.transforms import v2

import torchvision.transforms.v2 as transforms


sample_image_path = "../poetry-test-proj/samples/02"



class MaskDataset(Dataset):
    def __init__(self, transform=None):
        # self.images_filepaths = list(Path(images_folder).glob("*.jpeg"))
        p = Path("data/simple-rectangle") / "argumented_masks.pt"
        self.masks = torch.load(p).numpy()
        self.transform = transform

    def __len__(self):
        return self.masks.shape[0]

    def __getitem__(self, idx):
        image = self.masks[idx]
        # pil_image = Image.open(image_filepath)
        # image = image.numpy()
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image.squeeze(0)




def main():
    train_transform = A.Compose(
        [
            A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, fill_value=0, p=0.75),
            A.CoarseDropout(max_holes=64, max_height=16, max_width=16, fill_value=0, p=0.75),
            # .Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    datasets = MaskDataset(train_transform)
    batch_size = 32
    dataloader = DataLoader(
        datasets,
        batch_size=batch_size,
        shuffle=True
    )
    for k in dataloader:
        print(k.shape)
        break


if __name__ == "__main__":
    main()
