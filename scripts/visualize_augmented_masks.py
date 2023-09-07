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

from src.augmenting import train_transform


class MaskDataset(Dataset):
    def __init__(self, transform=None):
        # self.images_filepaths = list(Path(images_folder).glob("*.jpeg"))
        p = Path("data/simple-rectangle") / "argumented_masks_32x32.pt"
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
    datasets = MaskDataset(train_transform)
    batch_size = 32
    dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True)
    for images in dataloader:
        # print(k.shape)
        fig, axes = plt.subplots(2, 2, figsize=(9, 9), tight_layout=True)
        for ax, img in zip(axes.flatten(), images[:4]):
            ax.imshow(img.squeeze(0))
        break
    plt.show()


if __name__ == "__main__":
    main()
