from pathlib import Path

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
# from torchvision import transforms
import torchvision.transforms as T
from torchvision.transforms import v2

import torchvision.transforms.v2 as transforms

transform = transforms.Compose(
    [
        transforms.ColorJitter(contrast=0.5),
        transforms.RandomRotation(30),
        transforms.CenterCrop(480),
    ]
)

def main():
    masks = torch.load("data/simple-rectangle/MASKS_TENSOR.pt") # .numpy()
    # masks_train_loader = DataLoader(masks, batch_size=batch_size, shuffle=True)
    # for m in masks_train_loader:
    #     # print(m[0].shape)
    #     # print(m[0])
    #     im = m[0]
    #     im = transform(im.unsqueeze(0))
    im = masks[0]
    # print(masks_np)
    plt.imshow(im.squeeze(0))
    plt.show()


if __name__ == "__main__":
    main()
