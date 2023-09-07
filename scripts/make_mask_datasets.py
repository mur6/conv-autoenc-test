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

# from torchvision import transforms
import torchvision.transforms as T
from torchvision.transforms import v2


SIZE = 32

resize_transform = v2.Resize((SIZE, SIZE))

transforms = v2.Compose(
    [v2.RandomZoomOut(side_range=(1, 2.5), p=1.0), v2.Resize((SIZE, SIZE))]
)

# def test():
#     # masks_train_loader = DataLoader(masks, batch_size=batch_size, shuffle=True)
#     # for m in masks_train_loader:
#     #     # print(m[0].shape)
#     #     # print(m[0])
#     #     im = m[0]
#     #     im = transform(im.unsqueeze(0))
#     num = masks.shape[0]
#     index = random.randint(0, num - 1)
#     index = 288
#     print(num, index)
#     orig_img = masks[index].unsqueeze(0)
#     # print(im.shape)


def augments_data(orig_img):
    return [transforms(orig_img.unsqueeze(0)) for _ in range(4)]


def main():
    masks = torch.load("data/simple-rectangle/MASKS_TENSOR.pt")
    masks = resize_transform(masks)

    def _iter():
        for orig_img in masks:
            yield orig_img.unsqueeze(0)
            yield from augments_data(orig_img)

    image_list = list(_iter())
    images = torch.cat(image_list, dim=0)
    p = Path("data/simple-rectangle") / "argumented_masks_32x32.pt"
    torch.save(images, p)
    print(f"argumented masks saved! {images.shape}")
    show_result = False
    if show_result:
        fig, axes = plt.subplots(2, 2, figsize=(9, 9), tight_layout=True)
        sampled_images = random.sample(image_list, 4)
        for ax, img in zip(axes.flatten(), sampled_images):
            ax.imshow(img.squeeze(0))
        plt.show()


if __name__ == "__main__":
    main()
