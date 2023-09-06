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


def imshow(img):
    img = torchvision.utils.make_grid(img)
    img = img / 2 + 0.5
    npimg = img.detach().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def train(net, criterion, optimizer, epochs, trainloader):
    losses = []
    output_and_label = []

    for epoch in range(1, epochs + 1):
        print(f"epoch: {epoch}, ", end="")
        running_loss = 0.0
        for counter, (img, _) in enumerate(trainloader, 1):
            optimizer.zero_grad()
            output = net(img)
            loss = criterion(output, img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / counter
        losses.append(avg_loss)
        print("loss:", avg_loss)
        output_and_label.append((output, img))
    print("finished")
    return output_and_label, losses


class AutoEncoder2(torch.nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x


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
            A.Cutout(num_holes=4, max_h_size=16, max_w_size=16, fill_value=0, p=0.75),
            A.CoarseDropout(
                max_holes=12, max_height=16, max_width=16, fill_value=0, p=0.6
            ),
            # .Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    datasets = MaskDataset(train_transform)
    batch_size = 32
    train_loader = DataLoader(datasets, batch_size=batch_size, shuffle=True)
    # for images in train_loader:
    #     fig, axes = plt.subplots(2, 2, figsize=(9, 9), tight_layout=True)
    #     for ax, img in zip(axes.flatten(), images[:4]):
    #         ax.imshow(img.squeeze(0))
    #     break
    # plt.show()


if __name__ == "__main__":
    main()
