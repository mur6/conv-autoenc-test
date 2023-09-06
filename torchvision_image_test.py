from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
# from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import albumentations as A

# import torchvision.transforms as T
from torchvision.transforms import v2

import torchvision.transforms.v2 as transforms

transform = transforms.Compose(
    [
        transforms.ColorJitter(contrast=0.5),
        transforms.RandomRotation(30),
        transforms.CenterCrop(480),
    ]
)

# def imshow(img):
#     img = torchvision.utils.make_grid(img)
#     img = img / 2 + 0.5
#     npimg = img.detach().numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()


# def train(net, criterion, optimizer, epochs, trainloader):
#     losses = []
#     output_and_label = []

#     for epoch in range(1, epochs+1):
#         print(f'epoch: {epoch}, ', end='')
#         running_loss = 0.0
#         for counter, (img, _) in enumerate(trainloader, 1):
#             optimizer.zero_grad()
#             output = net(img)
#             loss = criterion(output, img)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item()
#         avg_loss = running_loss / counter
#         losses.append(avg_loss)
#         print('loss:', avg_loss)
#         output_and_label.append((output, img))
#     print('finished')
#     return output_and_label, losses

albumentations_transform = A.Compose(
    [
        A.Resize(256, 256),  # 画像のリサイズ
        A.RandomCrop(224, 224),  # ランダムクロップ
        A.HorizontalFlip(p=0.5),  # 水平反転
        A.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5
        ),  # 色調変更
    ]
)


def get_images():
    sample_image_path = "../poetry-test-proj/samples/02"
    images = list(Path(sample_image_path).glob("*.jpeg"))
    print(images)
    images = [Image.open(image) for image in images]
    transform = v2.ToTensor()
    images = list(map(transform, images))
    # images = torch.stack(images)
    # augmented = albumentations_transform(image=images)
    # images = augmented["image"]
    return images

class CatsVsDogsDataset(Dataset):
    def __init__(self, images_filepaths, transform=None):
        self.images_filepaths = images_filepaths
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image


def main():
    # # iterator = iter(trainloader)
    # # x, y = next(iterator)
    # # print(y)
    # masks = torch.load("data/simple-rectangle/MASKS_TENSOR.pt")
    # masks_train_loader = DataLoader(masks, batch_size=batch_size, shuffle=True)
    # for m in masks_train_loader:
    #     # print(m[0].shape)
    #     # print(m[0])
    #     im = m[0]
    #     im = transform(im.unsqueeze(0))
    #     plt.imshow(im.squeeze(0))
    #     break
    # plt.show()
    images = get_images()
    # print(images)
    batch_size = 32
    dataloader = DataLoader(
        images,
        batch_size=batch_size,
        shuffle=True,
        # num_workers=4,
        collate_fn=custom_transform,
    )
    for k in dataloader:
        print(k.shape)

def main2():
    images = get_images()
    images = transform(images)
    print(images.shape, len(images))
    fig, axes = plt.subplots(3, 3, figsize=(9, 9), tight_layout=True)
    for ax, img in zip(axes.flatten(), images):
        ax.imshow(img.permute(1, 2, 0))
    plt.savefig("images.png")
    # plt.show()


if __name__ == "__main__":
    main()
