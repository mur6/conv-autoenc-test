import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10
import numpy as np
import matplotlib.pyplot as plt


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


batch_size = 50

# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# trainset = CIFAR10('./data', train=True, transform=transform, download=True)
# testset = CIFAR10('./data', train=False, transform=transform, download=True)

# trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
# testloader = DataLoader(testset, batch_size=batch_size // 10, shuffle=False)


class AutoEncoder2(torch.nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x


import torchvision.transforms as T
import torchvision.transforms.v2 as t_v2
import albumentations as A

# transform = A.Compose([
#     A.RandomCrop(width=256, height=256),
#     A.HorizontalFlip(p=0.5),
#     A.RandomBrightnessContrast(p=0.2),
# ])

transform = t_v2.RandomZoomOut(p=1.0)


def main():
    # iterator = iter(trainloader)
    # x, y = next(iterator)
    # print(y)
    masks = torch.load("data/simple-rectangle/MASKS_TENSOR.pt")
    masks_train_loader = DataLoader(masks, batch_size=batch_size, shuffle=True)
    for m in masks_train_loader:
        # print(m[0].shape)
        # print(m[0])
        im = m[0]
        im = transform(im.unsqueeze(0))
        plt.imshow(im.squeeze(0))
        break
    plt.show()


if __name__ == "__main__":
    main()
