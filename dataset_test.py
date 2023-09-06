import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import CIFAR10
import numpy as np
import matplotlib.pyplot as plt



def main():

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
