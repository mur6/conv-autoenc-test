

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T
import torchvision.transforms.v2 as transforms


# download the dataset of MNIST
def prepare_mnist_dataloaders():
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testset = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=32, shuffle=True)
    return trainloader, testloader


# Display train_loader first 4 images as a sample
def show_sample_image(trainloader):
    sample_batch = next(iter(trainloader))
    images, labels = sample_batch
    grid = torchvision.utils.make_grid(images[:16])
    plt.imshow(grid.permute(1, 2, 0))
    plt.show()

# Convolutional Autoencoder in Pytorch on MNIST dataset
class ConvAutoencoder(torch.nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(16, 32, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=1)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(32, 16, 3, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 8, 3, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(8, 1, 4, stride=2),
            torch.nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Train ConvAutoencoder
def train_conv_autoencoder(net, criterion, optimizer, epochs, trainloader):
    losses = []
    output_and_label = []

    for epoch in range(1, epochs + 1):
        print(f"epoch: {epoch}, ", end="")
        running_loss = 0.0
        for counter, (img, _) in enumerate(trainloader, 1):
            img = img.cuda()
            # print(f"img={img.shape} {img.dtype}")
            # print(f"output_image={output_image.shape} {output_image.dtype}")
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
    return net, output_and_label, losses

def main():
    trainloader, testloader = prepare_mnist_dataloaders()
    # show_sample_image(trainloader)
    net = ConvAutoencoder().cuda()
    # x = torch.randn(8, 1, 96, 96)
    # out = net(x)
    # print(out.shape)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    EPOCHS = 400
    model, output_and_label, losses = train_conv_autoencoder(
        net, criterion, optimizer, EPOCHS, trainloader)

if __name__ == "__main__":
    main()
