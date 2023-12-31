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


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
# trainset = CIFAR10("./data", train=True, transform=transform, download=True)
# testset = CIFAR10("./data", train=False, transform=transform, download=True)

# batch_size = 50
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


def main():
    conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
    conv2 = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
    pool = torch.nn.MaxPool2d(2)

    # iterator = iter(trainloader)
    # x, _ = next(iterator)
    x = torch.ones(32, 1, 96, 96)

    #
    # x = conv1(x)
    # print("after conv1:", x.shape)
    # x = torch.relu(x)
    # x = pool(x)
    # print("after 1st pool:", x.shape)
    # x = conv2(x)
    # print("after conv2:", x.shape)
    # x = torch.relu(x)
    # x = pool(x)
    # print("after 2nd pool:", x.shape)

    # enc_old = torch.nn.Sequential(
    #     torch.nn.Conv2d(3, 16, 3, padding=1),
    #     torch.nn.ReLU(),
    #     torch.nn.MaxPool2d(2),
    #     torch.nn.Conv2d(16, 8, 3, padding=1),
    #     torch.nn.ReLU(),
    #     torch.nn.MaxPool2d(2),
    # )
    enc = torch.nn.Sequential(
        torch.nn.Conv2d(1, 16, kernel_size=4, padding=1, stride=2),
        torch.nn.ReLU(),
        # torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(16, 32, kernel_size=4, padding=1, stride=2),
        torch.nn.ReLU(),
    )
    print("init:", x.shape)
    x = enc(x)
    print("after 2nd pool:", x.shape)
    dec = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
        torch.nn.Tanh(),
    )
    x = dec(x)
    print("after decode:", x.shape)
    net = AutoEncoder2(enc, dec)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    EPOCHS = 100

    output_and_label, losses = train(net, criterion, optimizer, EPOCHS, trainloader)


if __name__ == "__main__":
    main()
