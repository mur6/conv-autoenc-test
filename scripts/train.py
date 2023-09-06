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
        for counter, (img, output_image) in enumerate(trainloader, 1):
            img = img.cuda()
            output_image = output_image.cuda()
            # print(f"img={img.shape} {img.dtype}")
            # print(f"output_image={output_image.shape} {output_image.dtype}")
            optimizer.zero_grad()
            output = net(img)
            loss = criterion(output, output_image)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / counter
        losses.append(avg_loss)
        print("loss:", avg_loss)
        output_and_label.append((output, img))
    print("finished")
    return net, output_and_label, losses


# Define the Convolutional Autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            # conv1
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # conv2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # conv3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Decoder layers
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),
        #     nn.ReLU(),
        #     # nn.Upsample(scale_factor=2, mode='bilinear'),
        #     nn.ConvTranspose2d(
        #         64, 32, kernel_size=3, stride=2),
        #     nn.ReLU(),
        #     nn.Upsample(scale_factor=2, mode='bilinear'),
        #     # nn.ConvTranspose2d(
        #     #     32, 1, kernel_size=3, stride=2
        #     # ),
        #     nn.Sigmoid(),  # Output values between 0 and 1 for binary images
        # )
        self.decoder = nn.Sequential(
            # conv 4
            nn.ConvTranspose2d(in_channels=128, out_channels=64,
                               kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # conv 5
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(in_channels=64, out_channels=32,
                               kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # conv 6 out
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(in_channels=32, out_channels=1,
                               kernel_size=3, stride=1, padding=1),
            nn.Softmax()
        )

    def forward(self, x):
        x = self.encoder(x)
        print(f"x={x.shape}")
        x = self.decoder(x)
        return x


class MaskDataset(Dataset):
    def __init__(self, transform=None):
        p = Path("data/simple-rectangle") / "argumented_masks.pt"
        self.masks_pt = torch.load(p)
        self.masks_np = self.masks_pt.numpy()
        self.transform = transform

    def __len__(self):
        return self.masks_pt.shape[0]

    def __getitem__(self, idx):
        image = self.masks_np[idx]
        # pil_image = Image.open(image_filepath)
        # image = image.numpy()
        # if self.transform is not None:
        transformed_image = self.transform(image=image)["image"]
        # return transformed_image.squeeze(0), self.masks_pt[idx].squeeze(0)
        return transformed_image, self.masks_pt[idx].unsqueeze(0)


def old_train():
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    EPOCHS = 400
    model, output_and_label, losses = train(
        net, criterion, optimizer, EPOCHS, train_loader
    )
    torch.save(model.state_dict(), "model_weight_400.pth")


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 32
    train_loader = DataLoader(datasets, batch_size=batch_size, shuffle=True)

    # Set up training parameters
    learning_rate = 0.001
    epochs = 250

    # Initialize the autoencoder and optimizer
    model = Autoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss

    # Create a directory to save checkpoints
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for img, label_img in train_loader:

            img = img.to(device)
            label_img = label_img.to(device)
            print(f"img={img.shape} {img.dtype}")
            print(f"output_image={label_img.shape} {label_img.dtype}")

            # Forward pass
            output = model(img)
            print(f"output={output.shape} {output.dtype}")
            loss = criterion(output, label_img)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Save checkpoint every epoch
        if (epoch + 1) % 20 == 0:
            checkpoint_path = checkpoint_dir / f"autoencoder_epoch{epoch + 1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved: {checkpoint_path}")

        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {total_loss / len(train_loader)}")

    print("Training finished.")


if __name__ == "__main__":
    main()
