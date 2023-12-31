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
from torchvision.transforms import v2
import torchvision.transforms.v2 as transforms

from src.models import AutoEncoderV4, AutoEncoderV5, CVAEv2, CVAEv3
from src.augmenting import train_transform


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
            output, z, ave, log_dev = net(img)
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


class MaskDataset(Dataset):
    def __init__(self, transform=None):
        p = Path("data/simple-rectangle") / "argumented_masks_32x32.pt"
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
    datasets = MaskDataset(train_transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 32
    train_loader = DataLoader(datasets, batch_size=batch_size, shuffle=True)

    # Set up training parameters
    learning_rate = 0.04
    epochs = 500

    # Initialize the autoencoder and optimizer
    model = CVAEv3(device).to(device)
    if False:
        criterion = torch.nn.BCELoss()  # Binary Cross-Entropy Loss
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    if False:
        criterion = torch.nn.MSELoss()

    def criterion(predict, target, ave, log_dev):
        bce_loss = F.binary_cross_entropy(predict, target, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + log_dev - ave**2 - log_dev.exp())
        loss = bce_loss + kl_loss
        return loss

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.95)

    # Create a directory to save checkpoints
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    (checkpoint_dir / "cvae").mkdir(exist_ok=True)

    # Training loop
    for epoch in range(epochs):
        # print(f"epoch: {epoch}, ", end="")
        total_loss = 0
        for img, label_img in train_loader:
            img = img.to(device)
            label_img = label_img.to(device)
            # print(f"img={img.shape} {img.dtype}")
            # print(f"output_image={label_img.shape} {label_img.dtype}")
            output, z, ave, log_dev = model(img)
            # print(f"output={output.shape} label_img={label_img.shape}")
            # print(torch.max(output), torch.min(output))
            # print(torch.max(label_img), torch.min(label_img))

            # loss = criterion(output, label_img)
            loss = criterion(output, label_img, ave, log_dev)

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # avg_loss = total_loss / counter
        # # losses.append(avg_loss)
        # print(f"loss:", avg_loss)

        # Save checkpoint every epoch
        if (epoch + 1) % 50 == 0:
            checkpoint_path = checkpoint_dir / "cvae" / f"cvae_v3_epoch{epoch + 1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved: {checkpoint_path}")

        print(
            f"Epoch [{epoch + 1}/{epochs}] Loss: {total_loss / len(train_loader)} {scheduler.get_last_lr()}"
        )
        scheduler.step()
    print("Training finished.")


if __name__ == "__main__":
    main()
