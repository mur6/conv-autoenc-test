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


sample_image_path = "../poetry-test-proj/samples/02"



class MaskDataset(Dataset):
    def __init__(self, transform=None):
        # self.images_filepaths = list(Path(images_folder).glob("*.jpeg"))
        p = Path("data/simple-rectangle") / "argumented_masks.pt"
        self.masks = torch.load(p)
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        pil_image = Image.open(image_filepath)
        image = np.asarray(pil_image)

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image


# データ拡張と正規化を組み合わせる
def custom_transform(image, label):
    image = torchvision_transform(image)  # 画像をPyTorchのTensorに変換
    augmented = albumentations_transform(image=image.numpy())  # NumPy配列でデータ拡張
    image = augmented["image"]
    return image, label


# データローダーを作成
batch_size = 32
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    collate_fn=custom_transform,
)
