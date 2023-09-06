import albumentations as A
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# torchvisionの変換関数を定義
torchvision_transform = T.Compose(
    [
        T.ToTensor(),  # Tensorに変換
        T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),  # 平均と標準偏差で正規化
    ]
)

# albumentationsのデータ拡張を定義
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

# データセットを読み込む
sample_image_path = "../poetry-test-proj/samples/02"
dataset = ImageFolder(
    sample_image_path, transform=None
)
# ここでtransformはNoneに設定します


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
