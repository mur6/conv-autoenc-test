import albumentations as A
from albumentations.pytorch import ToTensorV2

MASK_SIZE = 32
DROP_SIZE = int(MASK_SIZE * 0.1666)


train_transform_complicated = A.Compose(
    [
        A.CoarseDropout(max_holes=50, max_height=1, max_width=1, fill_value=0, p=0.9),
        A.OneOf(
            [
                A.Cutout(
                    num_holes=4,
                    max_h_size=DROP_SIZE,
                    max_w_size=DROP_SIZE,
                    fill_value=0,
                    p=0.75,
                ),
                A.CoarseDropout(
                    max_holes=12,
                    max_height=DROP_SIZE,
                    max_width=DROP_SIZE,
                    fill_value=0,
                    p=0.9,
                ),
            ],
            p=0.95,
        ),
        ToTensorV2(),
    ]
)

train_transform = A.Compose(
    [
        A.OneOf(
            [
                A.CoarseDropout(
                    max_holes=100, max_height=1, max_width=1, fill_value=0, p=0.95
                ),
                A.CoarseDropout(
                    max_holes=50, max_height=1, max_width=1, fill_value=0, p=0.95
                ),
            ],
            p=0.95,
        ),
        ToTensorV2(),
    ]
)
