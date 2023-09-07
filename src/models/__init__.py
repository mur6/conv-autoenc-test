from torch import nn


class AutoEncoderV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            # nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x


class AutoencoderV1(nn.Module):
    def __init__(self):
        super(AutoencoderV1, self).__init__()

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
        self.decoder = nn.Sequential(
            # conv 4
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=2
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # conv 5
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=2
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # conv 6 out
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.ConvTranspose2d(
                in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1
            ),
            nn.Softmax(),
        )

    def forward(self, x):
        x = self.encoder(x)
        # print(f"x={x.shape}")
        x = self.decoder(x)
        return x


class AutoencoderV2(nn.Module):
    def __init__(self):
        super(AutoencoderV2, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            # conv1
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=2),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # conv2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # conv3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            # conv 4
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            # conv 5
            # nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=2
            ),
            nn.ReLU(),
            # conv 6 out
            # nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ConvTranspose2d(
                in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=2
            ),
            nn.Softmax(),
        )

    def forward(self, x):
        x = self.encoder(x)
        # print(f"x={x.shape}")
        x = self.decoder(x)
        return x



class AutoEncoderV4(nn.Module):
    def __init__(self):
        super().__init__()
        latent_dim = 8
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(12 * 12 * 64, latent_dim)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 12 * 12 * 64),
            nn.ReLU(),
            nn.Unflatten(1, (64, 12, 12)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x
