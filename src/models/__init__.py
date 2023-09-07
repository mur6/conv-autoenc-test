from torch import nn


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

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
