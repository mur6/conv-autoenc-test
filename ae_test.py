from torch import nn


class AutoEncoderV0(nn.Module):
    def __init__(self, enc, dec):
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


def main():
    pass

main()
