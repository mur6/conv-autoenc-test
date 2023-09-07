import torch
from torch import nn
import torch.nn.functional as F


class AutoEncoderV0(nn.Module):
    def __init__(self):
        super().__init__()
        latent_dim = 8 * 2
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=4, padding=1, stride=2),
            # nn.ReLU(),
            nn.Flatten(),
            nn.Linear(24 * 24 * 32, latent_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 24 * 24 * 32),
            nn.ReLU(),
            nn.Unflatten(1, (32, 24, 24)),
            # nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.dec(x)
        return x


def main():
    x = torch.rand(8, 1, 96, 96)
    model = AutoEncoderV0()
    # enc = model.encoder
    # out = model(x)
    x = model.encoder(x)
    mean, logvar = torch.chunk(x, 2, dim=1)
    logvar = F.softplus(logvar)
    #print(f"out={out.shape}")
    print(f"mean={mean.shape} logvar={logvar.shape}")


main()
