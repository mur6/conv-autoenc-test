import torch
from torch import nn
import torch.nn.functional as F

from src.models import CVAEv2


class AutoEncoderV6(nn.Module):
    def __init__(self):
        super().__init__()
        latent_dim = 8
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            # nn.Conv2d(32, 64, kernel_size=4, padding=1, stride=2),
            # nn.ReLU(),
            nn.Flatten(),
            nn.Linear(24 * 24 * 32, latent_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 24 * 24 * 32),
            nn.ReLU(),
            nn.Unflatten(1, (32, 24, 24)),
            # nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            # nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def reparametrizaion(self, mean, log_var, device):
        eps = torch.randn(mean.shape).to(device)
        return mean + torch.sqrt(log_var) * eps

    def forward(self, x, device):
        # x = x.view(-1, self.x_dim)
        x = self.encoder(x)
        mean, log_var = torch.chunk(x, 2, dim=1)
        # print(f"mean={mean.shape} log_var={log_var.shape}")
        z = self.reparametrizaion(mean, log_var, device)
        x_hat = self.decoder(z)  # 潜在ベクトルを入力して、再構築画像 y を出力
        return x_hat, z, mean, log_var


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.rand(8, 1, 96, 96)
    model = CVAEv2(device)
    # enc = model.encoder
    output, z, ave, log_dev = model(x)
    print(f"output={output.shape} z={z.shape}")
    # x = model.encoder(x)
    # mean, logvar = torch.chunk(x, 2, dim=1)
    # log_var = F.softplus(log_var)
    # #print(f"out={out.shape}")
    # print(f"mean={mean.shape} log_var={log_var.shape}")
    # z = reparametrizaion(mean, log_var)


main()
