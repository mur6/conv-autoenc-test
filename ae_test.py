import torch
from torch import nn
import torch.nn.functional as F

from src.models import CVAEv2



class CVAEv3(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        latent_dim = 16
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, latent_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8 * 8 * 32),
            nn.ReLU(),
            nn.Unflatten(1, (32, 8, 8)),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def reparametrizaion(self, mean, log_var, device):
        eps = torch.randn_like(mean).to(device)
        return mean + torch.exp(log_var / 2) * eps

    def forward(self, x):
        # x = x.view(-1, self.x_dim)
        x = self.encoder(x)
        mean, log_var = torch.chunk(x, 2, dim=1)
        # print("mean=", mean)
        # print("log_var=", log_var)
        # log_var = F.softplus(log_var)
        # print("log_var=", log_var)
        # print(f"mean={mean.shape} log_var={log_var.shape}")
        z = self.reparametrizaion(mean, log_var, self.device)
        # z = F.relu(z)
        # print("z=", z)
        x_hat = self.decoder(z)
        return x_hat, z, mean, log_var


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.rand(8, 1, 32, 32)
    model = CVAEv3(device)
    # y = model.encoder(x)
    # print(f"y={y.shape}")
    output, z, ave, log_dev = model(x)
    print(f"output={output.shape} z={z.shape}")
    # mean, logvar = torch.chunk(x, 2, dim=1)
    # log_var = F.softplus(log_var)
    # #print(f"out={out.shape}")
    # print(f"mean={mean.shape} log_var={log_var.shape}")
    # z = reparametrizaion(mean, log_var)


main()
