import torch
from torch import nn
import torch.nn.functional as F

from src.models import CVAEv2, CVAEv3


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


if __name__ == "__main__":
    main()
