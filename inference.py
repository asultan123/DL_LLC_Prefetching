import torch
from torch import optim, nn
import torch.nn.functional as F


class MHARegressor(nn.Module):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
    ):
        super().__init__()
