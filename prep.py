import sys
from typing import Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, TensorDataset


def load_trace(filename: str) -> pd.DataFrame:
    return pd.read_csv(
        filename,
        compression="xz",
        names=["insn_id", "cycle", "addr", "pc", "hit"],
    )


def to_train_diffs(df: pd.DataFrame, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    diffs = np.diff(df["addr"].apply(lambda n: int(n, 16)).values).astype(np.float64)
    diffs /= max(diffs)
    sequences = sliding_window_view(diffs, window_size)
    targets = sequences[1:, -1]
    return sequences[:-1], targets


class RegressionRNN(nn.Module):
    def __init__(self, hidden: int, n_layers: int = 1):
        super().__init__()
        self.rnn = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=n_layers)
        self.regressor = nn.Linear(hidden, 1)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        latent_sequence, _ = self.rnn(input_seq.permute(1, 0, 2))
        return self.regressor(latent_sequence[-1])


if __name__ == "__main__":
    # CONSTANTS
    BATCH_SIZE = 64
    SEQ_LEN = 32
    N_LAYERS = 2
    HIDDEN_DIM = 256
    MAX_CLIP = 50

    # CODE
    df = load_trace(sys.argv[1])
    data = TensorDataset(*map(torch.tensor, to_train_diffs(df, SEQ_LEN)))
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    network = RegressionRNN(HIDDEN_DIM, n_layers=N_LAYERS).double().cuda()
    opt = optim.Adam(network.parameters(), lr=1e-3, eps=1e-7)
    criterion = nn.MSELoss().cuda()
    for i, (seq_batch, label_batch) in enumerate(loader):
        opt.zero_grad(set_to_none=True)
        loss = criterion(
            network(seq_batch.unsqueeze(-1).cuda()), label_batch.unsqueeze(-1).cuda()
        )
        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), MAX_CLIP)
        opt.step()
        if i % 100 == 99:
            print(f"Iter{i} loss: {loss}")
