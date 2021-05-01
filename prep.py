import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, TensorDataset
import config


# def load_trace(
#     filename: str, split_idx: int = 1_000_000
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     df = pd.read_csv(
#         filename,
#         compression="xz",
#         names=["insn_id", "cycle", "addr", "pc", "hit"],
#     )
#     df["insn_id"] = df["insn_id"].astype(np.int_)
#     df["cycle"] = df["cycle"].astype(np.int_)
#     df["addr"] = df["addr"].apply(lambda n: int(n, 16))
#     df["pc"] = df["pc"].apply(lambda n: int(n, 16))
#     return df[:split_idx], df[split_idx:]

def load_trace(filename: str) -> pd.DataFrame:
    return pd.read_csv(
        filename,
        compression="xz",
        names=["insn_id", "cycle", "addr", "pc", "hit"],
    )


def df_to_tensor(
    df: pd.DataFrame,
    diff_cols: Optional[List[str]] = None,
    window_size: int = config.DIFFS_DEFAULT_WINDOW,
) -> torch.Tensor:
    if diff_cols is not None:
        diffed = torch.diff(torch.from_numpy(df[diff_cols].values), dim=0)
        not_diffed = torch.from_numpy(
            df[[name for name in df.columns if name not in diff_cols]].values[1:]
        )
        samples = torch.hstack([not_diffed, diffed])
    else:
        samples = torch.from_numpy(df.values)

    return torch.stack(
        [samples[i: i + window_size]
            for i in range(len(samples) - window_size + 1)]
    )


def to_diffs(
    df: pd.DataFrame,
    window_size: int = config.DIFFS_DEFAULT_WINDOW,
    move_to_gpu: bool = False,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict, bool]:
    addr_t = df["addr"].apply(lambda n: int(n, 16)).values
    addr_w = sliding_window_view(addr_t, window_size + 2, writeable=True)
    diffs = np.diff(addr_w)
    max_diffs = np.max(diffs)
    min_diffs = np.min(diffs)
    range_diffs = (max_diffs-min_diffs)
    diffs = 2*((diffs-min_diffs)/range_diffs)-1
    sequences = diffs[:, :window_size]
    targets = diffs[:, window_size]
    targets = torch.tensor(targets).float()
    sequences = torch.tensor(sequences).unsqueeze(2).float()
    instr_id = torch.tensor(sliding_window_view(df["insn_id"].to_numpy()[:-2], window_size, writeable=True)).unsqueeze(2)
    #[-2] to ignore meta data for last delta
    addr = torch.tensor(sliding_window_view(addr_t[:-2], window_size, writeable=True)).unsqueeze(2).float()
    range_addr = (addr.max()-addr.min())
    min_addr = addr.min()
    addr = (addr-min_addr)/range_addr
    pc = torch.tensor(sliding_window_view([int(str(pc).strip(), 16)for pc in df["pc"].to_numpy()][:-2], window_size, writeable=True)).unsqueeze(2).float()
    min_pc = pc.min()
    range_pc = (pc.max()-pc.min())
    pc = (pc-min_pc)/range_pc
    hit = torch.tensor(sliding_window_view(df["hit"][:-2].to_numpy(), window_size, writeable=True)).unsqueeze(2).float()

    sequences = torch.cat((sequences, pc, addr, hit), 2)

    dataset_size = (
        sequences.numpy().nbytes
        + targets.numpy().nbytes
    )

    if move_to_gpu and (
        dataset_size < config.GPU_MEM_SIZE_BYTES - config.GPU_MEM_SIZE_MARGIN_BYTES
    ):
        dataset_on_gpu = True
        sequences.cuda()
        targets.cuda()
    else:
        dataset_on_gpu = False
        
    norm_data = {}
    norm_data["range_addr"] = range_addr
    norm_data["min_addr"] = min_addr
    norm_data["min_pc"] = min_pc
    norm_data["range_pc"] = range_pc
    norm_data["min_diffs"] = min_diffs
    norm_data["range_diffs"] = range_diffs
    norm_data["instr_id"] = instr_id

    return (sequences, targets), norm_data, dataset_on_gpu


def to_dataloader(
    dataset: np.ndarray, batch_size: int, dataset_on_gpu: bool = False, shuffle: bool = False
) -> DataLoader:
    dataset = TensorDataset(*dataset)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=(not dataset_on_gpu)
    )


if __name__ == "__main__":
    df, _ = load_trace(sys.argv[1])
    print(df_to_tensor(df, col_names=["addr", "pc"])[0])
