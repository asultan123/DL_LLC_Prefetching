import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, TensorDataset
import config


def load_trace(
    filename: str, split_idx: int = 1_000_000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(
        filename,
        compression="xz",
        names=["insn_id", "cycle", "addr", "pc", "hit"],
        dtype={
            "insn_id": np.int_,
            "cycle": np.int_,
            "addr": str,
            "pc": str,
            "hit": np.int_,
        },
    )
    df["insn_id"] = df["insn_id"].astype(np.int_)
    df["cycle"] = df["cycle"].astype(np.int_)
    df["addr"] = df["addr"].apply(lambda n: int(n, 16))
    df["pc"] = df["pc"].apply(lambda n: int(n, 16))
    return df[:split_idx], df[split_idx:]


def df_to_tensor(
    df: pd.DataFrame,
    diff_cols: Optional[List[str]] = None,
    window_size: int = config.DIFFS_DEFAULT_WINDOW,
) -> torch.Tensor:
    if diff_cols is not None:
        diffed = torch.from_numpy(np.diff(df[diff_cols].values, axis=0))
        not_diffed = torch.from_numpy(
            df[[name for name in df.columns if name not in diff_cols]].values[1:]
        )
        samples = torch.hstack([not_diffed, diffed]).double()
    else:
        samples = torch.from_numpy(df.values).double()

    samples /= torch.max(samples, dim=0)[0].double()
    return torch.stack(
        [samples[i : i + window_size] for i in range(len(samples) - window_size + 1)]
    ).cuda()


# def to_diffs(
#     df: pd.DataFrame,
#     window_size: int = config.DIFFS_DEFAULT_WINDOW,
#     move_to_gpu: bool = False,
# ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict, bool]:
#     addr_t = df["addr"].apply(lambda n: int(n, 16)).values
#     addr_w = np.copy(sliding_window_view(addr_t, window_size + 2))
#     diffs = np.diff(addr_w)
#     max_diffs = np.max(np.abs(diffs))
#     diffs = diffs / max_diffs
#     sequences = diffs[:, :window_size]
#     targets = diffs[:, window_size]
#     sequences = torch.tensor(sequences)
#     targets = torch.tensor(targets)
#     instr_id = torch.tensor(df["insn_id"].to_numpy()[window_size + 1 :])
#     last_addr = torch.tensor(addr_t[window_size + 1 :])
#     pc = torch.tensor(
#         [int(str(pc).strip(), 16) for pc in df["pc"].to_numpy()[window_size + 1 :]]
#     )
#     hit = torch.tensor(df["hit"].to_numpy()[window_size + 1 :])
#     dataset_size = (
#         sequences.numpy().nbytes
#         + targets.numpy().nbytes
#         + instr_id.numpy().nbytes
#         + last_addr.numpy().nbytes
#         + pc.numpy().nbytes
#         + hit.numpy().nbytes
#     )
#
#     if move_to_gpu and (
#         dataset_size < config.GPU_MEM_SIZE_BYTES - config.GPU_MEM_SIZE_MARGIN_BYTES
#     ):
#         dataset_on_gpu = True
#         sequences.cuda()
#         targets.cuda()
#         instr_id.cuda()
#         last_addr.cuda()
#         pc.cuda()
#         hit.cuda()
#     else:
#         dataset_on_gpu = False
#
#     metadata_tensors = {}
#     metadata_tensors["instr_id"] = instr_id
#     metadata_tensors["addr"] = last_addr
#     metadata_tensors["pc"] = pc
#     metadata_tensors["hit"] = hit
#     metadata_tensors["max_diffs"] = max_diffs
#
#     return (sequences, targets), metadata_tensors, dataset_on_gpu
#
#
# def to_dataloader(
#     dataset: np.ndarray, batch_size: int, dataset_on_gpu: bool = False
# ) -> DataLoader:
#     dataset = TensorDataset(*dataset)
#     return DataLoader(
#         dataset, batch_size=batch_size, shuffle=False, pin_memory=(not dataset_on_gpu)
#     )
#
#
# if __name__ == "__main__":
#     df, _ = load_trace(sys.argv[1])
#     print(df_to_tensor(df, col_names=["addr", "pc"])[0])
