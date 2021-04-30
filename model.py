from abc import ABC, abstractmethod
from typing import Optional
from torch.utils.data.dataset import TensorDataset
from tqdm import tqdm
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, dataloader
from lambda_networks import LambdaLayer
from math import floor
import config
import einops


class MLPrefetchModel(object):
    """
    Abstract base class for your models. For HW-based approaches such as the
    NextLineModel below, you can directly add your prediction code. For ML
    models, you may want to use it as a wrapper, but alternative approaches
    are fine so long as the behavior described below is respected.
    """

    @abstractmethod
    def load(self, path):
        """
        Loads your model from the filepath path
        """
        pass

    @abstractmethod
    def save(self, path):
        """
        Saves your model to the filepath path
        """
        pass

    @abstractmethod
    def train(self, data):
        """
        Train your model here. No return value. The data parameter is in the
        same format as the load traces. Namely,
        Unique Instr Id, Cycle Count, Load Address, Instruction Pointer of the Load, LLC hit/miss
        """
        pass

    @abstractmethod
    def generate(self, data):
        """
        Generate your prefetches here. Remember to limit yourself to 2 prefetches
        for each instruction ID and to not look into the future :).

        The return format for this will be a list of tuples containing the
        unique instruction ID and the prefetch. For example,
        [
            (A, A1),
            (A, A2),
            (C, C1),
            ...
        ]

        where A, B, and C are the unique instruction IDs and A1, A2 and C1 are
        the prefetch addresses.
        """
        pass


class NextLineModel(MLPrefetchModel):
    def load(self, path):
        # Load your pytorch / tensorflow model from the given filepath
        print("Loading " + path + " for NextLineModel")

    def save(self, path):
        # Save your model to a file
        print("Saving " + path + " for NextLineModel")

    def train(self, data):
        """
        Train your model here using the data

        The data is the same format given in the load traces. Namely:
        Unique Instr Id, Cycle Count, Load Address, Instruction Pointer of the Load, LLC hit/miss
        """
        print("Training NextLineModel")

    def generate(self, data):
        """
        Generate the prefetches for the prefetch file for ChampSim here

        As a reminder, no looking ahead in the data and no more than 2
        prefetches per unique instruction ID

        The return format for this function is a list of (instr_id, pf_addr)
        tuples as shown below
        """
        print("Generating for NextLineModel")
        prefetches = []
        for (instr_id, cycle_count, load_addr, load_ip, llc_hit) in data:
            # Prefetch the next two blocks
            prefetches.append((instr_id, ((load_addr >> 6) + 1) << 12))
            prefetches.append((instr_id, ((load_addr >> 6) + 2) << 12))

        return prefetches


class TimeSeriesLSTM(nn.Module):
    def __init__(self, hidden: int, n_layers: int = 1):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=1, hidden_size=hidden, num_layers=n_layers, batch_first=True
        )
        self.regressor = nn.Linear(hidden, 1)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        latent_sequence, _ = self.rnn(input_seq)
        return self.regressor(latent_sequence)


class TimeSeriesLSTMPrefetcher(MLPrefetchModel):
    def __init__(self, hidden: int, n_layers: int = config.N_LAYERS):
        super().__init__()
        self.model = TimeSeriesLSTM(hidden, n_layers).double().cuda()
        self.criterion = nn.MSELoss().cuda()
        self.optimizer = optim.Adam(self.model.parameters())
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=0.1)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def train(self, loader):
        self.model.train()
        bar = tqdm(total=len(loader))
        for i, (seq_batch, label_batch) in enumerate(loader):
            self.optimizer.zero_grad(set_to_none=True)
            model_prediction = self.model(seq_batch.unsqueeze(-1).cuda())[:, -1, :]
            loss = self.criterion(model_prediction, label_batch.unsqueeze(-1).cuda())
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), config.MAX_CLIP)
            self.optimizer.step()
            self.scheduler.step()
            bar.update(1)
            if i % 100 == 99:
                print(f"Iter{i} loss: {loss}")
        bar.close()

    def generate(self, loader, max_diffs):
        bar = tqdm(total=len(loader))
        with torch.no_grad():
            self.model.eval()
            prefetches = []
            for i, (seq_batch, _, instr_id, addr) in enumerate(loader):
                model_prediction = self.model(seq_batch.unsqueeze(-1).cuda())[:, -1, :]
                load_delta = (model_prediction * max_diffs).floor().long()
                load_addr = (addr.unsqueeze(1).cuda() + load_delta).squeeze(1).tolist()
                # prefetche counts greater than 2 will be ignored by simulator
                prefetches.extend(list(zip(instr_id.tolist(), load_addr)))
                bar.update(1)
        bar.close()
        return prefetches


class LambdaNetRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, key_dim: int, seq_length: int):
        super().__init__()
        self._attention_map_size = seq_length * hidden_dim
        self.sequence_embedder = nn.Linear(input_dim, hidden_dim)
        self.to_keys = nn.Linear(hidden_dim, key_dim)
        self.to_query = nn.Linear(hidden_dim, key_dim)
        self.to_values = nn.Linear(hidden_dim, hidden_dim)
        self.to_delta = nn.Linear(self._attention_map_size, 1)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        shape must be (sequence length, batch_size, dims)
        """
        embedded = self.sequence_embedder(input_seq)
        keys = torch.softmax(self.to_keys(embedded), dim=1)
        queries = self.to_query(embedded)
        values = self.to_values(embedded)
        context_lambda = torch.bmm(keys.permute(0, 2, 1), values)
        attention_map = torch.bmm(queries, context_lambda)
        return self.to_delta(attention_map.view(-1, self._attention_map_size))


class AttentionPrefetcher(MLPrefetchModel):
    def __init__(self, input_dim: int, hidden_dim: int, key_dim: int, seq_length: int) -> None:
        super().__init__()
        self.model = LambdaNetRegressor(input_dim, hidden_dim, key_dim, seq_length).double().cuda()

    def train(self, train_data: torch.Tensor):
        labels = train_data[1:, -1, -1]
        loader = DataLoader(
            TensorDataset(train_data[:-1], labels), batch_size=config.BATCH_SIZE,drop_last=True
        )
        opt = optim.Adam(self.model.parameters(), lr=1e-3)  # type: ignore
        criterion = nn.MSELoss()
        for i, (seq_batch, label_batch) in tqdm(enumerate(loader), total=len(loader)):
            opt.zero_grad(set_to_none=True)
            predictions = self.model(seq_batch)
            loss = criterion(predictions, label_batch)
            nn.utils.clip_grad_norm_(self.model.parameters(), config.MAX_CLIP)
            loss.backward()
            opt.step()
            if i % 100 == 99:
                print(f"Iter {i} loss: {loss:.3f}")

    def generate(self, data: torch.Tensor):
        pass
