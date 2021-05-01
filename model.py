from abc import ABC, abstractmethod
from typing import Optional
from torch.utils.data.dataset import TensorDataset
from tqdm import tqdm
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, dataloader
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import config
import numpy as np

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
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.double).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).double() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.encoder = nn.Linear(ntoken, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp*config.DIFFS_DEFAULT_WINDOW, 1)
        self.init_weights()

    def forward(self, src):
        src = src.squeeze().transpose(0,1)
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.transpose(0,1).flatten(1)
        output = self.decoder(output)
        return output
    
    def init_weights(self):
        initrange = 0.0001
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

class TransformerModelPrefetcher(MLPrefetchModel):
    def __init__(self, trace, ntokens : int = config.TRANSFORMER_ENCODER_NTOKENS, 
                 emsize : int = config.TRANSFORMER_ENCODER_EMSIZE, 
                 nhead : int = config.TRANSFORMER_ENCODER_NHID, 
                 nhid : int = config.TRANSFORMER_ENCODER_NHEAD, 
                 nlayers : int = config.TRANSFORMER_ENCODER_NLAYERS):
        super().__init__()
        self.model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers).double().cuda()
        self.criterion = nn.MSELoss().cuda()
        lr = 5.0 # learning rate
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.current_trace = trace
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def log_loss(self, loss, loss_type):
        loss_tensor = torch.tensor(loss)
        torch.save(loss_tensor, "./Graphs/{}_transformer_{}_losses.pt".format(self.current_trace,loss_type))

    def train(self, loader, iterations = None):
        self.model.train()
        bar = tqdm(total=len(loader))
        avg_loss = 0
        loss_list = []
        for i, (seq_batch, label_batch) in enumerate(loader):
            self.optimizer.zero_grad(set_to_none=True)
            model_prediction = self.model(seq_batch.unsqueeze(-1).cuda())
            loss = self.criterion(model_prediction, label_batch.unsqueeze(-1).cuda())
            avg_loss += loss.item()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), config.MAX_CLIP)
            self.optimizer.step()
            # self.scheduler.step()
            bar.update(1)
            if i % 100 == 99:
                avg_loss /= 100
                loss_list.append(avg_loss)
                print(f"Iter{i} avg_loss: {avg_loss}")
                avg_loss = 0
            if iterations is not None and ((i+1)%iterations==0):
                break
        self.log_loss(loss_list, "train")
        bar.close()

    def generate(self, loader, norm_data):
        
        range_addr = norm_data["range_addr"]
        min_addr = norm_data["min_addr"]
        mean_diffs = norm_data["mean_diffs"]
        std_diffs = norm_data["std_diffs"]
        
        bar = tqdm(total=len(loader))
        self.model.eval()
        def minmax_denormalize(val, min, range):
            return (((val + 1)/2)*range)+min
        def zscore_denormalize(val, mean, std):
            return (val*std)/mean
        loss_list = []
        
        with torch.no_grad():
            prefetches = []
            avg_loss = 0
            for i, (seq_batch, label_batch, instr_id) in enumerate(loader):
                model_prediction = self.model(seq_batch.unsqueeze(-1).cuda())
                loss = self.criterion(model_prediction, label_batch.unsqueeze(-1).cuda())
                avg_loss += loss.item()
                load_delta = zscore_denormalize(model_prediction, mean_diffs, std_diffs)
                addr = seq_batch[:, -1, 2] # : all entries in batch, -1 last entry in seq, 2 3rd feature (addr) 
                load_addr = minmax_denormalize(addr, min_addr, range_addr)
                target_addr = (load_addr + load_delta.squeeze(1).cpu()).long()
                target_addr = target_addr.numpy().astype("uint64")
                target_addr = (np.floor(target_addr/config.LLC_LINE_SIZE)*config.LLC_LINE_SIZE).astype("uint64")
                target_addr = target_addr.tolist()
                # prefetcher counts greater than 2 will be ignored by simulator
                instr_id = instr_id[:, -1, 0]
                prefetches.extend(list(zip(instr_id.tolist(), target_addr)))
                bar.update(1)
                if i % 100 == 99:
                    avg_loss /= 100
                    loss_list.append(avg_loss)
                    print(f"Iter{i} avg_loss: {avg_loss}")
                    avg_loss = 0
        self.log_loss(loss_list, "test")
        bar.close()
        return prefetches


class AttentionRegressor(nn.Module):
    def __init__(self, n_features: int, n_heads: int, hidden_dim: int):
        super().__init__()
        self.diff_embedding = nn.Linear(n_features, hidden_dim)
        self.regressor = LambdaLayer(
            hidden_dim, dim_k=hidden_dim, heads=n_heads, dim_out=hidden_dim // 2
        )
        self.output = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.diff_embedding(x)
        attention = self.regressor(embedded)
        print(attention.shape)
        return self.output(attention)


class AttentionPrefetcher(MLPrefetchModel):
    def __init__(self, n_features: int, n_heads, hidden_dim: int) -> None:
        super().__init__()
        self.model = AttentionRegressor(n_features, n_heads, hidden_dim).cuda()

    def train(self, train_data: torch.Tensor):
        labels = train_data[1:, -1]
        loader = DataLoader(
            TensorDataset(train_data[:-1], labels), batch_size=config.BATCH_SIZE
        )
        opt = optim.Adam(self.model.parameters(), lr=1e-3).cuda()  # type: ignore
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

Model = TransformerModelPrefetcher 