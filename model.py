from abc import ABC, abstractmethod
from torch import nn, optim, torch
import config


class MLPrefetchModel(object):
    '''
    Abstract base class for your models. For HW-based approaches such as the
    NextLineModel below, you can directly add your prediction code. For ML
    models, you may want to use it as a wrapper, but alternative approaches
    are fine so long as the behavior described below is respected.
    '''

    @abstractmethod
    def load(self, path):
        '''
        Loads your model from the filepath path
        '''
        pass

    @abstractmethod
    def save(self, path):
        '''
        Saves your model to the filepath path
        '''
        pass

    @abstractmethod
    def train(self, data):
        '''
        Train your model here. No return value. The data parameter is in the
        same format as the load traces. Namely,
        Unique Instr Id, Cycle Count, Load Address, Instruction Pointer of the Load, LLC hit/miss
        '''
        pass

    @abstractmethod
    def generate(self, data):
        '''
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
        '''
        pass


class NextLineModel(MLPrefetchModel):

    def load(self, path):
        # Load your pytorch / tensorflow model from the given filepath
        print('Loading ' + path + ' for NextLineModel')

    def save(self, path):
        # Save your model to a file
        print('Saving ' + path + ' for NextLineModel')

    def train(self, data):
        '''
        Train your model here using the data

        The data is the same format given in the load traces. Namely:
        Unique Instr Id, Cycle Count, Load Address, Instruction Pointer of the Load, LLC hit/miss
        '''
        print('Training NextLineModel')

    def generate(self, data):
        '''
        Generate the prefetches for the prefetch file for ChampSim here

        As a reminder, no looking ahead in the data and no more than 2
        prefetches per unique instruction ID

        The return format for this function is a list of (instr_id, pf_addr)
        tuples as shown below
        '''
        print('Generating for NextLineModel')
        prefetches = []
        for (instr_id, cycle_count, load_addr, load_ip, llc_hit) in data:
            # Prefetch the next two blocks
            prefetches.append((instr_id, ((load_addr >> 6) + 1) << 12))
            prefetches.append((instr_id, ((load_addr >> 6) + 2) << 12))

        return prefetches


class TimeSeriesLSTM(nn.Module):
    def __init__(self, hidden: int, n_layers: int = 1):
        super().__init__()
        self.rnn = nn.LSTM(input_size=1, hidden_size=hidden,
                           num_layers=n_layers, batch_first = True)
        self.regressor = nn.Linear(hidden, 1)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        latent_sequence, _ = self.rnn(input_seq)
        return self.regressor(latent_sequence)


class TimeSeriesLSTMPrefetcher(MLPrefetchModel):
    def __init__(self, hidden: int, n_layers: int = config.N_LAYERS):
        self.model = TimeSeriesLSTM(hidden, n_layers).double().cuda()
        self.criterion = nn.MSELoss().cuda()
        self.optimizer = optim.Adam(self.model.parameters())
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=0.1)

    def load(self, path):
        self.model = torch.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def train(self, loader, training_set_on_gpu: False):
        self.model.train()
        for i, (seq_batch, label_batch) in enumerate(loader):
            self.optimizer.zero_grad(set_to_none=True)
            model_prediction = self.model(seq_batch.unsqueeze(-1).cuda())[:,-1,:]
            loss = self.criterion(model_prediction, label_batch.unsqueeze(-1).cuda())
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), config.MAX_CLIP)
            self.optimizer.step()
            self.scheduler.step()
            if i % 100 == 99:
                print(f"Iter{i} loss: {loss}")

    def generate(self, data):
        self.model.eval()
        prefetches = []
        for i, (x, _) in enumerate(batch(data, random=False)):
            y_pred = self.model(x)

            for xi, yi in zip(x, y_pred):
                # Where instr_id is a function that extracts the unique instr_id
                prefetches.append((instr_id(xi), yi))

        return prefetches


Model = TimeSeriesLSTMPrefetcher
