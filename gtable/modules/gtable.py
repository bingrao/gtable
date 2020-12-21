import torch
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential
import numpy as np
import torch.nn as nn
import copy


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class DiscriminatorLayer(Module):
    def __init__(self, i, o):
        super(DiscriminatorLayer, self).__init__()
        self.fc = Linear(i, o)
        self.leakyReLu = LeakyReLU(0.2)
        self.dropout = Dropout(0.5)

    def forward(self, x):
        out = self.fc(x)
        out = self.leakyReLu(out)
        return self.dropout(out)


class Discriminator(Module):
    def __init__(self, input_dim, output_dim, opt):
        """
        (input_dim * pack) --> DiscriminatorLayer --> DiscriminatorLayer --> Linear--> (1)
        DiscriminatorLayer: Linear --> LeakyReLU --> Dropout
        """
        super(Discriminator, self).__init__()
        self.pack = opt.dis_pack
        self._input_dim = input_dim * self.pack
        self._output_dim = output_dim
        self.nums_layers = opt.dis_layers
        self.layer_dim = opt.dis_dim
        self.model = self.build_model()

    def build_model(self):
        dim = self._input_dim
        seq = []
        for i in range(self.nums_layers):
            seq += [DiscriminatorLayer(dim, self.layer_dim)]
            dim = self.layer_dim
        seq += [Linear(dim, self._output_dim)]
        return Sequential(*seq)

    def forward(self, x):
        assert x.size()[0] % self.pack == 0
        return self.model(x.view(-1, self._input_dim))

    def gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.view(-1, pac * real_data.size(1))
                             .norm(2, dim=1) - 1) ** 2).mean() * lambda_

        return gradient_penalty

    def loss(self, y_real, y_fake):
        return -(torch.mean(y_real) - torch.mean(y_fake))


class GeneratorLayer(Module):
    def __init__(self, i, o):
        super(GeneratorLayer, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, x):
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, x], dim=1)


class Generator(Module):
    def __init__(self, input_dim, output_dim, opt):
        """
        (embedding_dim) --> GeneratorLayer () --> GeneratorLayer --> Linear (data_dim)
        GeneratorLayer: Linear --> BatchNorm --> ReLU --> Concatenate
        """
        super(Generator, self).__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.nums_layers = opt.gen_layers
        self.layer_dim = opt.gen_dim
        self.model = self.build_model()

    def build_model(self):
        seq = []
        dim = self._input_dim
        for i in range(self.nums_layers):
            seq += [GeneratorLayer(dim, self.layer_dim)]
            dim += self.layer_dim
        seq.append(Linear(dim, self._output_dim))
        return Sequential(*seq)

    def forward(self, x):
        return self.model(x)

    def loss(self, y_fake):
        return -torch.mean(y_fake)


class ConditionalGenerator(object):
    def __init__(self, data, output_info, log_frequency, opt=None):
        self.model = []

        start = 0
        skip = False
        max_interval = 0
        counter = 0
        for item in output_info:
            if item[1] == 'tanh':
                start += item[0]
                skip = True
                continue
            elif item[1] == 'softmax':
                if skip:
                    skip = False
                    start += item[0]
                    continue

                end = start + item[0]
                max_interval = max(max_interval, end - start)
                counter += 1
                self.model.append(np.argmax(data[:, start:end], axis=-1))
                start = end
            else:
                assert 0

        assert start == data.shape[1]

        self.interval = []
        self.n_col = 0
        self.n_opt = 0
        skip = False
        start = 0
        self.p = np.zeros((counter, max_interval))
        for item in output_info:
            if item[1] == 'tanh':
                skip = True
                start += item[0]
                continue
            elif item[1] == 'softmax':
                if skip:
                    start += item[0]
                    skip = False
                    continue
                end = start + item[0]
                tmp = np.sum(data[:, start:end], axis=0)
                if log_frequency:
                    tmp = np.log(tmp + 1)
                tmp = tmp / np.sum(tmp)
                self.p[self.n_col, :item[0]] = tmp
                self.interval.append((self.n_opt, item[0]))
                self.n_opt += item[0]
                self.n_col += 1
                start = end
            else:
                assert 0

        self.interval = np.asarray(self.interval)

    def random_choice_prob_index(self, idx):
        a = self.p[idx]
        r = np.expand_dims(np.random.rand(a.shape[0]), axis=1)
        return (a.cumsum(axis=1) > r).argmax(axis=1)

    def sample(self, batch):
        if self.n_col == 0:
            return None

        batch = batch
        idx = np.random.choice(np.arange(self.n_col), batch)

        vec1 = np.zeros((batch, self.n_opt), dtype='float32')
        mask1 = np.zeros((batch, self.n_col), dtype='float32')
        mask1[np.arange(batch), idx] = 1
        opt1prime = self.random_choice_prob_index(idx)
        opt1 = self.interval[idx, 0] + opt1prime
        vec1[np.arange(batch), opt1] = 1

        return vec1, mask1, idx, opt1prime

    def sample_zero(self, batch):
        if self.n_col == 0:
            return None

        vec = np.zeros((batch, self.n_opt), dtype='float32')
        idx = np.random.choice(np.arange(self.n_col), batch)
        for i in range(batch):
            col = idx[i]
            pick = int(np.random.choice(self.model[col]))
            vec[i, pick + self.interval[col, 0]] = 1

        return vec

    def generate_cond_from_condition_column_info(self, condition_info, batch):
        vec = np.zeros((batch, self.n_opt), dtype='float32')
        ind = self.interval[condition_info["discrete_column_id"]][0] + condition_info["value_id"]
        vec[:, ind] = 1
        return vec
