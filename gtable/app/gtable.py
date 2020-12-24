from gtable.app.base import BaseSynthesizer
from torch.nn import functional
from gtable.data.sampler import RandomSampler
from gtable.data.inputter import write_tsv
from gtable.utils.misc import pbar
from gtable.utils.optimizers import build_torch_optimizer
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


class GTABLESynthesizer(BaseSynthesizer):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.

    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.
    """

    def __init__(self, ctx):
        super(GTABLESynthesizer, self).__init__(ctx)

        self.noise_dim = self.config.noise_dim
        self.batch_size = self.config.batch_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def build_model(self, log_frequency=True):

        self.data_sampler = RandomSampler(self.train_data, self.transformer.output_info)

        self.cond_generator = ConditionalGenerator(self.train_data,
                                                   self.transformer.output_info,
                                                   log_frequency,
                                                   self.config)

        self.generator = Generator(self.noise_dim + self.cond_generator.n_opt,
                                   self.data_dim, self.config).to(self.device)

        self.discriminator = Discriminator(self.data_dim + self.cond_generator.n_opt,
                                           1, self.config).to(self.device)

        self.config.learning_rate_decay = 1e-6
        self.optimizerG = build_torch_optimizer(self.generator, self.config)

        self.config.learning_rate_decay = 0
        self.optimizerD = build_torch_optimizer(self.discriminator, self.config)

    def discriminator_train_step(self, fakez):
        # 1. Reset gradients
        self.optimizerD.zero_grad()

        # 2. Sample noise and generate fake data
        condvec = self.cond_generator.sample(self.batch_size)
        if condvec is None:
            c1, m1, col, opt = None, None, None, None
            real = self.data_sampler.sample(self.batch_size, col, opt)
            c2 = c1
        else:
            c1, m1, col, opt = condvec
            c1 = torch.from_numpy(c1).to(self.device)
            # m1 = torch.from_numpy(m1).to(self.device)
            fakez = torch.cat([fakez, c1], dim=1)

            perm = np.arange(self.batch_size)
            np.random.shuffle(perm)
            real = self.data_sampler.sample(self.batch_size, col[perm], opt[perm])
            c2 = c1[perm]

        fake = self.generator(fakez)
        fakeact = self._apply_activate(fake)

        real = torch.from_numpy(real.astype('float32')).to(self.device)

        if c1 is not None:
            fake_cat = torch.cat([fakeact, c1], dim=1)
            real_cat = torch.cat([real, c2], dim=1)
        else:
            real_cat = real
            fake_cat = fake

        # 3. Sample noise and generate fake data
        y_real = self.discriminator(real_cat)
        y_fake = self.discriminator(fake_cat)

        # 4. Calculate error
        pen = self.config.g_penalty * self.discriminator.gradient_penalty(real_cat,
                                                                          fake_cat,
                                                                          self.device)
        loss_d = self.discriminator.loss(y_real, y_fake)

        # 5. loss backpropagate
        pen.backward(retain_graph=True)
        loss_d.backward()
        self.optimizerD.step()

        return loss_d + pen

    def generator_train_step(self, fakez):
        # Reset gradients
        self.optimizerG.zero_grad()

        # 2. Sample noise and generate fake data
        condvec = self.cond_generator.sample(self.batch_size)
        if condvec is None:
            c1, m1, col, opt = None, None, None, None
        else:
            c1, m1, col, opt = condvec
            c1 = torch.from_numpy(c1).to(self.device)
            m1 = torch.from_numpy(m1).to(self.device)
            fakez = torch.cat([fakez, c1], dim=1)

        fake = self.generator(fakez)
        fakeact = self._apply_activate(fake)

        if c1 is not None:
            y_fake = self.discriminator(torch.cat([fakeact, c1], dim=1))
        else:
            y_fake = self.discriminator(fakeact)

        if condvec is None:
            cross_entropy = 0
        else:
            cross_entropy = self._cond_loss(fake, c1, m1)

        # 4. Calculate error
        loss_g = self.generator.loss(y_fake) + cross_entropy

        # 5. loss backpropagate
        loss_g.backward()
        self.optimizerG.step()

        return loss_g

    def train(self):
        assert self.batch_size % 2 == 0
        mean = torch.zeros(self.batch_size, self.noise_dim, device=self.device)
        std = mean + 1

        steps_per_epoch = max(len(self.train_data) // self.batch_size, 1)
        for epoch in range(self.config.epochs):
            bar = pbar(self.num_samples, self.config.batch_size, epoch, self.config.epochs)
            for step in range(steps_per_epoch):
                loss_g = self.generator_train_step(torch.normal(mean=mean, std=std))

                for _ in range(self.config.n_critic):
                    loss_d = self.discriminator_train_step(torch.normal(mean=mean, std=std))

                bar.postfix['g_loss'] = f'{loss_g.detach().cpu():6.3f}'
                bar.postfix['d_loss'] = f'{loss_d.detach().cpu():6.3f}'
                bar.update(self.config.batch_size)

            bar.close()
            del bar

        if self.config.save is not None:
            self.save(self.config.save)

    def _apply_activate(self, data):
        data_t = []
        st = 0
        for item in self.transformer.output_info:
            if item[1] == 'tanh':
                ed = st + item[0]
                data_t.append(torch.tanh(data[:, st:ed]))
                st = ed
            elif item[1] == 'softmax':
                ed = st + item[0]
                data_t.append(functional.gumbel_softmax(data[:, st:ed], tau=0.2))
                st = ed
            else:
                assert 0

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        loss = []
        st = 0
        st_c = 0
        skip = False
        for item in self.transformer.output_info:
            if item[1] == 'tanh':
                st += item[0]
                skip = True

            elif item[1] == 'softmax':
                if skip:
                    skip = False
                    st += item[0]
                    continue

                ed = st + item[0]
                ed_c = st_c + item[0]
                tmp = functional.cross_entropy(
                    data[:, st:ed],
                    torch.argmax(c[:, st_c:ed_c], dim=1),
                    reduction='none'
                )
                loss.append(tmp)
                st = ed
                st_c = ed_c

            else:
                assert 0

        loss = torch.stack(loss, dim=1)

        return (loss * m).sum() / data.size()[0]

    def sample(self, num_samples, condition_column=None, condition_value=None):
        """Sample data similar to the training data.

        Args:
            num_samples (int):
                Number of rows to sample.
            condition_column:
            condition_value:
        Returns:
            numpy.ndarray or pandas.DataFrame
        """

        if condition_column is not None and condition_value is not None:
            condition_info = self.transformer.covert_column_name_value_to_id(
                condition_column, condition_value)
            global_condition_vec = self.cond_generator.generate_cond_from_condition_column_info(
                condition_info, self.batch_size)
        else:
            global_condition_vec = None

        steps = num_samples // self.batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self.batch_size, self.noise_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self.device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self.cond_generator.sample_zero(self.batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self.device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self.generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:num_samples]

        return self.transformer.inverse_transform(data, None)

    def save(self, path):
        assert hasattr(self, "generator")
        assert hasattr(self, "discriminator")
        assert hasattr(self, "transformer")

        # always save a cpu model.
        device_bak = self.device
        self.device = torch.device("cpu")
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        torch.save(self, path)

        self.device = device_bak
        self.generator.to(self.device)
        self.discriminator.to(self.device)

    @classmethod
    def load(cls, path):
        model = torch.load(path)
        model.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.generator.to(model.device)
        model.discriminator.to(model.device)
        return model

    def fit(self, dataset, categorical_columns=tuple(), ordinal_columns=tuple(), **kwargs):
        self.transformer = dataset.transformer

        # numpy array [nums_samples, dim] (32561, 157)
        self.train_data = dataset.X

        self.num_samples = dataset.num_samples

        self.data_dim = self.transformer.output_dimensions

        self.build_model()

        self.train()

    def run(self, dataset):
        assert dataset is not None

        self.fit(dataset)

        if self.config.sample_condition_column is not None:
            assert self.config.sample_condition_column_value is not None

        fake_data, org_fake_data = self.sample(self.num_samples,
                                               self.config.sample_condition_column,
                                               self.config.sample_condition_column_value)

        if self.config.tsv:
            write_tsv(org_fake_data, self.config.metadata, self.config.output)
        else:
            org_fake_data.to_csv(self.config.output, index=False)

