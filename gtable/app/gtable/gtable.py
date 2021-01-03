from gtable.app.base import BaseSynthesizer
from torch.nn import functional
from gtable.data.sampler import RandomSampler
from gtable.utils.misc import pbar
from gtable.utils.optimizers import build_torch_optimizer
from torch.autograd import Variable
from gtable.app.gtable.generator import get_generator, ConditionalGenerator
from gtable.app.gtable.discriminator import get_discriminator
import numpy as np
import torch


class GTABLESynthesizer(BaseSynthesizer):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.

    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.
    """

    def __init__(self, ctx):
        super(GTABLESynthesizer, self).__init__(ctx)
        self.model_name = self.config.gtable_model

    def build_model(self, log_frequency=True):

        self.data_sampler = RandomSampler(self.train_data, self.transformer.output_info)

        self.cond_generator = ConditionalGenerator(self.train_data,
                                                   self.transformer.output_info,
                                                   log_frequency,
                                                   self.config)
        n_col = self.num_columns + self.cond_generator.n_col
        self.generator = get_generator(self.model_name)(
            self.noise_dim + self.cond_generator.n_opt,
            self.data_dim, n_col, self.config).to(self.device)

        self.discriminator = get_discriminator(self.model_name)(
            self.data_dim + self.cond_generator.n_opt,
            1, n_col, self.config).to(self.device)

        self.config.learning_rate_decay = 1e-6
        self.optimizerG = build_torch_optimizer(self.generator, self.config)

        self.config.learning_rate_decay = 0
        self.optimizerD = build_torch_optimizer(self.discriminator, self.config)

    def discriminator_train_step(self):
        """
            fakez.shape torch.Size([500, 128])
        """
        fakez = self.get_input_noise()

        # 1. Reset gradients
        self.optimizerD.zero_grad()

        # 2. Sample noise and generate fake data
        condvec = self.cond_generator.sample(self.batch_size)
        if condvec is None:
            c1, m1, col, opt = None, None, None, None
            real = self.data_sampler.sample(self.batch_size, col, opt)
            c2 = c1
        else:
            # c1.shape torch.Size([500, 103]),
            # m1.shape (500, 9),
            # col.shape (500,),
            # opt.shape (500,)
            c1, m1, col, opt = condvec
            c1 = torch.from_numpy(c1).to(self.device)
            # m1 = torch.from_numpy(m1).to(self.device)

            # fakez.shape torch.Size([500, 231])
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

    def generator_train_step(self):
        """
            fakez.shape torch.Size([500, 128])
        """
        fakez = self.get_input_noise()

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

    def get_input_noise(self):

        mean = torch.zeros(self.batch_size,
                           self.noise_dim,
                           device=self.device)
        std = mean + 1
        fakez = torch.normal(mean=mean, std=std)

        if self.config.noise == "gmm":
            g_z = Variable(torch.randn(self.batch_size, self.noise_dim).type(torch.float32))
            g_sig = Variable(torch.randn(self.batch_size, self.noise_dim).type(torch.float32))
            return self.context.mapping_to_cuda(g_z) + \
                   torch.mul(fakez, self.context.mapping_to_cuda(g_sig))
        else:
            return fakez

    def train(self):
        assert self.batch_size % 2 == 0
        loss_d = 0.0
        steps_per_epoch = max(len(self.train_data) // self.batch_size, 1)
        for epoch in range(self.config.epochs):
            bar = pbar(self.num_samples, self.config.batch_size, epoch, self.config.epochs)
            for step in range(steps_per_epoch):
                loss_g = self.generator_train_step()

                for _ in range(self.config.n_critic):
                    loss_d = self.discriminator_train_step()

                bar.postfix['g_loss'] = f'{loss_g.detach().cpu():6.3f}'
                bar.postfix['d_loss'] = f'{loss_d.detach().cpu():6.3f}'
                bar.update(self.config.batch_size)

            bar.close()
            del bar

        if self.config.save is not None:
            self.save(self.config.save)

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

    def sample(self, num_samples, **kwargs):
        """Sample data similar to the training data.

        Args:
            num_samples (int):
                Number of rows to sample.
        Returns:
            numpy.ndarray or pandas.DataFrame
        """

        if self.config.sample_condition_column is not None:
            assert self.config.sample_condition_column_value is not None

        condition_column = self.config.sample_condition_column
        condition_value = self.config.sample_condition_column_value

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
        self.num_columns = dataset.num_columns

        self.transformer.fit(dataset.train_dataset)

        # numpy array [nums_samples, dim] (32561, 157)
        self.train_data = self.transformer.transform(dataset.train_dataset)

        self.num_samples = len(self.train_data)

        self.data_dim = self.transformer.output_dimensions

        self.build_model()

        self.train()

    def run(self, dataset):
        assert dataset is not None

        self.fit(dataset)

        fake_data = self.sample(self.num_samples)
        fake_data.to_csv(self.config.output)
