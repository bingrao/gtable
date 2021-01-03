from torch.nn import BatchNorm1d, Linear, Module, ReLU, Sequential, LayerNorm
from gtable.app.gtable.attention import MultiHeadedAttention
from gtable.app.gtable.attention import PositionwiseFeedForward
from gtable.utils.misc import ClassRegistry
import numpy as np
import torch


class ConditionalGenerator(object):
    def __init__(self, data, output_info, log_frequency, opt=None):
        self.model = []
        self.config = opt

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

                # The indices of the maximum values along an row axis
                self.model.append(np.argmax(data[:, start:end], axis=-1))

                start = end
            else:
                assert 0

        assert start == data.shape[1]

        self.interval = []
        self.n_col = 0  # nums of categorial columns
        self.n_opt = 0  # nums of dimentions for all categorial columns
        skip = False
        start = 0
        self.p = np.zeros((counter, max_interval))  # (nums_categorial, max_size_one_hot)
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
                tmp = np.sum(data[:, start:end], axis=0)  # Sum along with column axis
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
        # idx: 1D (batch_size, );
        # a (batch_size, max_internal)
        a = self.p[idx]

        # random values (batch_size, 1)
        r = np.expand_dims(np.random.rand(a.shape[0]), axis=1)

        # cumsum: Return the cumulative sum of the elements along a given axis.
        # argmax: Returns the indices of the maximum values along an axis.
        # return: 1D (batch_size, )
        return (a.cumsum(axis=1) > r).argmax(axis=1)

    def sample(self, batch):
        if self.n_col == 0:
            return None

        # Generating a [batch]-size samples from "np.arange(self.n_col)",
        # idx.shape: (batch_size, )
        idx = np.random.choice(np.arange(self.n_col), batch)

        vec1 = np.zeros((batch, self.n_opt), dtype='float32')  # (batch_size, self.n_opt)
        mask1 = np.zeros((batch, self.n_col), dtype='float32')  # (batch_size, self.n_col)

        mask1[np.arange(batch), idx] = 1
        # opt1prime.shape 1D (batch_size, )
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


class Generator(Module):
    def __init__(self, name, input_dim, output_dim, n_col, config):
        """
        (embedding_dim) --> GeneratorLayer () --> GeneratorLayer --> Linear (data_dim)
        GeneratorLayer: Linear --> BatchNorm --> ReLU --> Concatenate
        """
        super(Generator, self).__init__()
        self._name = name
        self._input_dim = input_dim
        self._output_dim = output_dim

        # nums of columns in orginal dataset and conditional datasets
        self.n_col = n_col

        self._config = config

        self.nums_layers = config.gen_layers
        self.layer_dim = config.gen_dim

        self.model = self.build_model()

    @property
    def name(self):
        return self._name

    def build_model(self):
        raise NotImplementedError

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def loss(y_fake):
        return -torch.mean(y_fake)


GENERATOR_REGISTRY = ClassRegistry(base_class=Generator)
register_generator = GENERATOR_REGISTRY.register  # pylint: disable=invalid-name


class StardardGeneratorLayer(Module):
    def __init__(self, input_dim, output_dim):
        super(StardardGeneratorLayer, self).__init__()
        self.fc = Linear(input_dim, output_dim)
        self.bn = BatchNorm1d(output_dim)
        self.relu = ReLU()

    def forward(self, x):
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, x], dim=1)


@register_generator(name="gtable_standard")
class StandardGenerator(Generator):
    def __init__(self, input_dim, output_dim, n_col, opt):
        """
        (embedding_dim) --> GeneratorLayer () --> GeneratorLayer --> Linear (data_dim)
        GeneratorLayer: Linear --> BatchNorm --> ReLU --> Concatenate
        """
        super(StandardGenerator, self).__init__("gtable_standard",
                                                input_dim,
                                                output_dim,
                                                n_col,
                                                opt)

    def build_model(self):
        seq = []
        dim = self._input_dim

        for i in range(self.nums_layers):
            seq += [StardardGeneratorLayer(input_dim=dim,
                                           output_dim=self.layer_dim)]
            dim += self.layer_dim

        seq.append(Linear(dim, self._output_dim))
        return Sequential(*seq)


class AttentionGeneratorLayer(Module):
    def __init__(self, input_dim, output_dim, head, n_col, dropout=0.1):
        super(AttentionGeneratorLayer, self).__init__()
        self.layer_norm = LayerNorm(input_dim, eps=1e-6)
        self.attention = MultiHeadedAttention(n_col, head, input_dim)
        self.feed_forward = PositionwiseFeedForward(input_dim, 512, output_dim, dropout)

    def forward(self, x):
        out = self.layer_norm(x)
        out = self.attention(out)
        out = self.feed_forward(out)
        return torch.cat([out, x], dim=1)


@register_generator(name="gtable_attention")
class AttentionGenerator(Generator):
    def __init__(self, input_dim, output_dim, n_col, opt):
        """
        (embedding_dim) --> GeneratorLayer () --> GeneratorLayer --> Linear (data_dim)
        GeneratorLayer: Linear --> BatchNorm --> ReLU --> Concatenate
        """
        self.h = opt.head
        self.dropout = opt.dropout
        super(AttentionGenerator, self).__init__("gtable_attention",
                                                 input_dim,
                                                 output_dim,
                                                 n_col,
                                                 opt)

    def build_model(self):
        seq = []
        dim = self._input_dim
        n_col = self.n_col

        for i in range(self.nums_layers):
            seq += [AttentionGeneratorLayer(input_dim=dim,
                                            output_dim=self.layer_dim,
                                            head=self.h,
                                            n_col=n_col,
                                            dropout=self.dropout)]
            dim += self.layer_dim
            n_col += self.n_col

        seq.append(Linear(dim, self._output_dim))
        return Sequential(*seq)


def get_generator(name):
    _class = GENERATOR_REGISTRY.get(name.lower())
    if _class is None:
        raise ValueError("No Embedding model associated with the name: {}".format(name))
    else:
        return _class
