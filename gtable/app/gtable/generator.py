from torch.nn import BatchNorm1d, Linear, Module, ReLU, Sequential, LayerNorm
from gtable.app.gtable.attention import MultiHeadedAttention
from gtable.app.gtable.attention import PositionwiseFeedForward
from gtable.app.gtable.transformer import TransformerEncoder
from gtable.utils.misc import ClassRegistry
import numpy as np
import torch


class ConditionalGenerator(object):
    def __init__(self, data, output_info, log_frequency, opt=None):
        self.data = data
        self.output_info = output_info
        self.log_frequency = log_frequency
        self.config = opt

        self.sample_model = []
        self.p = []     # For a record of categorial element, the probability of of each category
        self.n_col = 0  # nums of categorial columns
        self.n_opt = 0  # nums of dimentions for all categorial columns
        self.interval = [] # The interval dim of each categorial one-hot vector in [self.n_opt]
        self.build_sample_model()

    def build_sample_model(self):
        start = 0
        skip = False
        max_interval = 0  # The max dim for one-hot vector (categorial)
        counter = 0  # The number of one-hot vector (categorial)
        for item in self.output_info:
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
                self.sample_model.append(np.argmax(self.data[:, start:end], axis=-1))

                start = end
            else:
                assert 0

        assert start == self.data.shape[1]

        self.p = np.zeros((counter, max_interval))  # (nums_categorial, max_size_one_hot)

        interval = []
        skip = False
        start = 0
        for item in self.output_info:
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
                tmp = np.sum(self.data[:, start:end], axis=0)  # Sum along with column axis (dim_one_hot,)
                if self.log_frequency:
                    tmp = np.log(tmp + 1)
                tmp = tmp / np.sum(tmp)
                self.p[self.n_col, :item[0]] = tmp
                interval.append((self.n_opt, item[0])) # The interval of each categorial one-hot vector in [self.n_opt]
                self.n_opt += item[0]
                self.n_col += 1
                start = end
            else:
                assert 0

        self.interval = np.asarray(interval)

    def random_choice_prob_index(self, idx):
        # idx: 1D (batch_size, );
        # Select probability according the input idx. a (batch_size, max_internal)
        a = self.p[idx]

        # random values (batch_size, 1)
        r = np.expand_dims(np.random.rand(a.shape[0]), axis=1)

        # cumsum: Return the cumulative sum of the elements along a given axis.
        # argmax: Returns the indices of the maximum values along an axis.
        # return: 1D (batch_size, ), batch_size array in which an
        # elemement indicates which category is chosen
        return (a.cumsum(axis=1) > r).argmax(axis=1)

    def sample(self, batch):
        if self.n_col == 0:
            return None

        # Generating a [batch]-size samples from "np.arange(self.n_col)",
        # idx.shape: (batch_size, )
        idx = np.random.choice(np.arange(self.n_col), batch)

        vec = np.zeros((batch, self.n_opt), dtype='float32')  # (batch_size, self.n_opt)
        mask = np.zeros((batch, self.n_col), dtype='float32')  # (batch_size, self.n_col)

        mask[np.arange(batch), idx] = 1
        # optprime.shape 1D (batch_size, )
        optprime = self.random_choice_prob_index(idx)

        # self.interval[idx][:, 0]  == self.interval[idx, 0]
        vec[np.arange(batch), self.interval[idx][:, 0] + optprime] = 1

        """
        Return: A batch of categorial columns is selected. For each element in a batch,
                it is a [self.n_opt]-dimension size vector, but only one categorial
                column is selected.

            vec: [Batch_size, self.n_opt], e.g. (500, 103)
                 Selected categorial column vector representation
            mask: [Batch_size, self.n_col], e.g. (500, 9)
                 indicates which categorial column is selected
            idx: [Batch_size, ], e.g. (500,)
                 The index of catogorial column is selected
            optprime: [Batch_size, ], e.g. (500,)
                 The index of category is selected for a categorial column
        """

        return vec, mask, idx, optprime

    def sample_zero(self, batch):
        if self.n_col == 0:
            return None

        vec = np.zeros((batch, self.n_opt), dtype='float32')
        idx = np.random.choice(np.arange(self.n_col), batch)
        for i in range(batch):
            col = idx[i]
            pick = int(np.random.choice(self.sample_model[col]))
            vec[i, pick + self.interval[col][:, 0]] = 1

        return vec

    def generate_cond_from_condition_column_info(self, condition_info, batch):
        vec = np.zeros((batch, self.n_opt), dtype='float32')
        ind = self.interval[condition_info["discrete_column_id"]][0] + condition_info["value_id"]
        vec[:, ind] = 1
        return vec


class Generator(Module):
    def __init__(self, name, input_dim, output_dim, n_col, config, metadata):
        """
        (embedding_dim) --> GeneratorLayer () --> GeneratorLayer --> Linear (data_dim)
        GeneratorLayer: Linear --> BatchNorm --> ReLU --> Concatenate
        """
        super(Generator, self).__init__()
        self._name = name
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.metadata = metadata

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
    def __init__(self, input_dim, output_dim, n_col, opt, metadata=None):
        """
        (embedding_dim) --> GeneratorLayer () --> GeneratorLayer --> Linear (data_dim)
        GeneratorLayer: Linear --> BatchNorm --> ReLU --> Concatenate
        """
        super(StandardGenerator, self).__init__("gtable_standard",
                                                input_dim,
                                                output_dim,
                                                n_col,
                                                opt,
                                                metadata)

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
    def __init__(self, input_dim, output_dim, n_col, opt, metadata=None):
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
                                                 opt,
                                                 metadata)

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


@register_generator(name="gtable_transformer")
class TransformerGenerator(Generator):
    def __init__(self, input_dim, output_dim, n_col, opt, metadata=None):
        """
        (embedding_dim) --> GeneratorLayer () --> GeneratorLayer --> Linear (data_dim)
        GeneratorLayer: Linear --> BatchNorm --> ReLU --> Concatenate
        """
        self.h = opt.head
        self.dropout = opt.dropout
        super(TransformerGenerator, self).__init__("gtable_transformer",
                                                   input_dim,
                                                   output_dim,
                                                   n_col,
                                                   opt,
                                                   metadata)

    def build_model(self):
        return TransformerEncoder(input_dim=self._input_dim,
                                  output_dim=self._output_dim,
                                  n_col=self.n_col,
                                  opt=self._config,
                                  metadata=self.metadata,
                                  is_generator=True)

    # def build_model(self):
    #     seq = []
    #     dim = self._input_dim
    #     n_col = self.n_col
    #
    #     for i in range(self.nums_layers):
    #         seq += []
    #         dim += self.layer_dim
    #         n_col += self.n_col
    #
    #     seq.append(Linear(dim, self._output_dim))
    #     return Sequential(*seq)


def get_generator(name):
    _class = GENERATOR_REGISTRY.get(name.lower())
    if _class is None:
        raise ValueError("No Embedding model associated with the name: {}".format(name))
    else:
        return _class
