from torch.nn import Dropout, LeakyReLU, Linear, Module, Sequential, LayerNorm
from gtable.utils.misc import ClassRegistry
from gtable.app.gtable.transformer import TransformerEncoder, SelfAttention, FeedForward
import torch


class Discriminator(Module):
    def __init__(self, name, input_dim, output_dim, n_col, opt, metadata=None):
        """
        (input_dim * pack) --> DiscriminatorLayer --> DiscriminatorLayer --> Linear--> (1)
        DiscriminatorLayer: Linear --> LeakyReLU --> Dropout
        """
        super(Discriminator, self).__init__()
        self._name = name
        self.pack = opt.dis_pack
        self._input_dim = input_dim * self.pack
        self._output_dim = output_dim
        self.metadata = metadata
        # nums of columns in orginal dataset and conditional dataset
        self.n_col = n_col

        self.config = opt

        self.nums_layers = opt.dis_layers
        self.layer_dim = opt.dis_dim

        self.model = self.build_model()

    @property
    def name(self):
        return self._name

    def build_model(self):
        raise NotImplementedError

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

    @staticmethod
    def loss(y_real, y_fake):
        return -(torch.mean(y_real) - torch.mean(y_fake))


DISCRIMINATOR_REGISTRY = ClassRegistry(base_class=Discriminator)
register_disc = DISCRIMINATOR_REGISTRY.register  # pylint: disable=invalid-name


class StandardDiscriminatorLayer(Module):
    def __init__(self, input_dim, output_dim):
        super(StandardDiscriminatorLayer, self).__init__()
        self.fc = Linear(input_dim, output_dim)
        self.leakyReLu = LeakyReLU(0.2)
        self.dropout = Dropout(0.5)

    def forward(self, x):
        out = self.fc(x)
        out = self.leakyReLu(out)
        return self.dropout(out)


@register_disc(name="gtable_standard")
class StandardDiscriminator(Discriminator):
    def __init__(self, input_dim, output_dim, n_col, opt, metadata=None):
        """
        (input_dim * pack) --> DiscriminatorLayer --> DiscriminatorLayer --> Linear--> (1)
        DiscriminatorLayer: Linear --> LeakyReLU --> Dropout
        """
        super(StandardDiscriminator, self).__init__("gtable_standard",
                                                    input_dim,
                                                    output_dim,
                                                    n_col,
                                                    opt,
                                                    metadata)

    def build_model(self):
        dim = self._input_dim
        seq = []
        for i in range(self.nums_layers):
            seq += [StandardDiscriminatorLayer(input_dim=dim,
                                               output_dim=self.layer_dim)]
            dim = self.layer_dim

        seq += [Linear(dim, self._output_dim)]
        return Sequential(*seq)


class AttentionDiscriminatorLayer(Module):
    def __init__(self, input_dim, output_dim, head, n_col, dropout=0.1):
        super(AttentionDiscriminatorLayer, self).__init__()
        self.layer_norm_1 = LayerNorm(input_dim, eps=1e-6)
        self.layer_norm_2 = LayerNorm(input_dim, eps=1e-6)
        self.attention = SelfAttention(input_dim, 128, n_col, head)
        self.feed_forward = FeedForward(input_dim, 512, output_dim, dropout, False)

        self.fc = Linear(input_dim, output_dim)

        self.d_ff = 512
        self.fc_1 = Linear(input_dim, self.d_ff)
        self.fc_2 = Linear(self.d_ff, output_dim)

        self.leakyReLu = LeakyReLU(0.2)
        self.dropout = Dropout(dropout)
        self.dropout_1 = Dropout(dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.fc(out)
        out = self.leakyReLu(out)
        return self.dropout(out)

    # def forward(self, x):
    #     out = self.layer_norm_1(x)
    #     # out = self.attention(out)
    #     out = self.dropout(self.leakyReLu(self.fc_1(out)))
    #     out = self.dropout_1(self.fc_2(out))
    #     return out


@register_disc(name="gtable_attention")
class AttentionDiscriminator(Discriminator):
    def __init__(self, input_dim, output_dim, n_col, opt, metadata=None):
        """
        (input_dim * pack) --> DiscriminatorLayer --> DiscriminatorLayer --> Linear--> (1)
        DiscriminatorLayer: Linear --> LeakyReLU --> Dropout
        """

        self.h = opt.head
        self.dropout = opt.dropout
        super(AttentionDiscriminator, self).__init__("gtable_attention",
                                                     input_dim,
                                                     output_dim,
                                                     n_col,
                                                     opt,
                                                     metadata)

    def build_model(self):
        dim = self._input_dim
        seq = []
        n_col = self.n_col
        for i in range(self.nums_layers):
            seq += [AttentionDiscriminatorLayer(input_dim=dim,
                                                output_dim=self.layer_dim,
                                                head=self.h,
                                                n_col=n_col,
                                                dropout=self.dropout)]
            dim = self.layer_dim

        seq += [Linear(dim, self._output_dim)]
        return Sequential(*seq)


@register_disc(name="gtable_transformer")
class TransformerDiscriminator(Discriminator):
    def __init__(self, input_dim, output_dim, n_col, opt, metadata=None):
        """
        (input_dim * pack) --> DiscriminatorLayer --> DiscriminatorLayer --> Linear--> (1)
        DiscriminatorLayer: Linear --> LeakyReLU --> Dropout
        """

        self.h = opt.head
        self.dropout = opt.dropout
        super(TransformerDiscriminator, self).__init__("gtable_transformer",
                                                       input_dim,
                                                       output_dim,
                                                       n_col,
                                                       opt,
                                                       metadata)

    def build_model(self):
        return TransformerEncoder(input_dim=self._input_dim,
                                  output_dim=self._output_dim,
                                  n_col=self.n_col,
                                  opt=self.config,
                                  metadata=self.metadata,
                                  is_generator=False)
    # def build_model(self):
    #     dim = self._input_dim
    #     seq = []
    #     n_col = self.n_col
    #     for i in range(self.nums_layers):
    #         seq += [AttentionDiscriminatorLayer(input_dim=dim,
    #                                             output_dim=self.layer_dim,
    #                                             head=self.h,
    #                                             n_col=n_col,
    #                                             dropout=self.dropout)]
    #         dim = self.layer_dim
    #
    #     seq += [Linear(dim, self._output_dim)]
    #     return Sequential(*seq)


def get_discriminator(name):
    _class = DISCRIMINATOR_REGISTRY.get(name.lower())
    if _class is None:
        raise ValueError("No Embedding model associated with the name: {}".format(name))
    else:
        return _class
