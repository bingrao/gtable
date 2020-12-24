from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from gtable.app.base import BaseSynthesizer
import numpy as np
from gtable.data.inputter import write_tsv
from gtable.data.transformer import build_transformer
import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential


class Encoder(Module):
    def __init__(self, data_dim, compress_dims, embedding_dim):
        super(Encoder, self).__init__()
        dim = data_dim
        seq = []
        for item in list(compress_dims):
            seq += [
                Linear(dim, item),
                ReLU()
            ]
            dim = item
        self.seq = Sequential(*seq)
        self.fc1 = Linear(dim, embedding_dim)
        self.fc2 = Linear(dim, embedding_dim)

    def forward(self, x):
        feature = self.seq(x)
        mu = self.fc1(feature)
        logvar = self.fc2(feature)
        std = torch.exp(0.5 * logvar)
        return mu, std, logvar


class Decoder(Module):
    def __init__(self, embedding_dim, decompress_dims, data_dim):
        super(Decoder, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(decompress_dims):
            seq += [Linear(dim, item), ReLU()]
            dim = item

        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, x):
        return self.seq(x), self.sigma


class TVAESynthesizer(BaseSynthesizer):
    """TVAESynthesizer."""

    def __init__(
        self,
        ctx,
        embedding_dim=128,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        l2scale=1e-5,
        batch_size=500,
        epochs=300
    ):
        super(TVAESynthesizer, self).__init__(ctx)
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = 2
        self.epochs = self.config.epochs

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, dataset, categorical_columns=tuple(), ordinal_columns=tuple(), **kwargs):
        self.transformer = dataset.transformer
        data = dataset.X

        dataset = TensorDataset(torch.from_numpy(data.astype('float32')).to(self._device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        data_dim = self.transformer.output_dimensions
        encoder = Encoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = Decoder(self.embedding_dim, self.compress_dims, data_dim).to(self._device)
        optimizerAE = Adam(
            list(encoder.parameters()) + list(self.decoder.parameters()),
            weight_decay=self.l2scale)

        for i in range(self.epochs):
            for id_, data in enumerate(loader):
                optimizerAE.zero_grad()
                real = data[0].to(self._device)
                mu, std, logvar = encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec, sigmas = self.decoder(emb)
                loss_1, loss_2 = self.loss_function(rec, real, sigmas, mu, logvar,
                                                    self.transformer.output_info,
                                                    self.loss_factor)
                loss = loss_1 + loss_2
                loss.backward()
                optimizerAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)
            self.logging.info(
                f"Epoch {i} / {self.epochs},\tloss 1: {loss_1:.4f},\tloss 2: {loss_2:.4f}")

    def sample(self, num_samples, **kwargs):
        self.decoder.eval()

        steps = num_samples // self.batch_size + 1
        data = []
        for _ in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self._device)
            fake, sigmas = self.decoder(noise)
            fake = torch.tanh(fake)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:num_samples]
        return self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())

    def set_device(self, device):
        self._device = device
        self.decoder.to(self._device)

    def run(self, dataset=None):

        self.fit(dataset)

        if self.config.save is not None:
            self.save(self.config.save)

        num_samples = self.config.num_samples or len(dataset.X)

        sampled = self.sample(num_samples)

        if self.config.tsv:
            write_tsv(sampled, self.config.metadata, self.config.output)
        else:
            sampled.to_csv(self.config.output, index=False)

    def loss_function(self, recon_x, x, sigmas, mu, logvar, output_info, factor):
        st = 0
        loss = []
        for item in output_info:
            if item[1] == 'tanh':
                ed = st + item[0]
                std = sigmas[st]
                loss.append(((x[:, st] - torch.tanh(recon_x[:, st])) ** 2 / 2 / (std ** 2)).sum())
                loss.append(torch.log(std) * x.size()[0])
                st = ed
            elif item[1] == 'softmax':
                ed = st + item[0]
                loss.append(cross_entropy(
                    recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'))
                st = ed
            else:
                assert 0

        assert st == recon_x.size()[1]
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return sum(loss) * factor / x.size()[0], KLD / x.size()[0]
