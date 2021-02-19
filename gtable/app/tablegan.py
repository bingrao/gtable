#!/usr/bin/env python
# coding: utf-8
import numpy as np
import torch
from torch.nn import (
    BatchNorm2d, Conv2d, ConvTranspose2d, LeakyReLU, Module, ReLU, Sequential, Sigmoid, Tanh, init)
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from gtable.utils.constants import CATEGORICAL
from gtable.app.base import BaseSynthesizer


class Discriminator(Module):
    def __init__(self, meta, side, layers):
        super(Discriminator, self).__init__()
        self.meta = meta
        self.side = side
        self.seq = Sequential(*layers)
        # self.layers = layers

    def forward(self, input):
        return self.seq(input)


class Generator(Module):
    def __init__(self, meta, side, layers):
        super(Generator, self).__init__()
        self.meta = meta
        self.side = side
        self.seq = Sequential(*layers)
        # self.layers = layers

    def forward(self, x):
        return self.seq(x)


class Classifier(Module):
    def __init__(self, meta, side, layers, device):
        super(Classifier, self).__init__()
        self.meta = meta
        self.side = side
        self.seq = Sequential(*layers)
        self.valid = True
        if meta[-1]['name'] != 'label' or meta[-1]['type'] != CATEGORICAL or meta[-1]['size'] != 2:
            self.valid = False

        masking = np.ones((1, 1, side, side), dtype='float32')
        index = len(self.meta) - 1
        self.r = index // side
        self.c = index % side
        masking[0, 0, self.r, self.c] = 0
        self.masking = torch.from_numpy(masking).to(device)

    def forward(self, input):
        label = (input[:, :, self.r, self.c].view(-1) + 1) / 2
        input = input * self.masking.expand(input.size())
        return self.seq(input).view(-1), label


def determine_layers(side, random_dim, num_channels):

    assert 4 <= side <= 32

    layer_dims = [(1, side), (num_channels, side // 2)]

    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

    layers_D = []
    for prev, curr in zip(layer_dims, layer_dims[1:]):
        layers_D += [
            Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),
            BatchNorm2d(curr[0]),
            LeakyReLU(0.2, inplace=True)
        ]
    layers_D += [
        Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0),
        Sigmoid()
    ]

    layers_G = [
        ConvTranspose2d(
            random_dim, layer_dims[-1][0], layer_dims[-1][1], 1, 0, output_padding=0, bias=False)
    ]

    for prev, curr in zip(reversed(layer_dims), reversed(layer_dims[:-1])):
        layers_G += [
            BatchNorm2d(prev[0]),
            ReLU(True),
            ConvTranspose2d(prev[0], curr[0], 4, 2, 1, output_padding=0, bias=True)
        ]
    layers_G += [Tanh()]

    layers_C = []
    for prev, curr in zip(layer_dims, layer_dims[1:]):
        layers_C += [
            Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),
            BatchNorm2d(curr[0]),
            LeakyReLU(0.2, inplace=True)
        ]

    layers_C += [Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0)]

    return layers_D, layers_G, layers_C


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0)


class TableganSynthesizer(BaseSynthesizer):
    """docstring for TableganSynthesizer??"""

    def __init__(self, ctx):
        super(TableganSynthesizer, self).__init__(ctx)

        self.random_dim = self.config.noise_dim
        self.num_channels = self.config.num_channels
        self.l2scale = self.config.learning_rate

        self.batch_size = self.config.batch_size
        self.epochs = self.config.epochs

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, data, categorical_columns=tuple(), ordinal_columns=tuple(), **kwargs):
        sides = [4, 8, 16, 24, 32]
        for i in sides:
            if i * i >= data.num_columns:
                self.side = i
                break

        self.transformer = data.transformer

        self.transformer.fit(data.train_dataset, categorical_columns, ordinal_columns)

        data = self.transformer.transform(data.train_dataset)

        data = torch.from_numpy(data.astype('float32')).to(self.device)
        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        layers_D, layers_G, layers_C = determine_layers(self.side,
                                                        self.random_dim,
                                                        self.num_channels)

        self.generator = Generator(self.transformer.meta,
                                   self.side,
                                   layers_G).to(self.device)
        self.discriminator = Discriminator(self.transformer.meta,
                                           self.side,
                                           layers_D).to(self.device)
        self.classifier = Classifier(self.transformer.meta,
                                     self.side,
                                     layers_C,
                                     self.device).to(self.device)

        optimizer_params = dict(lr=2e-4, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.l2scale)
        optimizerG = Adam(self.generator.parameters(), **optimizer_params)
        optimizerD = Adam(self.discriminator.parameters(), **optimizer_params)
        optimizerC = Adam(self.classifier.parameters(), **optimizer_params)

        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)
        self.classifier.apply(weights_init)

        for i in range(self.epochs):
            for id_, data in enumerate(loader):
                real = data[0].to(self.device)
                noise = torch.randn(self.batch_size, self.random_dim, 1, 1, device=self.device)
                fake = self.generator(noise)

                optimizerD.zero_grad()
                y_real = self.discriminator(real)
                y_fake = self.discriminator(fake)
                loss_d = (
                    -(torch.log(y_real + 1e-4).mean()) - (torch.log(1. - y_fake + 1e-4).mean()))
                loss_d.backward()
                optimizerD.step()

                noise = torch.randn(self.batch_size, self.random_dim, 1, 1, device=self.device)
                fake = self.generator(noise)
                optimizerG.zero_grad()
                y_fake = self.discriminator(fake)
                loss_g = -(torch.log(y_fake + 1e-4).mean())
                loss_g.backward(retain_graph=True)
                loss_mean = torch.norm(torch.mean(fake, dim=0) - torch.mean(real, dim=0), 1)
                loss_std = torch.norm(torch.std(fake, dim=0) - torch.std(real, dim=0), 1)
                loss_info = loss_mean + loss_std
                loss_info.backward()
                optimizerG.step()

                noise = torch.randn(self.batch_size, self.random_dim, 1, 1, device=self.device)
                fake = self.generator(noise)
                if self.classifier.valid:
                    real_pre, real_label = self.classifier(real)
                    fake_pre, fake_label = self.classifier(fake)

                    loss_cc = binary_cross_entropy_with_logits(real_pre, real_label)
                    loss_cg = binary_cross_entropy_with_logits(fake_pre, fake_label)

                    optimizerG.zero_grad()
                    loss_cg.backward()
                    optimizerG.step()

                    optimizerC.zero_grad()
                    loss_cc.backward()
                    optimizerC.step()
                    loss_c = (loss_cc, loss_cg)
                else:
                    loss_c = None

            self.logging.info(f"Epoch {i} / {self.epochs}, \t loss_d: {loss_d:.4f}, "
                              f"\tloss_g: {loss_g:.4f}, \tloss_c: {loss_c}")

    def sample(self, num_samples, **kwargs):
        self.generator.eval()

        steps = num_samples // self.batch_size + 1
        data = []
        for i in range(steps):
            noise = torch.randn(self.batch_size, self.random_dim, 1, 1, device=self.device)
            fake = self.generator(noise)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        return self.transformer.inverse_transform(data[:num_samples])

    def run(self, dataset):
        self.fit(dataset)
        self.num_samples = len(dataset.train_dataset)
        self.sample(self.num_samples)
