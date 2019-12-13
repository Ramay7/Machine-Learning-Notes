# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import numpy as np
import os, sys, time
import matplotlib.pyplot as plt

BATCH_SIZE = 256
IMG_SIZE = 28
CHANNEL = 1
IMG_SHAPE = (CHANNEL, IMG_SIZE, IMG_SIZE)

LATENT_DIM = 100 # for generation
HIDDEN_DIM = 256
LAYER_NUM = 3

LEARNING_RATE = 1e-4
BETA = (0.5, 0.999)

DATA_PATH = "../dataset/"
SAVE_PATH = "./images/"
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(SAVE_PATH, exist_ok=True)

EPOCHS = 400
print_interval = 400

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num, activation):
        super(Generator, self).__init__()

        self.layer_num = layer_num
        self.layers = nn.ModuleList()
        self.layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        for i in range(layer_num):
            self.layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim, 0.8),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        self.layers.extend([
                nn.Linear(hidden_dim, output_dim),
                activation
        ])


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        img = x.view(x.size(0), *IMG_SHAPE)
        return img

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, layer_num, activation):
        super(Discriminator, self).__init__()

        self.layer_num = layer_num
        self.layers = nn.ModuleList()
        self.layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True)
        ])
        for i in range(self.layer_num):
            self.layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        self.layers.extend([
                nn.Linear(hidden_dim, output_dim),
                activation
        ])

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x

def get_data():
    loader = torch.utils.data.DataLoader(
            datasets.MNIST(DATA_PATH, train=True, download=True,
            transform=transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]), ),
            batch_size=BATCH_SIZE, shuffle=True,
    )
    return loader

if __name__ == '__main__':
    img_shape = int(np.prod(IMG_SHAPE))
    loader = get_data()

    g_activation = nn.Tanh()
    d_activation = nn.Sigmoid()

    G = Generator(input_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, output_dim=img_shape, layer_num=LAYER_NUM, activation=g_activation)
    D = Discriminator(input_dim=img_shape, hidden_dim=HIDDEN_DIM, output_dim=1, layer_num=LAYER_NUM, activation=d_activation)

    g_optimizer = torch.optim.Adam(G.parameters(), lr=LEARNING_RATE, betas=BETA)
    d_optimizer = torch.optim.Adam(D.parameters(), lr=LEARNING_RATE, betas=BETA)

    criterion = nn.BCELoss()

    is_cuda = True if torch.cuda.is_available() else False
    Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor

    # print(f"------- is_cuda={is_cuda}")

    if is_cuda:
        G.cuda()
        D.cuda()
        criterion.cuda()

    begin_time = time.time()
    for epoch in range(EPOCHS):
        for batch_id, (img, _) in enumerate(loader):

            assert img.shape[0] == img.size(0), f"img.shape[0]={img.shape[0]} != img.size(0)={img.size(0)}"

            real_label = torch.ones([img.size(0), 1])
            fake_label = torch.zeros([img.size(0), 1])
            if is_cuda:
                real_label = real_label.cuda()
                fake_label = fake_label.cuda()

            G.zero_grad()
            g_fake_input = Variable(Tensor(np.random.normal(0, 1, (img.shape[0], LATENT_DIM))))
            g_fake_output = G(g_fake_input)
            g_loss = criterion(D(g_fake_output), real_label)
            g_loss.backward(retain_graph=True)
            g_optimizer.step()

            D.zero_grad()
            real_img = Variable(img.type(Tensor))
            d_real_loss = criterion(D(real_img), real_label)
            d_fake_loss = criterion(D(g_fake_output), fake_label)

            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            d_optimizer.step()

            if (batch_id + epoch * len(loader) + 1) % print_interval == 0:
                time_cost = time.time()-begin_time
                print(f"Epoch:{epoch:03d}/{EPOCHS:03d} Batch:{batch_id:03d}/{len(loader):03d} g_loss={g_loss:.3f} d_loss={d_loss:.3f} time={time_cost:.3f}s")
                save_image(g_fake_output.data[:25], SAVE_PATH + f"{epoch+1:03d}_{batch_id:03d}.png", nrow=5, normalize=True)
