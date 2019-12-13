# -*- coding: utf-8 -*-

import os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets
from torchvision.utils import save_image
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

IMG_SIZE = 28
CHANNEL = 1
IMG_SHAPE = (CHANNEL, IMG_SIZE, IMG_SIZE)
BATCH_SIZE = 128

DATA_PATH = '../dataset/'
SAVE_PATH = './images/VAE/'
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(SAVE_PATH, exist_ok=True)

LEARNING_RATE = 1e-3
BETAS = (0.5, 0.999)

EPOCHS = 100
print_interval = 1000

is_cuda = True if torch.cuda.is_available() else False

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # encode
        self.fc1 = nn.Linear(IMG_SIZE*IMG_SIZE, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)

        # decode
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, IMG_SIZE*IMG_SIZE)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar): # add noise
        std = logvar.mul(0.5).exp_()
        if is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    mse = nn.MSELoss(reduction='sum')
    mse_loss = mse(recon_x, x)
    # KL_loss = 0.5 * (1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return mse_loss + KLD

def get_data():
    trans = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    dataset = datasets.MNIST(DATA_PATH, train=True, download=True, transform=trans)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader

if __name__ == '__main__':
    loader = get_data()

    vae = VAE()
    optimizer = torch.optim.Adam(vae.parameters(), lr=LEARNING_RATE, betas=BETAS)
    criterion = loss_function

    Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor

    if is_cuda:
        vae.cuda()
        # criterion.cuda()

    begin_time = time.time()
    for epoch in range(EPOCHS):
        for batch_id, (img, label) in enumerate(loader):
            vae.zero_grad()

            img = img.view(img.shape[0], -1)
            input = Variable(img.type(Tensor))
            if is_cuda:
                input = input.cuda()

            decode, mu, logvar = vae(input)
            loss = criterion(decode, input, mu, logvar) / img.shape[0]
            loss.backward()
            optimizer.step()

            if (epoch * len(loader) + batch_id + 1) % print_interval == 0:
                time_cost = time.time() - begin_time
                print(f"Epoch: {epoch:03d}/{EPOCHS:03d} Batch: {batch_id:03d}/{len(loader):03d} loss={loss:.3f} time={time_cost:.3f}s")
                img = img.view(img.shape[0], *IMG_SHAPE)
                decode = decode.view(decode.shape[0], *IMG_SHAPE)

                input, output = img[:25].cuda(), decode.data[:25].cuda()

                # normalize decode
                Min, Max = float(output.min()), float(output.max())
                output.clamp_(min=Min, max=Max)
                output.add_(-Min).div_(Max - Min + 1e-5)

                imgs = []
                for i in range(25):
                    imgs.append(input[i])
                    imgs.append(output[i])
                save_image(imgs, SAVE_PATH + f"{epoch + 1:03d}_{batch_id:03d}.png", nrow=10, normalize=False)

                # save_image(img[:25], SAVE_PATH + f"{epoch + 1:03d}_{batch_id:03d}_input.png", nrow=5, normalize=False)
                # save_image(decode.data[:25], SAVE_PATH + f"{epoch + 1:03d}_{batch_id:03d}_decoder.png", nrow=5, normalize=False)
