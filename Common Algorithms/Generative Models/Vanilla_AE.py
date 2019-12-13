# -*- coding: utf-8 -*-

import os, sys, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets
from torchvision.utils import save_image
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

DATA_PATH = '../dataset/'
SAVE_PATH = './images/Vanilla_AE/'
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(SAVE_PATH, exist_ok=True)

BATCH_SIZE = 128
IMG_SIZE = 28
CHANNEL = 1
IMG_SHAPE = (CHANNEL, IMG_SIZE, IMG_SIZE)

LEARNING_RATE = 1e-3
BETAS = (0.5, 0.999)

EPOCHS = 100
print_interval = 1000

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
                nn.Linear(IMG_SIZE*IMG_SIZE, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(128, 32),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(32, 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(8, 3)
        )

        self.decoder = nn.Sequential(
                nn.Linear(3, 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(8, 32),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(32, 128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(128, IMG_SIZE * IMG_SIZE),
                nn.Tanh()
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode


def get_data():
    trans = transforms.Compose([transforms.Resize(IMG_SIZE), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    dataset = datasets.MNIST(DATA_PATH, train=True, download=True, transform=trans)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return loader

if __name__ == "__main__":
    is_cuda = True if torch.cuda.is_available() else False

    loader = get_data()

    ae = AutoEncoder()
    optimizer = torch.optim.Adam(ae.parameters(), lr=LEARNING_RATE, betas=BETAS)
    #criterion = nn.BCELoss()
    criterion = nn.MSELoss(reduction='sum')

    Tensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor

    if is_cuda:
        ae.cuda()
        criterion.cuda()

    begin_time = time.time()
    for epoch in range(EPOCHS):
        for batch_id, (img, label) in enumerate(loader):
            ae.zero_grad()

            img = img.view(img.shape[0], -1)
            input = Variable(img.type(Tensor))
            encode, decode = ae(input)
            loss = criterion(decode, input) / img.shape[0]
            loss.backward()
            optimizer.step()

            if (epoch * len(loader) + batch_id+1) % print_interval == 0:
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

                #save_image(img[:25], SAVE_PATH + f"{epoch + 1:03d}_{batch_id:03d}_input.png", nrow=5, normalize=False)
                #save_image(decode.data[:25], SAVE_PATH + f"{epoch + 1:03d}_{batch_id:03d}_decoder.png", nrow=5, normalize=False)

                fig = plt.figure(2)
                ax = Axes3D(fig)  # 3D graph
                encode = encode.cpu()
                X = encode.data[:, 0].numpy()
                Y = encode.data[:, 1].numpy()
                Z = encode.data[:, 2].numpy()
                for x, y, z, s in zip(X, Y, Z, label.numpy()):
                    c = cm.rainbow(int(255 * s / 9))  # color
                    ax.text(x, y, z, s, backgroundcolor=c)  # position
                ax.set_xlim(X.min(), X.max())
                ax.set_ylim(Y.min(), Y.max())
                ax.set_zlim(Z.min(), Z.max())
                plt.savefig(SAVE_PATH + f"{epoch+1:03d}_{batch_id:03d}_encoder.png")

                # plt.show(block=False)
                # plt.pause(2)
                # plt.close('all')