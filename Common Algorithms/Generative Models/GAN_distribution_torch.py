# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer_num, acti_func, last_acti=True):
        super(NeuralNetwork, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size, bias=True))
        for i in range(layer_num):
            self.layers.append(nn.Linear(hidden_size, hidden_size, bias=True))
        self.layers.append(nn.Linear(hidden_size, output_size, bias=True))

        self.activation = acti_func
        self.last_acti = last_acti
        self.layer_num = layer_num + 2

    def forward(self, x):
        for index, layer in enumerate(self.layers):
            if index == self.layer_num - 1 and self.last_acti == False:
                x = layer(x)
            else:
                x = self.activation(layer(x))
        return x

def get_4_moments(x):
    mean = torch.mean(x)
    diff = x - mean
    vars = torch.mean(torch.pow(diff, 2.0))
    std = torch.pow(vars, 0.5)
    zscores = diff / std
    skews = torch.mean(torch.pow(zscores, 3.0))
    kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0
    ret = torch.cat((mean.reshape(1, ), std.reshape(1, ), skews.reshape(1, ), kurtoses.reshape(1, )))
    return ret


if __name__ == '__main__':
    g_input_size, g_hidden_size, g_output_size, g_layer_num = 1, 5, 1, 1
    d_input_size, d_hidden_size, d_output_size, d_layer_num = 500, 10, 1, 1
    batch_size = d_input_size
    g_activation, d_activation = torch.tanh, torch.sigmoid

    g_learning_rate, d_learning_rate = 1e-3, 1e-3
    sgd_momentum = 0.9
    epochs = 5000
    print_interval = 100
    g_steps, d_steps = 20, 20

    input_mean, input_std = 4, 2.5

    g_sampler = lambda m, n: torch.rand(m, n)
    d_sampler = lambda n, mu, sigma: torch.Tensor(np.random.normal(mu, sigma, (1, n)))

    G = NeuralNetwork(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size,
                      layer_num=g_layer_num, acti_func=g_activation, last_acti=False)
    D = NeuralNetwork(input_size=4, hidden_size=d_hidden_size, output_size=d_output_size, # there are 4 moments
                      layer_num=d_layer_num, acti_func=d_activation, last_acti=True)
    criterion = nn.BCELoss()
    g_optimizer = torch.optim.SGD(G.parameters(), lr=g_learning_rate, momentum=sgd_momentum)
    d_optimizer = torch.optim.SGD(D.parameters(), lr=d_learning_rate, momentum=sgd_momentum)

    for epoch in range(epochs):
        for d_iter in range(d_steps):
            D.zero_grad()

            d_real_input = Variable(d_sampler(n=d_input_size, mu=input_mean, sigma=input_std))
            d_real_output = D(get_4_moments(d_real_input))
            d_real_loss = criterion(d_real_output, Variable(torch.ones([1, 1])))
            d_real_loss.backward()

            d_gen_input = Variable(g_sampler(m=batch_size, n=g_input_size))
            d_fake_input = G(d_gen_input).detach()
            d_fake_output = D(get_4_moments((d_fake_input.t())))
            d_fake_loss = criterion(d_fake_output, Variable(torch.zeros([1, 1])))
            d_fake_loss.backward()

            d_optimizer.step()

            d_real_error = d_real_loss.data.storage().tolist()[0]
            d_fake_error = d_fake_loss.data.storage().tolist()[0]


        for g_iter in range(g_steps):
            G.zero_grad()

            g_input = Variable(g_sampler(m=batch_size, n=g_input_size))
            g_output = G(g_input)
            d_output = D(get_4_moments(g_output.t()))
            g_loss = criterion(d_output, Variable(torch.ones([1, 1])))
            g_loss.backward()

            g_optimizer.step()

            g_error = g_loss.data.storage().tolist()[0]

        if (epoch + 1) % print_interval == 0:
            real_input = d_real_input.data.storage().tolist()
            fake_input = d_fake_input.data.storage().tolist()
            # print(type(d_real_error), type(np.mean(real_input)))
            print(f"Epoch:{epoch+1:04d} D(real_err={d_real_error:.3f}, fake_err={d_fake_error:.3f}) G(err={g_error:.3f}) "
                  f"Real(mean={np.mean(real_input):.3f}, std={np.std(real_input):.3f}) "
                  f"Fake(mean={np.mean(fake_input):.3f}, std={np.std(fake_input):.3f})")

    print("Plotting the generated distribution...")
    values = g_output.data.storage().tolist()
    # print(" Values: %s" % (str(values)))
    plt.hist(values, bins=50)
    plt.xlabel('Value')
    plt.ylabel('Count')
    plt.title('Histogram of Generated Distribution')
    plt.grid(True)
    plt.show()