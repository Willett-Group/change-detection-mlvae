'''
Networks and custom loss functions
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import resnet


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2, label):
        d = F.pairwise_distance(x1, x2, keepdim=True)
        loss = torch.mean((1-label) * torch.pow(d, 2) + label * torch.pow(torch.clamp(self.margin-d, min=0.0)), 2)

        return loss


class SiameseNet(nn.Module):
    def __init__(self, loss='ruslan', arch='cnn'):
        super(SiameseNet, self).__init__()
        self.loss = loss
        self.arch = arch

        self.conv1 = nn.Sequential(
            resnet.ResNet18(),
        ) # output dim = 512

        self.conv2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),

            nn.Flatten(),
        ) # output dim = 1024
        self.fc3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(512, 1)
        )

    def forward(self, input):
        if self.arch == 'resnet':
            encode1 = self.conv1(input[:, 0, :, :])
            encode2 = self.conv1(input[:, 1, :, :])
        elif self.arch == 'cnn':
            encode1 = self.fc3(self.conv2(input[:, 0, :, :]))
            encode2 = self.fc3(self.conv2(input[:, 1, :, :]))
        else:
            raise Exception("Invalid architecture")

        if self.loss == 'ruslan':
            return self.fc4(torch.abs(encode1 - encode2))
        elif self.loss == 'contrastive':
            return encode1, encode2
        else:
            raise Exception("Invalid method name")


class linearVAE(nn.Module):
    def __init__(self, input_dim, l1_dim, z_dim):
        super(linearVAE, self).__init__()
        self.input_dim = input_dim
        flat_dim = int(np.prod(input_dim))

        self.linear1 = nn.Sequential(
            nn.Linear(in_features=flat_dim, out_features=l1_dim, bias=True),
            nn.ReLU()
        )

        self.mu = nn.Linear(l1_dim, z_dim, bias=True)
        self.log_var = nn.Linear(l1_dim, z_dim, bias=True)

        self.linear2 = nn.Sequential(
            nn.Linear(z_dim, l1_dim, bias=True),
            nn.ReLU(),
            nn.Linear(l1_dim, flat_dim, bias=True),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        mu = self.mu(x)
        logvar = self.log_var(x)

        return mu, logvar

    def decode(self, z):
        x = self.linear2(z)
        x = x.view((-1,) + self.input_dim)
        return x


class linearMLVAE(nn.Module):
    def __init__(self, input_dim, l1_dim, cs_dim):
        super(linearMLVAE, self).__init__()
        self.input_dim = input_dim # a tuple of the dimension of the input
        flat_dim = np.prod(input_dim)

        self.linear1 = nn.Sequential(
            nn.Linear(in_features=flat_dim, out_features=l1_dim, bias=True),
            nn.ReLU()
        )

        # style
        self.s_mu = nn.Linear(l1_dim, cs_dim, bias=True)
        self.s_logvar = nn.Linear(l1_dim, cs_dim, bias=True)
        # content
        self.c_mu = nn.Linear(l1_dim, cs_dim, bias=True)
        self.c_logvar = nn.Linear(l1_dim, cs_dim, bias=True)

        self.linear2 = nn.Sequential(
            nn.Linear(2*cs_dim, l1_dim, bias=True),
            nn.ReLU(),
            nn.Linear(l1_dim, flat_dim, bias=True),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        # style
        s_mu = self.s_mu(x)
        s_logvar = self.s_logvar(x)
        # content
        c_mu = self.c_mu(x)
        c_logvar = self.c_logvar(x)

        return s_mu, s_logvar, c_mu, c_logvar
    
    def decode(self, s, c):
        z = torch.cat((s, c), dim=1)
        x = self.linear2(z)
        x = x.view((-1,) + self.input_dim)
        return x