import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable

import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np

# compose a transform configuration
trans_config = transforms.Compose([
    transforms.ToTensor()
])

trans_config1 = transforms.Compose([
    transforms.Resize([64, 64]),
    transforms.ToTensor()
])

trans_config1_special = transforms.Compose([
    transforms.Resize([48, 64]),
    transforms.ToTensor()
])

trans_config2 = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor()
])

trans_config2_norm = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_rgb = transforms.Lambda(lambda image: image.convert('RGB'))
transform_flatten = transforms.Lambda(lambda image: torch.flatten(image))

transform_config4 = transforms.Compose([
    transforms.Resize([64, 64]),
    transform_rgb,
    transforms.ToTensor()
])

transform_config5 = transforms.Compose([
    transforms.ToTensor(),
    transform_flatten
])


def accumulate_group_evidence(class_mu, class_logvar, labels_batch):
    """
    :param class_mu: mu values for class latent embeddings of each sample in the mini-batch
    :param class_logvar: logvar values for class latent embeddings for each sample in the mini-batch
    :param labels_batch: class labels of each sample (the operation of accumulating class evidence can also
        be performed using group labels instead of actual class labels)
    :param is_cuda:
    :return:
    """
    var_dict = {}
    mu_dict = {}

    # convert logvar to variance for calculations
    class_var = class_logvar.exp()

    # calculate var inverse for each group using group vars
    for i in range(len(labels_batch)):
        group_label = labels_batch[i].item()

        # remove 0 values from variances
        class_var[i][class_var[i] == float(0)] = 1e-6

        if group_label in var_dict.keys():
            var_dict[group_label] += 1 / class_var[i]
        else:
            var_dict[group_label] = 1 / class_var[i]

    # invert var inverses to calculate mu and return value
    for group_label in var_dict:
        var_dict[group_label] = 1 / var_dict[group_label]

    # calculate mu for each group
    for i in range(len(labels_batch)):
        group_label = labels_batch[i].item()

        if group_label in mu_dict.keys():
            mu_dict[group_label] += class_mu[i] * (1 / class_var[i])
        else:
            mu_dict[group_label] = class_mu[i] * (1 / class_var[i])

    # multiply group var with sums calculated above to get mu for the group
    for group_label in mu_dict:
        mu_dict[group_label] *= var_dict[group_label]

    # replace individual mu and logvar values for each sample with group mu and logvar
    group_mu = torch.FloatTensor(class_mu.size(0), class_mu.size(1))
    group_var = torch.FloatTensor(class_var.size(0), class_var.size(1))

    group_mu = group_mu.cuda()
    group_var = group_var.cuda()

    for i in range(len(labels_batch)):
        group_label = labels_batch[i].item()

        group_mu[i] = mu_dict[group_label]
        group_var[i] = var_dict[group_label]

        # remove 0 from var before taking log
        group_var[i][group_var[i] == float(0)] = 1e-6

    # convert group vars into logvars before returning
    return Variable(group_mu, requires_grad=True), Variable(torch.log(group_var), requires_grad=True), var_dict, mu_dict


def mse_loss(input, target):
    return torch.sum((input - target).pow(2)) / input.data.nelement()


def l1_loss(input, target):
    return torch.sum(torch.abs(input - target)) / input.data.nelement()


def normal_density(eps):
    # eps is a 1 by 1 tensor
    return 1


def reparameterize(mu, logvar, training):
    if training:
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    else:
        return mu


def group_wise_reparameterize(mu, logvar, labels_batch, cuda, training):
    eps_dict = {}

    # generate only 1 eps value per group label
    for label in torch.unique(labels_batch):
        if cuda:
            eps_dict[label.item()] = torch.cuda.FloatTensor(1, logvar.size(1)).normal_(0., 0.1)
        else:
            eps_dict[label.item()] = torch.FloatTensor(1, logvar.size(1)).normal_(0., 0.1)

    if training:
        std = logvar.mul(0.5).exp_()
        reparameterized_var = Variable(std.data.new(std.size()))
        # multiply std by correct eps and add mu
        for i in range(logvar.size(0)):
            reparameterized_var[i] = std[i].mul(Variable(eps_dict[labels_batch[i].item()]))
            reparameterized_var[i].add_(mu[i])

        return reparameterized_var
    else:
        return mu


def weights_init(layer):
    if isinstance(layer, nn.Conv2d):
        layer.weight.data.normal_(0.0, 0.05)
        if layer.bias is not None:
            layer.bias.data.zero_()
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.normal_(1.0, 0.02)
        if layer.bias is not None:
            layer.bias.data.zero_()
    elif isinstance(layer, nn.BatchNorm1d):
        layer.weight.data.normal_(1.0, 0.02)
        if layer.bias is not None:
            layer.bias.data.zero_()
    elif isinstance(layer, nn.Linear):
        layer.weight.data.normal_(0.0, 0.05)
        if layer.bias is not None:
            layer.bias.data.zero_()


def subset_sampler(ds, T, test_split, shuffle, random_seed):
    n = len(ds) // T
    print(n)
    indices = list(range(n))
    split = int(np.floor(test_split * n))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    train_indices_individuals = [j for i in train_indices for j in range(10*i, 10*i+10)]
    train_sampler = SubsetRandomSampler(train_indices_individuals)

    return train_sampler, test_indices