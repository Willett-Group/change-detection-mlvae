import os
import shutil

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


transforms = {
    'mnist': transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]),
    'cifar10': transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.49139968, 0.48215827, 0.44653124],
        #                      std=[0.24703233, 0.24348505, 0.26158768])
    ]),
    'cifar100': transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
        #                      std=[0.2675, 0.2565, 0.2761])
    ]),
    'celeba': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]),
    'celeba_standard': transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]),
    'rgb': transforms.Lambda(lambda image: image.convert('RGB'))
}


def drawlines(image, eta):
    image[eta-1, 0, :, -5:] = 0
    image[eta-1, 1, :, -5:] = 239
    image[eta-1, 2, :, -5:] = 255
    image[eta, 0, :, :5] = 0
    image[eta, 1, :, :5] = 239
    image[eta, 2, :, :5] = 255

    return image


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.cnt = 0
        self.avg = 0
        self.values = []

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
        for _ in range(n):
            self.values.append(val)


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def percent_by_bound(values, bound=0):
    cnt = 0
    for v in values:
        if v <= bound:
            cnt += 1
    
    return cnt / len(values)


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def create_exp_dir(path, scripts_to_save=None):
    os.makedirs(path, exist_ok=True)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.makedirs(os.path.join(path, 'scripts'), exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)





def accumulate_group_evidence(mu, logvar, target, device):
    target = target.detach().cpu().numpy()
    var_dict = {}
    mu_dict = {}

    var = logvar.exp()

    for i in range(len(target)):
        label = target[i]
        var_dict[label] = var_dict.get(label, 0) + 1/var[i]
    for label in var_dict:
        var_dict[label] = 1/var_dict[label]
    
    for i in range(len(target)):
        label = target[i]
        mu_dict[label] = mu_dict.get(label, 0) + mu[i]/var[i]
    for label in mu_dict:
        mu_dict[label] *= var_dict[label]
    
    grouped_mu = torch.empty_like(mu).to(device)
    grouped_var = torch.empty_like(var).to(device)

    for i in range(len(target)):
        label = target[i]
        grouped_mu[i] = mu_dict[label]
        grouped_var[i] = var_dict[label]
    
    return grouped_mu, torch.log(grouped_var)


def reparameterize(mu, logvar, training):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)

    if training:
        return mu + std * eps
    else:
        return mu


def group_wise_reparameterize(mu, logvar, target, training):
    target = target.detach().cpu().numpy()
    eps_dict = {}

    # generate only 1 eps value per target value
    for t in np.unique(target):
        eps_dict[t] = torch.randn_like(mu[0])

    if training:
        std = torch.exp(0.5 * logvar)
        sample = mu.clone()
        # multiply std by correct eps and add mu
        for i in range(len(target)):
            sample[i] += std[i] * eps_dict[target[i]]

        return sample
    else:
        return mu


def reconstruct(model, s_mu, s_logvar, grouped_mu, grouped_logvar, target, use_s=True):
    if use_s:
        style_z = reparameterize(s_mu, s_logvar, training=True)
    else:
        style_z = torch.zeros_like(s_mu)
    content_z = group_wise_reparameterize(grouped_mu, grouped_logvar, target, training=True)
    
    recon = model.decode(style_z, content_z)
    
    return recon


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
    train_indices_individuals = [j for i in train_indices for j in range(10 * i, 10 * i + 10)]
    train_sampler = SubsetRandomSampler(train_indices_individuals)

    return train_sampler, test_indices