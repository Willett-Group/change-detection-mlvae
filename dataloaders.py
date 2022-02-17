import random
import os
import os.path as path
from math import comb

import numpy as np
import pandas
from PIL import Image
import torch
from torchvision import datasets
from torch.utils.data import Dataset

import utils
from utils import transforms

#################################################################################

def get_splits(t, margin_candidates, k, nclasses, nsubclasses, classes, subclasses):
    assert k >= 2
    assert (nclasses == 1 or nsubclasses == 1) and nclasses * nsubclasses == k

    margins = random.sample(margin_candidates, k - 1)
    margins.append(t)
    margins.insert(0, 0) # first parameter is index
    segments = []
    for i in range(len(margins) - 1):
        segments.append(margins[i + 1] - margins[i])

    classes = random.sample(classes, nclasses)
    subclasses = random.sample(subclasses, nsubclasses)
    if nclasses == 1:
        classes = classes * k
    elif nsubclasses == 1:
        subclasses = subclasses * k

    return [segments, classes, subclasses]


def cumulatize(splits):
    n, _, k = splits.shape
    result = [{} for i in range(n)]
    snapshot = {}
    for i in range(n):
        result[i] = snapshot.copy()
        for j, class_key in enumerate(splits[i][1:].transpose()):
            class_key = tuple(class_key)
            snapshot[class_key] = snapshot.get(class_key, 0) + splits[i][0][j]

    return result


def discretize(segments, t):
    k = 0
    while t >= segments[k]:
        t -= segments[k]
        k += 1

    return k


def absolute(segments):
    result = [0]*(len(segments)+1)
    for i in range(1, len(result)):
        result[i] = result[i-1] + segments[i-1]

    return result


# time-series datasets
class TS(Dataset):
    def __init__(self, datapath, dataset, split, n_max, t_max, classes, transform):
        is_train = split == 'train'
        if dataset == 'mnist':
            self.dataset = datasets.MNIST(root=datapath, download=True, train=is_train, transform=transform)
        elif dataset == 'cifar10':
            self.dataset = datasets.CIFAR10(root=datapath, download=True, train=is_train, transform=transform)
        elif dataset == 'cifar100':
            self.dataset = datasets.CIFAR100(root=datapath, download=True, train=is_train, transform=transform)
        elif dataset == 'celeba':
            self.dataset = datasets.CelebA(root=datapath, download=True, split=split, transform=transform)
        else:
            raise Exception("Dataset not found")

        self.dims = self.dataset[0][0].size()
        self.N = n_max
        self.T = t_max
        min_t = 5
        assert self.T > 2*min_t
        margin_candidates = list(range(min_t, self.T - min_t))

        if dataset != 'celeba':
            indices = np.arange(len(self.dataset))
            targets = np.array(self.dataset.targets)
            indices_by_c = {}
            for c in classes:
                indices_by_c[(c, 0)] = indices[targets == c]
            self.indices_by_c = indices_by_c

            self.splits = np.asarray(
                [get_splits(self.T, margin_candidates, 2, 2, 1, classes, [0]) for _ in range(self.N)])
            self.splits_cum = cumulatize(self.splits)
        else:
            subclasses = np.unique(self.dataset.attr)
            indices_by_c = {}
            for subc in subclasses:
                hits = np.where(self.dataset.attr == subc)
                for c in classes:
                    indices_by_c[(c, subc)] = hits[0][hits[1] == c]
            for c_key in indices_by_c:
                random.shuffle(indices_by_c[c_key])
            self.indices_by_c = indices_by_c

            self.splits = np.asarray(
                [get_splits(self.T, margin_candidates, 2, 1, 2, classes, [0, 1]) for _ in range(self.N)])
            self.splits_cum = cumulatize(self.splits)

    def __len__(self):
        return self.N * self.T

    def __getitem__(self, idx):
        n = idx // self.T
        t = idx % self.T
        N, _, K = self.splits.shape

        segments = self.splits[n][0]
        k = discretize(segments, t)

        c_key = tuple(self.splits[n][1:].transpose()[k])

        if c_key not in self.splits_cum[n]:
            index = t
        else:
            index = self.splits_cum[n][c_key] + t
        if index >= len(self.indices_by_c[c_key]):
            random.shuffle(self.indices_by_c[c_key])
            index = index % len(self.indices_by_c[c_key])
        sample, _ = self.dataset[self.indices_by_c[c_key][index]]
        label = n * K + k

        return sample, label

    def get_x_n(self, n):
        x = torch.empty(size=(self.T,) + self.dims)
        for t in range(self.T):
            x[t], _ = TS.__getitem__(self, n * self.T + t)

        return x


# contrastive dataset based on TS dataset
class CON(TS):
    def __init__(self, dataset, split, n_max, t_max, p_max, classes, transform):
        super().__init__(dataset, split, n_max, t_max, classes, transform)
        self.P = p_max
        N, _, K = self.splits.shape
        self.K = K

    def __len__(self):
        return self.N * self.P

    def __getitem__(self, idx):
        n = idx // self.P
        N, _, K = self.splits.shape

        segments = self.splits[n][0]
        segments_cum = absolute(segments)
        npros = np.asarray([comb(s, 2) for s in segments])
        npros = npros / np.sum(npros)

        is_positive = idx % self.P <= self.P // 2  # whether is positive pair
        if is_positive:
            k = random.choices(range(K), weights=npros, k=1)[0]
            chosen_ts = random.sample(range(segments_cum[k], segments_cum[k+1]), K)
        else:
            chosen_ts = [random.sample(range(segments_cum[k], segments_cum[k+1]), 1)[0] for k in range(0, len(segments_cum)-1)]
        samples = torch.empty(size=(self.K,) + self.dims)

        for k, t in enumerate(chosen_ts):
            samples[k] = super().__getitem__(n * self.T + t)[0]

        return samples, 1 - int(is_positive)  # positive sample is labeled 0


class clevr_change_contrastive_explicit(Dataset):
    def __init__(self, name, T, P, transform_configuration):
        self.dir1 = path.join(dataset_dir, 'clevr', name, 'outputnsc_images/')
        self.dir2 = path.join(dataset_dir, 'clevr', name, 'outputsc_images/')
        self.transform = transform_configuration

        all_filenames = os.listdir(self.dir1)
        image_1 = Image.open(self.dir1 + all_filenames[0]).convert('RGB')
        self.data_dim = self.transform(image_1).shape

        cps = {}
        for filename in all_filenames:
            _, _, i, t = filename.split('_')  # i-th image at time t
            t = int(t[:-4])
            i = int(i)
            cps[i] = cps.get(i, 0) + 1

        self.cps = cps

        self.n, self.T, self.P = len(cps), T, P

    def __len__(self):
        return self.n * self.P

    def __getitem__(self, item):
        i, t = item // self.T, item % self.T

        if t < self.cps[i]:
            label = 2 * i
            file_name = path.join(self.dir1, 'CLEVR_nonsemantic_' + str(i).zfill(6) + '_' + str(t) + '.png')
            img = Image.open(file_name).convert('RGB')
        else:
            label = 2 * i + 1
            file_name = path.join(self.dir2, 'CLEVR_semantic_' + str(i).zfill(6) + '_' + str(t - self.cps[i]) + '.png')
            img = Image.open(file_name).convert('RGB')

        return self.transform(img), label

    def get_time_series_sample(self, n):
        X = torch.empty(size=(self.T,) + self.data_dim)
        for t in range(self.T):
            X[t] = self.__getitem__(self.T * n + t)[0]
        return X


class clevr_change(Dataset):
    def __init__(self, name, T, transform_configuration):
        self.dir1 = path.join(dataset_dir, 'clevr', name, 'outputnsc_images/')
        self.dir2 = path.join(dataset_dir, 'clevr', name, 'outputsc_images/')
        self.transform = transform_configuration

        all_filenames = os.listdir(self.dir1)
        image_1 = Image.open(self.dir1 + all_filenames[0]).convert('RGB')
        self.data_dim = self.transform(image_1).shape

        cps = {}
        for filename in all_filenames:
            _, _, i, t = filename.split('_')  # i-th image at time t
            t = int(t[:-4])
            i = int(i)
            cps[i] = cps.get(i, 0) + 1

        self.cps = cps
        self.n = len(cps)
        self.T = T

    def __len__(self):
        return self.n * self.T

    def __getitem__(self, item):
        i, t = item // self.T, item % self.T

        if t < self.cps[i]:
            label = 2 * i
            file_name = path.join(self.dir1, 'CLEVR_nonsemantic_' + str(i).zfill(6) + '_' + str(t) + '.png')
            img = Image.open(file_name).convert('RGB')
        else:
            label = 2 * i + 1
            file_name = path.join(self.dir2, 'CLEVR_semantic_' + str(i).zfill(6) + '_' + str(t - self.cps[i]) + '.png')
            img = Image.open(file_name).convert('RGB')

        return self.transform(img), label

    def get_time_series_sample(self, n):
        X = torch.empty(size=(self.T,) + self.data_dim)
        for t in range(self.T):
            X[t] = self.__getitem__(self.T * n + t)[0]
        return X