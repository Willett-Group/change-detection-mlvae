import random
import os
import os.path as path
import numpy as np
from PIL import Image
import pickle

import torch
from torchvision import datasets
from torch.utils.data import Dataset

import utils

#################################################################################
# directory where all datasets are saved
dataset_dir = '../datasets/'

'''data loaders for CelebA-based time series
all samples before change points are men, all samples after change point are women
each man sample is randomly selected from the set of all man samples in celeba
each woman sample is randomly selected from the set of all woman samples in celeba
label is the group number. there are 2n groups in total'''


class celeba_vanilla(Dataset):
    def __init__(self, transform, train=True):
        split = 'train' if train else 'test'
        self.celeba = datasets.CelebA(root=dataset_dir, download=True, split=split, transform=transform)
        self.data_dim = self.celeba[0][0].size()

    def __len__(self):
        return len(self.celeba)

    def __getitem__(self, idx):
        sample = self.celeba[idx][0]
        label = self.celeba[idx][1][20]
        return sample, label

class celeba_gender_change(Dataset):
    def __init__(self, n, T, transform, train=True, seed=7):
        random.seed(seed)
        split = 'train' if train else 'test'
        self.celeba = datasets.CelebA(root=dataset_dir, download=True, split=split, transform=transform)
        self.n = n
        self.T = T
        self.data_dim = self.celeba[0][0].size()

        possible_cps = list(range(T//4, 3*T//4+1))
        self.cps = [random.choice(possible_cps) for _ in range(n)]
        self.men_indices = []
        self.women_indices = []

        temp_men_save = 'men_I_train.pickle' if train else 'men_I_test.pickle'
        temp_women_save = 'women_I_train.pickle' if train else 'women_I_test.pickle'
        if path.exists(path.join(dataset_dir, 'celeba', temp_men_save)):
            with open(path.join(dataset_dir, 'celeba', temp_men_save), 'rb') as f:
                self.men_indices = pickle.load(f)
        if path.exists(path.join(dataset_dir, 'celeba', temp_women_save)):
            with open(path.join(dataset_dir, 'celeba', temp_women_save), 'rb') as f:
                self.women_indices = pickle.load(f)
        if not self.men_indices and not self.women_indices:
            for i in range(len(self.celeba)):
                if self.celeba[i][1][20] == 1:
                    self.men_indices.append(i)
                else:
                    self.women_indices.append(i)
            with open(path.join(dataset_dir, 'celeba', temp_men_save), 'wb') as f:
                pickle.dump(self.men_indices, f)
            with open(path.join(dataset_dir, 'celeba', temp_women_save), 'wb') as f:
                pickle.dump(self.women_indices, f)

    def __len__(self):
        return self.n*self.T
    
    def __getitem__(self, idx):
        t = idx % self.T
        row = idx // self.T
        label = 2*row if t < self.cps[row] else 2*row + 1

        if t < self.cps[row]:
            sample = self.celeba[random.choice(self.men_indices)][0]
        else:
            sample = self.celeba[random.choice(self.women_indices)][0]

        return sample, label

    def get_time_series_sample(self, n):
        X = torch.empty(size=(self.T,) + self.data_dim)
        for t in range(self.T):
            X[t] = self.__getitem__(self.T*n + t)[0]
        return X


class celeba_contrastive(celeba_gender_change):
    def __init__(self, n, T, transform, train=True, seed=7):
        random.seed(seed)
        split = 'train' if train else 'test'
        self.celeba = datasets.CelebA(root=dataset_dir, download=True, split=split, transform=transform)
        self.n = n
        self.T = T
        self.data_dim = self.celeba[0][0].size()

        possible_cps = list(range(T // 4, 3 * T // 4 + 1))
        self.cps = [random.choice(possible_cps) for _ in range(n)]
        self.men_indices = []
        self.women_indices = []

        temp_men_save = 'men_I_train.pickle' if train else 'men_I_test.pickle'
        temp_women_save = 'women_I_train.pickle' if train else 'women_I_test.pickle'
        if path.exists(path.join(dataset_dir, 'celeba', temp_men_save)):
            with open(path.join(dataset_dir, 'celeba', temp_men_save), 'rb') as f:
                self.men_indices = pickle.load(f)
        if path.exists(path.join(dataset_dir, 'celeba', temp_women_save)):
            with open(path.join(dataset_dir, 'celeba', temp_women_save), 'rb') as f:
                self.women_indices = pickle.load(f)
        if not self.men_indices and not self.women_indices:
            for i in range(len(self.celeba)):
                if self.celeba[i][1][20] == 1:
                    self.men_indices.append(i)
                else:
                    self.women_indices.append(i)
            with open(path.join(dataset_dir, 'celeba', temp_men_save), 'wb') as f:
                pickle.dump(self.men_indices, f)
            with open(path.join(dataset_dir, 'celeba', temp_women_save), 'wb') as f:
                pickle.dump(self.women_indices, f)

    def __len__(self):
        return self.n * self.T

    def __getitem__(self, idx):
        man1 = self.celeba[random.choice(self.men_indices)][0]
        man2 = self.celeba[random.choice(self.men_indices)][0]
        woman1 = self.celeba[random.choice(self.women_indices)][0]
        woman2 = self.celeba[random.choice(self.women_indices)][0]

        if idx % 2 == 0:
            v = torch.bernoulli(torch.tensor(0.5)).item()
            if v == 0:
                sample = [man1, man2]
            else:
                sample = [woman1, woman2]
            label = 0
        else:
            sample = [man1, woman1]
            label = 1

        return sample, label

    def get_normal(self, idx):
        t = idx % self.T
        row = idx // self.T
        label = 2*row if t < self.cps[row] else 2*row + 1

        if t < self.cps[row]:
            sample = self.celeba[random.choice(self.men_indices)][0]
        else:
            sample = self.celeba[random.choice(self.women_indices)][0]

        return sample, label

    def get_ts_sample(self, n):
        X = torch.empty(size=(self.T,) + self.data_dim)
        for t in range(self.T):
            X[t] = self.get_normal(self.T * n + t)[0]
        return X



class mnist_vanilla(Dataset):
    def __init__(self, transform, train=True):
        self.mnist = datasets.MNIST(root=dataset_dir, download=True, train=train, transform=transform)
        self.data_dim = (3, 64, 64)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        sample = torch.cat(3*[self.mnist[idx][0]])
        label = self.mnist[idx][1]
        return sample, label

####################################################################################
# Data loaders related to MNSIT-based time series datasets
class mnist_loader(Dataset):
    def __init__(self, n, T, transform, train=True, seed=7):
        random.seed(seed)
        self.mnist = datasets.MNIST(root=dataset_dir, download=True, train=train, transform=transform)
        self.n = n
        self.T = T
        self.data_dim = (3, 64, 64)

        possible_cps = list(range(T//4, 3*T//4+1))
        
        # change points of time series samples
        self.cps = [random.choice(possible_cps) for _ in range(self.n)]

        # digits of 2 groups before and after change point
        self.digit_pairs = [random.sample(range(10), 2) for _ in range(self.n)]

        # indices of samples in mnist grouped by digits
        self.indices_by_digit = {}
        indices = np.array(range(len(self.mnist.data)))
        for d in range(10):
            self.indices_by_digit[d] = indices[self.mnist.targets == d].tolist()

    def __len__(self):
        return self.n * self.T
    
    def __getitem__(self, idx):
        row = idx // self.T
        t = idx % self.T

        # get label
        label = 2*row if t < self.cps[row] else 2*row + 1

        # get sample
        digit1, digit2 = self.digit_pairs[row]
        indices1 = self.indices_by_digit[digit1]
        indices2 = self.indices_by_digit[digit2]

        if t < self.cps[row]:
            sample = self.mnist[random.choice(indices1)][0]
            # label = self.mnist[random.choice(indices1)][1]
        else:
            sample = self.mnist[random.choice(indices2)][0]
            # label = self.mnist[random.choice(indices2)][1]

        return torch.cat(3*[sample]), label

    def get_time_series_sample(self, n):
        X = torch.empty(size=(self.T,) + self.data_dim)
        for t in range(self.T):
            X[t] = self.__getitem__(self.T*n + t)[0]
        return X


class mnist_contrastive(Dataset):
    def __init__(self, n, T, transform, train=True, seed=7):
        random.seed(seed)
        self.mnist = datasets.MNIST(root=dataset_dir, download=True, train=train, transform=transform)
        self.n = n
        self.T = T
        self.data_dim = (3, 64, 64)

        possible_cps = list(range(T // 4, 3 * T // 4 + 1))

        # change points of time series samples
        self.cps = [random.choice(possible_cps) for _ in range(self.n)]

        # digits of 2 groups before and after change point
        self.digit_pairs = [random.sample(range(10), 2) for _ in range(self.n)]

        # indices of samples in mnist grouped by digits
        self.indices_by_digit = {}
        indices = np.array(range(len(self.mnist.data)))
        for d in range(10):
            self.indices_by_digit[d] = indices[self.mnist.targets == d].tolist()

    def __len__(self):
        return self.n * self.T

    def __getitem__(self, idx):
        d1, d2 = random.sample(range(10), 2)
        indices1 = self.indices_by_digit[d1]
        indices2 = self.indices_by_digit[d2]
        d1_img1 = self.mnist[random.choice(indices1)][0]
        d1_img2 = self.mnist[random.choice(indices1)][0]
        d2_img1 = self.mnist[random.choice(indices2)][0]
        d2_img2 = self.mnist[random.choice(indices2)][0]

        if idx % 2 == 0:
            v = torch.bernoulli(torch.tensor(0.5)).item()
            if v == 0:
                sample = [torch.cat(3 * [d1_img1]), torch.cat(3 * [d1_img2])]
            else:
                sample = [torch.cat(3 * [d2_img1]), torch.cat(3 * [d2_img2])]
            label = 0
        else:
            sample = [torch.cat(3 * [d1_img1]), torch.cat(3 * [d2_img1])]
            label = 1

        return sample, label

    def get_normal(self, idx):
        row = idx // self.T
        t = idx % self.T

        # get label
        label = 2 * row if t < self.cps[row] else 2 * row + 1

        # get sample
        digit1, digit2 = self.digit_pairs[row]
        indices1 = self.indices_by_digit[digit1]
        indices2 = self.indices_by_digit[digit2]

        if t < self.cps[row]:
            sample = self.mnist[random.choice(indices1)][0]
            # label = self.mnist[random.choice(indices1)][1]
        else:
            sample = self.mnist[random.choice(indices2)][0]
            # label = self.mnist[random.choice(indices2)][1]

        return torch.cat(3 * [sample])

    def get_ts_sample(self, n):
        X = torch.empty(size=(self.T,) + self.data_dim)
        for t in range(self.T):
            X[t] = self.get_normal(self.T * n + t)[0]
        return X


class cifar10_vanilla(Dataset):
    def __init__(self, transform, train=True):
        self.cifar10 = datasets.CIFAR10(root=dataset_dir, download=True, train=train, transform=transform)
        self.data_dim = (3, 64, 64)

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, idx):
        sample = self.cifar10[idx][0]
        label = self.cifar10[idx][1]
        return sample, label

class cifar10_loader(Dataset):
    def __init__(self, n, T, transform, train=True, seed=7):
        random.seed(seed)
        self.cifar = datasets.CIFAR10(root=dataset_dir, download=True, train=train,
                                      transform=transform)
        self.n = n
        self.T = T
        self.data_dim = self.cifar[0][0].size()

        possible_cps = list(range(T//4, 3*T//4+1))
        
        # change points of time series samples
        self.cps = [random.choice(possible_cps) for _ in range(self.n)]

        # digits of 2 groups before and after change point
        self.digit_pairs = [random.sample(range(10), 2) for _ in range(self.n)]

        # indices of samples in mnist grouped by digits
        self.indices_by_digit = {}
        indices = np.array(range(len(self.cifar.data)))
        targets = np.asarray(self.cifar.targets) # labels in numpy array
        for d in range(10):
            self.indices_by_digit[d] = indices[targets == d].tolist()

    def __len__(self):
        return self.n * self.T
    
    def __getitem__(self, idx):
        row = idx // self.T
        t = idx % self.T

        # get label
        label = 2*row if t < self.cps[row] else 2*row + 1

        # get sample
        digit1, digit2 = self.digit_pairs[row]
        indices1 = self.indices_by_digit[digit1]
        indices2 = self.indices_by_digit[digit2]

        if t < self.cps[row]:
            sample = self.cifar[random.choice(indices1)][0]
        else:
            sample = self.cifar[random.choice(indices2)][0]
        sample = sample.view(self.data_dim)

        return sample, label

    def get_time_series_sample(self, n):
        X = torch.empty(size=(self.T,) + self.data_dim)
        for t in range(self.T):
            X[t] = self.__getitem__(self.T*n + t)[0]
        return X


class cifar10_contrastive(Dataset):
    def __init__(self, n, T, transform, train=True, seed=7):
        random.seed(seed)
        self.cifar = datasets.CIFAR10(root=dataset_dir, download=True, train=train,
                                      transform=transform)
        self.n = n
        self.T = T
        self.data_dim = (3, 64, 64)

        possible_cps = list(range(T // 4, 3 * T // 4 + 1))

        # change points of time series samples
        self.cps = [random.choice(possible_cps) for _ in range(self.n)]

        # digits of 2 groups before and after change point
        self.digit_pairs = [random.sample(range(10), 2) for _ in range(self.n)]

        # indices of samples in cifar10 grouped by digits
        self.indices_by_digit = {}
        indices = np.array(range(len(self.cifar.data)))
        targets = np.asarray(self.cifar.targets) # labels in numpy array
        for d in range(10):
            self.indices_by_digit[d] = indices[targets == d].tolist()

    def __len__(self):
        return self.n * self.T

    def __getitem__(self, idx):
        d1, d2 = random.sample(range(10), 2)
        indices1 = self.indices_by_digit[d1]
        indices2 = self.indices_by_digit[d2]
        d1_img1 = self.cifar[random.choice(indices1)][0]
        d1_img2 = self.cifar[random.choice(indices1)][0]
        d2_img1 = self.cifar[random.choice(indices2)][0]
        d2_img2 = self.cifar[random.choice(indices2)][0]

        if idx % 2 == 0:
            v = torch.bernoulli(torch.tensor(0.5)).item()
            if v == 0:
                sample = [d1_img1, d1_img2]
            else:
                sample = [d2_img1, d2_img2]
            label = 0
        else:
            sample = [d1_img1, d2_img1]
            label = 1

        return sample, label

    def get_normal(self, idx):
        row = idx // self.T
        t = idx % self.T

        # get label
        label = 2 * row if t < self.cps[row] else 2 * row + 1

        # get sample
        digit1, digit2 = self.digit_pairs[row]
        indices1 = self.indices_by_digit[digit1]
        indices2 = self.indices_by_digit[digit2]

        if t < self.cps[row]:
            sample = self.cifar[random.choice(indices1)][0]
            # label = self.mnist[random.choice(indices1)][1]
        else:
            sample = self.cifar[random.choice(indices2)][0]
            # label = self.mnist[random.choice(indices2)][1]

        return sample, label

    def get_ts_sample(self, n):
        X = torch.empty(size=(self.T,) + self.data_dim)
        for t in range(self.T):
            X[t] = self.get_normal(self.T * n + t)[0]
        return X



class mnist_contrastive(Dataset):
    def __init__(self, n, T, transform, train=True, seed=7):
        random.seed(seed)
        self.mnist = datasets.MNIST(root=dataset_dir, download=True, train=train, transform=transform)
        self.n = n
        self.T = T
        self.data_dim = (3, 64, 64)

        possible_cps = list(range(T // 4, 3 * T // 4 + 1))

        # change points of time series samples
        self.cps = [random.choice(possible_cps) for _ in range(self.n)]

        # digits of 2 groups before and after change point
        self.digit_pairs = [random.sample(range(10), 2) for _ in range(self.n)]

        # indices of samples in mnist grouped by digits
        self.indices_by_digit = {}
        indices = np.array(range(len(self.mnist.data)))
        for d in range(10):
            self.indices_by_digit[d] = indices[self.mnist.targets == d].tolist()

    def __len__(self):
        return self.n * self.T

    def __getitem__(self, idx):
        d1, d2 = random.sample(range(10), 2)
        indices1 = self.indices_by_digit[d1]
        indices2 = self.indices_by_digit[d2]
        d1_img1 = self.mnist[random.choice(indices1)][0]
        d1_img2 = self.mnist[random.choice(indices1)][0]
        d2_img1 = self.mnist[random.choice(indices2)][0]
        d2_img2 = self.mnist[random.choice(indices2)][0]

        if idx % 2 == 0:
            v = torch.bernoulli(torch.tensor(0.5)).item()
            if v == 0:
                sample = [torch.cat(3 * [d1_img1]), torch.cat(3 * [d1_img2])]
            else:
                sample = [torch.cat(3 * [d2_img1]), torch.cat(3 * [d2_img2])]
            label = 0
        else:
            sample = [torch.cat(3 * [d1_img1]), torch.cat(3 * [d2_img1])]
            label = 1

        return sample, label

    def get_normal(self, idx):
        row = idx // self.T
        t = idx % self.T

        # get label
        label = 2 * row if t < self.cps[row] else 2 * row + 1

        # get sample
        digit1, digit2 = self.digit_pairs[row]
        indices1 = self.indices_by_digit[digit1]
        indices2 = self.indices_by_digit[digit2]

        if t < self.cps[row]:
            sample = self.mnist[random.choice(indices1)][0]
            # label = self.mnist[random.choice(indices1)][1]
        else:
            sample = self.mnist[random.choice(indices2)][0]
            # label = self.mnist[random.choice(indices2)][1]

        return torch.cat(3 * [sample]), label

    def get_ts_sample(self, n):
        X = torch.empty(size=(self.T,) + self.data_dim)
        for t in range(self.T):
            X[t] = self.get_normal(self.T * n + t)[0]
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
            _, _, i, t = filename.split('_') # i-th image at time t
            t = int(t[:-4])
            i = int(i)
            cps[i] = cps.get(i, 0) + 1

        self.cps = cps
        self.n = len(cps)
        self.T = T

    def __len__(self):
        return self.n*self.T

    def __getitem__(self, item):
        i, t = item // self.T, item % self.T

        if t < self.cps[i]:
            label = 2*i
            file_name = path.join(self.dir1, 'CLEVR_nonsemantic_'+str(i).zfill(6)+'_'+str(t)+'.png')
            img = Image.open(file_name).convert('RGB')
        else:
            label = 2*i+1
            file_name = path.join(self.dir2, 'CLEVR_semantic_'+str(i).zfill(6)+'_'+str(t-self.cps[i])+'.png')
            img = Image.open(file_name).convert('RGB')

        return self.transform(img), label

    def get_time_series_sample(self, n):
        X = torch.empty(size=(self.T,) + self.data_dim)
        for t in range(self.T):
            X[t] = self.__getitem__(self.T*n + t)[0]
        return X