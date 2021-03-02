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



# directory where all datasets are saved
dataset_dir = '../datasets/'

#################################################################################
# data loaders for CelebA-based time series

# all samples before change points are men, all samples after change point are women
# each man sample is randomly selected from the set of all man samples in celeba
# each woman sample is randomly selected from the set of all woman samples in celeba
# label is the group number. there are 2n groups in total.
class celeba_gender_change(Dataset):
    def __init__(self, n, T, train=True, seed=7, transform=utils.trans_config1):
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

        for i in range(len(self.celeba)):
            if self.celeba[i][1][20] == 1:
                self.men_indices.append(i)
            else:
                self.women_indices.append(i)

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

# get samples in the same way as in celeba_gender_change,
# but the label is 1 for man, 0 for woman
class celeba_classification(celeba_gender_change):
    def __init__(self, n, T, train=True, seed=7):
        super().__init__(n, T, train, seed)
    
    def __getitem__(self, idx):
        label = 1 if idx in self.men_indices else 0

        if t < self.cps[row]:
            sample = self.celeba[random.choice(self.men_indices)][0]
        else:
            sample = self.mnist[random.choice(self.women_indices)][0]

        return sample, label
    
    def get_time_series_sample(self, n):
        X = torch.empty(size=(self.T,) + self.data_dim)
        for t in range(self.T):
            X[t] = self.__getitem__(self.T*n + t)[0]
        return X



# change in person
class celeba_change_person(Dataset):
    def __init__(self, n, T, train = True, seed = 7):
        random.seed(seed)
        split = 'train' if train else 'test'
        self.celeba = datasets.CelebA(root=dataset_dir, download=True, split=split,
                                        transform=utils.transform_config2)
        self.n = n
        self.T = T
        self.data_dim = self.celeba[0][0].size()

        possible_cps = list(range(2, T-1)) # [2,...,8] for T = 10
        self.cps = [random.choice(possible_cps) for _ in range(n)]
        self.groups_iterated = {}

    def __len__(self):
        return self.n*self.T
    
    def __getitem__(self, idx):
        t = idx % self.T
        row = idx // self.T
        label = 2*row if t < self.cps[row] else 2*row + 1

        if label not in self.groups_iterated:
            idx = random.choice(range(len(self.celeba)))
            self.groups_iterated[label] = idx
        sample = self.celeba[self.groups_iterated[label]][0]

        return sample, label
    
    def get_time_series_sample(self, n):
        X = torch.empty(size=(self.T,) + self.data_dim)
        for t in range(self.T):
            X[t] = self.__getitem__(self.T*n + t)[0]
        return X


####################################################################################
# Data loaders related to MNSIT-based time series datasets
class mnist_loader(Dataset):
    def __init__(self, n, T, cp_way=3, train=True, seed=7, model='linearvae'):
        if model == 'linearvae':
            transform = utils.transform_config5
        else:
            transform = utils.transform_config4
        
        random.seed(seed)
        self.mnist = datasets.MNIST(root=dataset_dir, download=True, train=train,
                                    transform=transform)
        self.n = n
        self.T = T
        self.data_dim = self.mnist[0][0].size()

        possible_cps = [] # possible change points
        if cp_way == 1: # fixed cp value
            possible_cps = [T//2]
        elif cp_way == 2: # set pf possible cp values
            possible_cps = [T//2, T//3, T//4]
        elif cp_way == 3: # interval of cp values
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
        else:
            sample = self.mnist[random.choice(indices2)][0]

        return sample, label

    def get_time_series_sample(self, n):
        X = torch.empty(size=(self.T,) + self.data_dim)
        for t in range(self.T):
            X[t] = self.__getitem__(self.T*n + t)[0]
        return X
        

class mnist_loader_repetitive(mnist_loader):
    def __init__(self, n, T, cp_way=3, train=True, seed=7, model='linearvae'):
        super().__init__(n, T, cp_way, train, seed, model)
        print(model)
        self.groups_iterated = {}
    
    def __getitem__(self, idx):
        row = idx // self.T
        t = idx % self.T

        # get label
        label = 2*row if t < self.cps[row] else 2*row + 1

        # get sample
        digit1, digit2 = self.digit_pairs[row]
        indices1 = self.indices_by_digit[digit1]
        indices2 = self.indices_by_digit[digit2]
        if label not in self.groups_iterated:
            # for each group/label, there is only 1 index
            if t < self.cps[row]:
                index = random.choice(indices1)
            else:
                index = random.choice(indices2)
            self.groups_iterated[label] = index
        sample = self.mnist[self.groups_iterated[label]][0]
        sample = sample.view(self.data_dim)

        return sample, label
    




class cifar10_loader(Dataset):
    def __init__(self, n, T, train=True, seed=7, transform=utils.trans_config):
        random.seed(seed)
        self.cifar = datasets.CIFAR10(root=dataset_dir, download=True, train=train, transform=transform)
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


class cifar10_loader_repetitive(Dataset):
    def __init__(self, n, T, train=True, seed=7):
        random.seed(seed)
        self.cifar = datasets.CIFAR10(root=dataset_dir, download=True, train=train,
                                    transform=utils.transform_config2)
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
        
        self.groups_iterated = {}

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
        if label not in self.groups_iterated:
            # for each group/label, there is only 1 index
            if t < self.cps[row]:
                index = random.choice(indices1)
            else:
                index = random.choice(indices2)
            self.groups_iterated[label] = index
        sample = self.cifar[self.groups_iterated[label]][0]

        return sample, label

    def get_time_series_sample(self, n):
        X = torch.empty(size=(self.T,) + self.data_dim)
        for t in range(self.T):
            X[t] = self.__getitem__(self.T*n + t)[0]
        return X



class clevr_change(Dataset):
    def __init__(self, name, T, transform_configuration):
        self.dir1 = path.join(dataset_dir, name, 'nsc_images/')
        self.dir2 = path.join(dataset_dir, name, 'sc_images/')
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