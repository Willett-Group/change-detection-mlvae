import torch
import random
import pickle
import os
import numpy as np
from PIL import Image

import utils
from torchvision import datasets
from torch.utils.data import Dataset
from torchvision.utils import save_image

dataset_dir = '/home/renyi/Documents/datasets/'

'''
n = 10000
ds_train = datasets.CelebA(root=root_dir, download=True, split='train', transform=utils.transform_config)
men = []
women = []
for i in range(n):
    if ds_train[i][1][20] == 1:
        men.append(i)
    else:
        women.append(i)
with open(os.path.join(root_dir, 'celeba', 'men.pickle'), 'wb') as f:
    pickle.dump(men, f)
with open(os.path.join(root_dir, 'celeba', 'women.pickle'), 'wb') as f:
    pickle.dump(women, f)
'''

'''
n = 1000
ds_train = datasets.CelebA(root=root_dir, download=True, split='test', transform=utils.transform_config)
men = []
women = []
for i in range(n):
    if ds_train[i][1][20] == 1:
        men.append(i)
    else:
        women.append(i)
with open(os.path.join(root_dir, 'celeba', 'men1.pickle'), 'wb') as f:
    pickle.dump(men, f)
with open(os.path.join(root_dir, 'celeba', 'women1.pickle'), 'wb') as f:
    pickle.dump(women, f)
'''


class celeba(Dataset):
    def __init__(self, n, T):
        self.n = n
        self.T = int(T)
        self.ds_train = datasets.CelebA(root='/media/renyi/HDD/', download=True, split='train', transform=utils.transform_config2)
        self.data_dim = self.ds_train[0][0].size()

        possible_cps = [2,3,4,5,6,7,8]
        random.seed(7)
        cps = [random.sample(possible_cps, 1)[0] for _ in range(n)]
        self.cps = cps
        self.men_indices = [i for i in range(n*T) if (i%T < cps[i//T])]
        self.women_indices = [i for i in range(n*T) if (i%T >= cps[i//T])]

        with open(os.path.join(root_dir, 'celeba', 'men.pickle'), 'rb') as f:
            self.men = pickle.load(f)
        with open(os.path.join(root_dir, 'celeba', 'women.pickle'), 'rb') as f:
            self.women = pickle.load(f)
        
        self.map = {}
        l1 = len(self.men_indices)
        l2 = len(self.men)
        l3 = len(self.women_indices)
        l4 = len(self.women)
        for i in range(l1):
            self.map[self.men_indices[i]] = self.men[i % l2]
        for i in range(l3):
            self.map[self.women_indices[i]] = self.women[i % l4]

    def __len__(self):
        return self.n*self.T
    
    def __getitem__(self, idx):
        if idx%self.T < self.cps[idx//self.T]:
            label = 2*(idx//self.T)
        else:
            label = 2*(idx//self.T) + 1

        return self.ds_train[self.map[idx]][0], label
    

class celeba_test(Dataset):
    def __init__(self, N, T):
        self.N = N
        self.T = int(T)
        self.ds_train = datasets.CelebA(root='/media/renyi/HDD/', download=True, split='test', transform=utils.transform_config2)
        self.data_dim = self.ds_train[0][0].size()

        possible_cps = [2,3,4,5,6,7,8]
        random.seed(7)
        cps = [random.sample(possible_cps, 1)[0] for _ in range(N)]
        self.cps = cps
        self.men_indices = [i for i in range(N*T) if (i%T < cps[i//T])]
        self.women_indices = [i for i in range(N*T) if (i%T >= cps[i//T])]

        with open(os.path.join(root_dir, 'celeba', 'men1.pickle'), 'rb') as f:
            self.men = pickle.load(f)
        with open(os.path.join(root_dir, 'celeba', 'women1.pickle'), 'rb') as f:
            self.women = pickle.load(f)
        
        self.map = {}
        l1 = len(self.men_indices)
        l2 = len(self.men)
        l3 = len(self.women_indices)
        l4 = len(self.women)
        for i in range(l1):
            self.map[self.men_indices[i]] = self.men[i % l2]
        for i in range(l3):
            self.map[self.women_indices[i]] = self.women[i % l4]

    def __len__(self):
        return self.N*self.T
    
    def __getitem__(self, idx):
        if idx%self.T < self.cps[idx//self.T]:
            label = 2*(idx//self.T)
        else:
            label = 2*(idx//self.T) + 1

        return self.ds_train[self.map[idx]][0], label
    
    def get_time_series_sample(self, n):
        X = torch.empty(size=(self.T,) + self.data_dim)
        for t in range(self.T):
            X[t] = self.__getitem__(self.T*n + t)[0]
        return X




class celeba_classification(Dataset):
    def __init__(self, n, T):
        self.n = n
        self.T = int(T)
        self.ds_train = datasets.CelebA(root='/media/renyi/HDD/', download=True, split='train', transform=utils.transform_config2)
        self.data_dim = self.ds_train[0][0].size()

        possible_cps = [2,3,4,5,6,7,8]
        random.seed(7)
        cps = [random.sample(possible_cps, 1)[0] for _ in range(n)]
        self.cps = cps
        self.men_indices = [i for i in range(n*T) if (i%T < cps[i//T])]
        self.women_indices = [i for i in range(n*T) if (i%T >= cps[i//T])]

        with open(os.path.join(root_dir, 'celeba', 'men.pickle'), 'rb') as f:
            self.men = pickle.load(f)
        with open(os.path.join(root_dir, 'celeba', 'women.pickle'), 'rb') as f:
            self.women = pickle.load(f)
        
        self.map = {}
        l1 = len(self.men_indices)
        l2 = len(self.men)
        l3 = len(self.women_indices)
        l4 = len(self.women)
        for i in range(l1):
            self.map[self.men_indices[i]] = self.men[i % l2]
        for i in range(l3):
            self.map[self.women_indices[i]] = self.women[i % l4]

    def __len__(self):
        return self.n*self.T
    
    def __getitem__(self, idx):
        if idx in self.men_indices:
            label = 1
        else:
            label = 0

        return self.ds_train[self.map[idx]][0], label


class celeba_test_classification(Dataset):
    def __init__(self, N, T):
        self.N = N
        self.T = int(T)
        self.ds_train = datasets.CelebA(root='/media/renyi/HDD/', download=True, split='test', transform=utils.transform_config2)
        self.data_dim = self.ds_train[0][0].size()

        possible_cps = [2,3,4,5,6,7,8]
        random.seed(7)
        cps = [random.sample(possible_cps, 1)[0] for _ in range(N)]
        self.cps = cps
        self.men_indices = [i for i in range(N*T) if (i%T < cps[i//T])]
        self.women_indices = [i for i in range(N*T) if (i%T >= cps[i//T])]

        with open(os.path.join(root_dir, 'celeba', 'men1.pickle'), 'rb') as f:
            self.men = pickle.load(f)
        with open(os.path.join(root_dir, 'celeba', 'women1.pickle'), 'rb') as f:
            self.women = pickle.load(f)
        
        self.map = {}
        l1 = len(self.men_indices)
        l2 = len(self.men)
        l3 = len(self.women_indices)
        l4 = len(self.women)
        for i in range(l1):
            self.map[self.men_indices[i]] = self.men[i % l2]
        for i in range(l3):
            self.map[self.women_indices[i]] = self.women[i % l4]

    def __len__(self):
        return self.N*self.T
    
    def __getitem__(self, idx):
        if idx in self.men_indices:
            label = 1
        else:
            label = 0

        return self.ds_train[self.map[idx]][0], label
    
    def get_time_series_sample(self, n):
        X = torch.empty(size=(self.T,) + self.data_dim)
        for t in range(self.T):
            X[t] = self.__getitem__(self.T*n + t)[0]
        return X




class celeba_change_person(Dataset):
    def __init__(self, n, T, seed, train=True):
        if train:
            self.ds_train = datasets.CelebA(root=dataset_dir, download=True, split='train',
            transform=utils.transform_config2)
        else:
            self.ds_train = datasets.CelebA(root=dataset_dir, download=True, split='test',
            transform=utils.transform_config2)
        self.n = n
        self.T = int(T)
        self.data_dim = self.ds_train[0][0].size()

        possible_cps = [2,3,4,5,6,7,8]
        random.seed(seed)
        cps = [random.sample(possible_cps, 1)[0] for _ in range(n)]
        self.cps = cps
        self.groups_iterated = {}

    def __len__(self):
        return self.n*self.T
    
    def __getitem__(self, idx):
        t = idx % self.T
        row = idx // self.T
        label = 2*row if t < self.cps[row] else 2*row + 1

        if label not in self.groups_iterated:
            idx = random.choice(range(10000))
            self.groups_iterated[label] = idx
        sample = self.ds_train[self.groups_iterated[label]][0]

        return sample, label
    
    def get_time_series_sample(self, n):
        X = torch.empty(size=(self.T,) + self.data_dim)
        for t in range(self.T):
            X[t] = self.__getitem__(self.T*n + t)[0]
        return X










class experiment1(Dataset):
    def __init__(self):
        self.mnist = datasets.MNIST(root='mnist', download=True, train=True, transform=transform_config)
        self.mnist.data = np.true_divide(self.mnist.data, 255)
        # data is X, label is y
        outputs_to_concat = []
        for idx in range(5):
            indices1 = self.mnist.targets == 2*idx
            tmp1 = self.mnist.data[indices1]
            first_5000 = tmp1.view(tmp1.size(0), -1)[0:5000]
            first_5000 = torch.transpose(first_5000, 0, 1)

            indices2 = self.mnist.targets == (2*idx+1)
            tmp2 = self.mnist.data[indices2]
            second_5000 = tmp2.view(tmp2.size(0), -1)[0:5000]
            second_5000 = torch.transpose(second_5000, 0, 1)
        
            row = torch.cat((first_5000, second_5000), dim=1)
            outputs_to_concat.append(row)

        self.sample = torch.stack(outputs_to_concat, dim=0)


    def __len__(self):
        return 50000
    
    def __getitem__(self, idx):
        d1 = idx // 10000
        d2 = idx % 10000
        if d2 < 5000:
            label = 2*d1
        else:
            label = 2*d1 + 1

        return (self.sample[d1, :, d2].view(1, 28, 28), label)


class experiment3(Dataset):
    def __init__(self, n, T, cp_way):
        self.n = n
        self.T = int(T)
        self.mnist = datasets.MNIST(root='mnist', download=True, train=True, transform=transform_config)
        self.mnist.data = np.true_divide(self.mnist.data, 255)
        self.labels = []
        self.cps = []

        # data is X, label is y
        outputs_to_concat = []
        possible_cps = []
        if cp_way == 1: # fixed cp value
            possible_cps = [T//2]
        elif cp_way == 2: # set pf possible cp values
            possible_cps = [T//2, T//3, T//4]
        elif cp_way == 3: # interval of cp values
            possible_cps = list(range(T//4, 3*T//4+1, 1))
        
        for i in range(n):
            i1, i2 = random.sample(range(10), 2) # sample 2 digits
            cp = random.sample(possible_cps, 1)[0] # sample change point

            data1 = self.mnist.data[self.mnist.targets == i1]
            idx1 = random.sample(range(data1.size(0)), cp)
            part1 = data1.view(data1.size(0), -1)[idx1]
            part1 = torch.transpose(part1, 0, 1) # convert it to 784 by l dimension

            data2 = self.mnist.data[self.mnist.targets == i2]
            idx2 = random.sample(range(data2.size(0)), T-cp)
            part2 = data2.view(data2.size(0), -1)[idx2]
            part2 = torch.transpose(part2, 0, 1)
        
            row = torch.cat((part1, part2), dim=1)
            outputs_to_concat.append(row)

            # i1, i2 if repetitive labels, otherwisse just 2n many
            self.labels.append([2*i, 2*i+1])
            self.cps.append(cp)

        self.sample = torch.stack(outputs_to_concat, dim=0) # get the finally formatted date

    def __len__(self):
        return self.n*self.T
    
    def __getitem__(self, idx):
        d1 = idx // self.T
        d2 = idx % self.T
        if d2 < self.cps[d1]:
            label = self.labels[d1][0]
        else:
            label = self.labels[d1][1]

        return (self.sample[d1, :, d2], label)