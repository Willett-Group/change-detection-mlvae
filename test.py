"""
Compute score matrix of size T by T and compute L(eta) for each eta
"""

import sys
import time
import glob
import argparse
import logging
import json
import random
from math import comb
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from torchvision.utils import make_grid

import datasets
import networks
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--datapath', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='mnist', help='which dataset')
parser.add_argument('--n_max', type=int, default=1000, help='numbe of time series samples')
parser.add_argument('--t_max', type=int, default=50, help='number of timestamps in a time series sample')
parser.add_argument('--method', type=str, default='siamese_p', help='which method of computing the matrix')
parser.add_argument('--loss_variant', type=int, default=1, help='which variant of the loss on matrix')

parser.add_argument('--model_path', type=str, default='TRAIN-20220213-200526', help='path of pre-trained weights')
parser.add_argument('--save', type=str, default='TEST', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed for reproducible test dataset and results')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
args = parser.parse_args()
# save each test run in the train directory
args.save = Path('runs', args.model_path, f'{args.save}-{time.strftime("%Y%m%d-%H%M%S")}')
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))


# logging
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(Path(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def PairsGroupsLoss(scores, eta, variant):
    l1 = utils.AvgrageMeter()
    l2 = utils.AvgrageMeter()
    l3 = utils.AvgrageMeter()

    T = scores.shape[0]
    for t1 in range(T):
        for t2 in range(t1+1, T):
            s = scores[t1][t2]
            if t1 < eta and t2 >= eta:
                l1.update(s, 1)
            elif t1 < eta and t2 < eta:
                l2.update(s, 1)
            else:
                l3.update(s, 1)
    assert l1.cnt == eta*(T-eta)
    assert l2.cnt == comb(eta, 2)
    assert l3.cnt == comb(T-eta, 2)
    
    if variant == 0:
        l = l2.avg + l3.avg - 2*l1.avg
    else:
        l = (l2.sum + l3.sum) / (l2.cnt + l3.cnt) - l1.avg
    
    return l


def main():
    logging.info("args = %s", args)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(args.seed)
    cudnn.benchmark = True
    cudnn.enabled = True

    classes = {
        'mnist': range(10),
        'cifar10': range(10),
        'cifar100': range(100),
        'celeba': [4, 9, 17, 20, 24]
    }
    test_data_ts = datasets.TS(
        datapath=args.datapath,
        dataset=args.dataset,
        split='valid',
        n_max=args.n_max,
        t_max=args.t_max,
        classes=classes[args.dataset],
        transform=utils.transforms[args.dataset])
    test_queue_ts = DataLoader(
        test_data_ts,
        shuffle=False,
        num_workers=args.num_workers,
        batch_size=test_data_ts.T)

    ckpt = torch.load(Path('runs', args.model_path, 'checkpoint.pth.tar'))
    if args.dataset in ['mnist', 'cifar10', 'cifar100']:
        model = networks.SiameseNet32(arch='cnn').to(device)
    else:
        model = networks.SiameseNet128(arch='cnn').to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    errors = utils.AvgrageMeter()
    for step, (input, _) in enumerate(test_queue_ts): # step = n, input = X_n
        input = input.to(device)
        T = input.size(0)


        scores = np.zeros((T, T))
        for t1 in range(T):
            for t2 in range(T):
                if args.method == 'naive':
                    s = torch.norm(input[t1]-input[t2])
                elif args.method == 'mlvae':
                    pass
                elif args.method == 'siamese_p':
                    s = model(torch.unsqueeze(torch.stack((input[t1], input[t2]), dim=0), 0))
                    s = torch.sigmoid(s).item()
                elif args.method == 'siamese_logp':
                    s = model(torch.unsqueeze(torch.stack((input[t1], input[t2]), dim=0), 0))
                    s = torch.sigmoid(s).item()
                    if s == 1:
                        s -= 0.0001
                    s = np.log(s / (1-s))
                elif args.method == 'siamese_dist':
                    pass
                else:
                    raise Exception("incorrect method to compute the scores matrix")
                scores[t1][t2] = s

        ls = {} # eta: L(eta)
        min_l = float('inf')
        eta_hat = None
        min_t = 2
        for eta in range(min_t, test_data_ts.T - min_t + 1):
            l = PairsGroupsLoss(scores, eta, variant=args.loss_variant)
            ls[eta] = l
            if l < min_l:
                min_l = l
                eta_hat = eta
        
        eta = test_data_ts.splits[step][0][0]
        err = np.abs(eta - eta_hat)
        errors.update(err)
        pct_0 = utils.percent_by_bound(errors.values, 0)
        pct_1 = utils.percent_by_bound(errors.values, 1)
        pct_2 = utils.percent_by_bound(errors.values, 2)
        pct_5 = utils.percent_by_bound(errors.values, 5)
        pct_10 = utils.percent_by_bound(errors.values, 10)
        logging.info(f'[{step}] err {errors.avg:03f} err_std {np.std(errors.values):03f}')
        logging.info(f'[{step}] pct_0 {pct_0:03f} pct_1 {pct_1:03f} pct_2 {pct_2:03f} pct_5 {pct_5:03f} pct_10 {pct_10:03f}')
        
        if err > 1: # visualize bad predictions
            save_image(make_grid(input, nrow=T), Path(args.save, f'X_{step}.png'))
            plt.scatter(list(ls.keys()), list(ls.values()))
            plt.axvline(x=eta, color='b')
            plt.axvline(x=eta_hat, color='r')
            plt.xlabel('eta (red: prediction, blue: groundtruth)')
            plt.ylabel('L(eta)')
            plt.savefig(Path(args.save, 'X_{}_errors.png'.format(step)))
            plt.close()


if __name__ == '__main__':
    main()