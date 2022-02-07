import os
import os.path as path
import sys
import argparse
import logging
import glob
import json
import random
import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.utils import make_grid
import torchvision.transforms as transforms

import dataloaders
import networks
import utils
from TNN import Mining, Model
from TNN.Plot import scatter
from TNN.Loss_Fn import triplet_loss

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--method', type=str)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--save', type=str, default='TEST')
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--model_path', type=str)
args = parser.parse_args()

dirs = ['runs', 'runs_trash']
for d in dirs:
    os.makedirs(os.path.join(d, args.method), exist_ok=True)
run = dirs[1] if args.debug else dirs[0]
args.save = os.path.join(run, args.method, '{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S")))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def visualize(dist_matrix, eta, T):
    df = pd.DataFrame(columns=['d', 'group'])
    for t1 in range(eta):  # remember, 0, 1, ..., eta-1 are before change point, eta, ... , T-1 are after
        for t2 in range(eta, T):
            df = df.append({'d': dist_matrix[t1][t2], 'group': 'across'}, ignore_index=True)
    for t1 in range(eta):
        for t2 in range(t1 + 1, eta):
            df = df.append({'d': dist_matrix[t1][t2], 'group': 'g1'}, ignore_index=True)
    for t1 in range(eta, T):
        for t2 in range(t1 + 1, T):
            df = df.append({'d': dist_matrix[t1][t2], 'group': 'g2'}, ignore_index=True)

    return df


def measure(dist_matrix, eta, T, method=1):
    term1, term2, term3 = 0, 0, 0

    for t1 in range(eta):  # remember, 0, 1, ..., eta-1 are before change point, eta, ... , T-1 are after
        for t2 in range(eta, T):
            term1 += dist_matrix[t1][t2]
    for t1 in range(eta):
        for t2 in range(t1 + 1, eta):
            term2 += dist_matrix[t1][t2]
    for t1 in range(eta, T):
        for t2 in range(t1 + 1, T):
            term3 += dist_matrix[t1][t2]
    if method == 1:
        w1 = 1 / (eta * (T - eta))
        w2 = -1 / (eta * (eta - 1))
        w3 = -1 / ((T - eta) * (T - eta - 1))
    elif method == 2:
        w1 = 1 / (eta * (T - eta))
        w2 = -1 / (eta * (eta - 1) + (T - eta) * (T - eta - 1))
        w3 = -1 / (eta * (eta - 1) + (T - eta) * (T - eta - 1))

    return w1 * term1 + w2 * term2 + w3 * term3


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(device)
    ngpu = torch.cuda.device_count()
    logging.info('ngpu = %d', ngpu)
    gpus = list(range(ngpu))

    set_seed(args.seed)
    ds_test = dataloaders.MCCON(args.dataset, is_train=False, n_max=200, t_max=50, p_max=50)
    dims = ds_test.dims  # tensor
    N, T = ds_test.N, ds_test.T
    if args.method == 'triplet':
        model = Model.TNN_CIFAR10(input_shape=dims, output_size=dims[1]).to(device)
        model.load_state_dict(torch.load(args.model_path))
    model.eval()

    eta_hats = []  # save predicted change points
    etas = []

    for i in range(N):
        logging.info("Time series sample X_%d" % i)

        etas.append(ds_test.cps[i])
        X = ds_test.get_x_i(i).to(device)
        grid = make_grid(X, nrow=T)
        save_image(grid, path.join(args.save, 'X_{}.png'.format(i)))

        scores = {}  # save errors for all candidate etas
        min_eta = 2
        max_eta = T - 2
        max_score = -float('inf')
        eta_hat = -1

        with torch.no_grad():
            embeddings = model(X)
        distances = Mining._pairwise_distances(embeddings, squared=True, device=device).cpu().numpy()

        df = visualize(distances, ds_test.cps[i], T)
        # df.to_csv(path.join(args.save, 'X_%d.csv' % i))
        distribution = sns.displot(df, x="d", hue="group", element="step")
        plt.savefig(path.join(args.save, 'X_%d_dist.png' % i))
        plt.close()

        # computer L(eta) for each eta
        for eta in range(min_eta, max_eta + 1):
            score = measure(distances, eta, T, method=1)
            scores[eta] = score
            if score > max_score:
                max_score = score
                eta_hat = eta
        eta_hats.append(eta_hat)

        # save errors
        plt.scatter(list(scores.keys()), list(scores.values()))
        plt.axvline(x=ds_test.cps[i])
        plt.axvline(x=eta_hat, color='r')
        plt.xlabel('etas (red: eta_hat, blue: true eta)')
        plt.ylabel('scores')
        plt.savefig(path.join(args.save, 'X_{}_errors.png'.format(i)))
        plt.close()

    # compute mean of |eta-eta_hat| among all test samples
    diff = np.abs(np.asarray(etas) - np.asarray(eta_hats))
    score_mean = np.mean(diff)
    score_std = np.std(diff)
    # keep track of the errors associated with epochs
    with open(path.join(args.save, 'test_stats.txt'), 'w') as f:
        json.dump({'mean': score_mean, 'std': score_std}, f, indent=2)
    # save etas and eta_hats
    with open(args.save + '/cps.txt', 'w') as cps_r:
        for tmp in eta_hats:
            cps_r.write('{} '.format(tmp))
        cps_r.write('\n')
        for tmp in etas:
            cps_r.write('{} '.format(tmp))


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Total test time: %ds', duration)
