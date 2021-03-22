import os
import os.path as path
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from torchvision.utils import make_grid

import dataloaders
import networks
import utils

################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--N', type=int, default=1000)
parser.add_argument('--T', type=int, default=50)

#################################################################################

def main():
    print('Initializing training and testing datasets...')
    if args.dataset == 'mnist':
        #ds = dataloaders.mnist_loader(args.T, args.T, train=True, seed=7, transform=utils.trans_config)
        ds_test = dataloaders.mnist_loader(100, args.T, train=False, seed=7, transform=utils.trans_config)
    elif args.dataset == 'cifar10':
        #ds = dataloaders.cifar10_loader(args.N, args.T, train=True, seed=7, transform=utils.trans_config)
        ds_test = dataloaders.cifar10_loader(100, args.T, train=False, seed=7, transform=utils.trans_config)
    elif args.dataset == 'celeba':
        #ds = dataloaders.celeba_gender_change(args.N, args.T, train=True, seed=7, transform=utils.trans_config1)
        ds_test = dataloaders.celeba_gender_change(100, args.T, train=False, seed=7, transform=utils.trans_config1)
    elif args.dataset == 'clevr':
        #ds = dataloaders.clevr_change('n=2100T=50', args.T, utils.trans_config1_special)
        ds_test = dataloaders.clevr_change('n=2100T=50', args.T, utils.trans_config1_special)
    else:
        raise Exception("invalid dataset name")

    # create new directory for this training run
    root_dir = path.join(dir1, 'originals')
    if not path.exists(root_dir):
        os.makedirs(root_dir)

    if args.dataset == 'clevr':
        all_i = [args.T * 6 * (i - 1) + j for i in range(1, 7) for j in range(5)]
    else:
        all_i = range(ds_test.n)


    etas = []
    eta_hats = []
    for i in all_i:
        etas.append(ds_test.cps[i])
        # load the test sample X_i
        X = ds_test.get_time_series_sample(i)
        X = X.to(device=device)

        errors = {}  # save errors for all candidate etas
        min_eta = 2
        max_eta = ds_test.T - 2
        min_total_error = float('inf')
        eta_hat, min_diff1, min_diff2 = -1, None, None
        for eta in range(min_eta, max_eta + 1):
            G1 = X[0:eta]
            G2 = X[eta:ds_test.T]
            diff1 = G1 - torch.mean(G1, 0)
            diff2 = G2 - torch.mean(G2, 0)

            total_error = torch.sum(diff1.pow(2)).detach().item() + \
                          torch.sum(diff2.pow(2)).detach().item()
            errors[eta] = total_error
            if total_error < min_total_error:
                min_total_error = total_error
                eta_hat = eta
                min_diff1 = diff1
                min_diff2 = diff2
        eta_hats.append(eta_hat)

        grid = make_grid(torch.cat([X,
                                    min_diff1, min_diff2
                                    ]), nrow=ds_test.T)
        save_image(grid, path.join(root_dir, 'X_{}.png'.format(i)))

        # save square errors
        plt.scatter(list(errors.keys()), list(errors.values()))
        plt.axvline(x=ds_test.cps[i])
        plt.axvline(x=eta_hat, color='r')
        plt.xlabel('etas (red: eta_hat, blue: true eta)')
        plt.ylabel('squared errors')
        plt.savefig(path.join(root_dir, 'X_{}_errors.png'.format(i)))
        plt.close()

    diff = np.abs(np.asarray(etas) - np.asarray(eta_hats))
    error = np.mean(diff)
    with open(path.join(root_dir, 'error.txt'), 'w') as f:
        json.dump({'error': error}, f, indent=2)



if __name__ == '__main__':
    args = parser.parse_args()
    # create parent directories, like 'experiments/cifar10/linearmlvae_50'
    # and 'experiments/cifar10/dfcmlvae_128'
    dir0 = 'experiments'
    dir1 = path.join(dir0, args.dataset)
    for d in [dir0, dir1]:
        if not path.exists(d):
            os.makedirs(d)

    # use cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # start training and/or testing
    main()
