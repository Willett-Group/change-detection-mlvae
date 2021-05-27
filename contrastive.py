import os
import os.path as path
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from torchvision.utils import make_grid
import torchvision.transforms as transforms

import dataloaders
import networks
import utils

################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--model', type=str)

parser.add_argument('--N', type=int, default=1000)
parser.add_argument('--T', type=int, default=50)

parser.add_argument('--nepochs', type=int)
parser.add_argument('--batch_size', type=int, default=250)
parser.add_argument('--initial_lr', type=float, default=0.001)
parser.add_argument('--test', type=int, default=0)
parser.add_argument('--test_method', default='1')
# 1 for graph cut, using G(i, j) as the score
# 2 for

parser.add_argument('--log_file', default='log.txt')
parser.add_argument('--continue_saved', default=False)

parser.add_argument('--channels', default=3)
parser.add_argument('--dim_x', default=64)
parser.add_argument('--dim_y', default=64)

args = parser.parse_args()
img_shape = (args.channels, args.dim_x, args.dim_y)
################################################################################


def graph_cut(model, ds, X, eta):
    E_r = 0
    E_rc = 0
    E_r_count = 0
    E_rc_count = 0
    for d1 in range(ds.T):
        for d2 in range(ds.T):
            p = model.forward(X[d1].unsqueeze_(0), X[d2].unsqueeze_(0))
            p = torch.sigmoid(p).item()
            if p == 1:
                p -= 0.0001
            g = p / (1 - p)
            if (d1 <= eta and d2 < eta) or (d1 > eta and d2 > eta):
                E_rc += g
                E_rc_count += 1
            else:
                E_r += g
                E_r_count += 1
    score = E_r / E_r_count - E_rc / E_rc_count

    return score

def mean_distance(representation, ds, X, eta):


    return score


def representation(model, ds, X, M):
    random.seed(7)
    L = torch.zeros((M,) + ds.data_dim)
    for i in range(M):
        index = random.choice(list(range(ds.n * ds.T)))
        L[i] = torch.FloatTensor(ds.get_normal(index)[0])
    L = L.to(device)
    R = torch.zeros(X.size(0), M)
    for i in range(X.size(0)):
        R[i] = torch.FloatTensor([model.forward(X[i].unsqueeze_(0), L[m].unsqueeze_(0)) for m in range(M)])
        R[i] = torch.sigmoid(R[i])

    return R


def train(model, ds, root_dir):
    # should not shuffle here
    train_loader = DataLoader(ds, args.batch_size, shuffle=True, drop_last=False)

    # load saved models if load_saved flag is true
    if args.continue_saved:
        model.load_state_dict(torch.load(path.join(root_dir, 'model')))

    # optimizer definition
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.initial_lr
    )

    # load_saved is false when training is started from 0th iteration
    if not args.continue_saved:
        with open(path.join(root_dir, args.log_file), 'w') as log:
            log.write('Epoch\tIteration\tLoss\n')
    # initialize summary writer
    writer = SummaryWriter()

    # start training
    for epoch in range(0, args.nepochs):
        print('Epoch {}'.format(epoch))

        # the total loss at each epoch after running all iterations of batches
        iteration = 0
        correct = 0

        for batch_index, (X, y) in enumerate(train_loader):
            # set zero grad for the optimizer
            optimizer.zero_grad()

            # move data to cuda
            X1, X2 = X[0], X[1]
            X1 = X1.to(device=device)
            X2 = X2.to(device=device)
            y = y.to(device=device)

            p = model.forward(X1, X2)
            p = p.view(p.size(0))

            criterion = nn.BCEWithLogitsLoss()

            prediction = p > 0.5
            correct += (prediction == y).sum().item()
            # print(correct)

            loss = criterion(p, y.float())
            loss.backward()
            optimizer.step()

            # print losses
            if batch_index % 50 == 0 or batch_index == args.batch_size - 1:
                print('[%d/%d][%d/%d]\tLoss: %.4E' % (epoch, args.nepochs, batch_index, len(train_loader), loss.item()))

            # write to log
            with open(path.join(root_dir, args.log_file), 'a') as log:
                log.write('{0}\t{1}\t{2}\n'.format(
                    epoch,
                    batch_index,
                    loss.detach().item()
                ))

            # write to tensorboard
            itr = epoch * (int(len(ds) / args.batch_size) + 1) + batch_index
            writer.add_scalar('Loss', loss.detach().item(), itr)

        print('Training accuracy %.4E' % (correct / len(ds)))
        # save the model at every epoch
        torch.save(model.state_dict(), path.join(root_dir, 'model'))


def test(model, ds, dir):
    print("Running tests...")
    if args.test_method == '1':
        test_dir = 'errors_graph_cut'
    elif args.test_method == '2':
        test_dir = 'errors_representation_2a'
    test_dir_path = path.join(dir, test_dir)
    if not path.exists(test_dir_path):
        os.makedirs(test_dir_path)

    eta_hats = []  # save predicted change points
    etas = []
    # iterate over ts test samples X_1, X_2, etc...
    all_i = range(ds.n) if not args.dataset == 'clevr' else [args.T*6*(i-1)+j for i in range(1, 7) for j in range(5)]
    for i in all_i:
        etas.append(ds.cps[i])
        # load test sample X_i
        X = ds.get_ts_sample(i).to(device)

        scores = {}  # save errors for all candidate etas
        min_eta = 2
        max_eta = ds.T - 2
        max_score = -float('inf')
        eta_hat = -1

        for eta in range(min_eta, max_eta + 1):
            if args.test_method == '1':
                score = graph_cut(model, ds, X, eta)
            elif args.test_method == '2':
                r = representation(model, ds, X, 20)
                r1, r2 = r[0:eta], r[eta:ds.T]
                diff1 = r1 - torch.mean(r1, 0)
                diff2 = r1 - torch.mean(r2, 0)

                score = torch.sum(diff1.pow(2)).detach().item() + \
                          torch.sum(diff2.pow(2)).detach().item()

            scores[eta] = score
            if score > max_score:
                max_score = score
                eta_hat = eta
        eta_hats.append(eta_hat)

        # save errors
        plt.scatter(list(scores.keys()), list(scores.values()))
        plt.axvline(x=ds.cps[i])
        plt.axvline(x=eta_hat, color='r')
        plt.xlabel('etas (red: eta_hat, blue: true eta)')
        plt.ylabel('errors')
        plt.savefig(path.join(test_dir_path, 'X_{}_errors.png'.format(i)))
        plt.close()

    # compute mean of |eta-eta_hat| among all test samples
    diff = np.abs(np.asarray(etas) - np.asarray(eta_hats))
    score_mean = np.mean(diff)
    score_std = np.std(diff)
    # keep track of the errors associated with epochs
    with open(path.join(test_dir_path, 'test_stats.txt'), 'w') as f:
        json.dump({'mean': score_mean, 'std': score_std}, f, indent=2)
    # save etas and eta_hats
    with open(test_dir_path + '/cps.txt', 'w') as cps_r:
        for tmp in eta_hats:
            cps_r.write('{} '.format(tmp))
        cps_r.write('\n')
        for tmp in etas:
            cps_r.write('{} '.format(tmp))


if __name__ == '__main__':
    # create parent directories, like 'experiments/cifar10/linearmlvae_50'
    # and 'experiments/cifar10/dfcmlvae_128'
    dir0 = 'experiments'
    dir1 = path.join(dir0, args.dataset)
    dir2 = path.join(dir1, args.model)
    for d in [dir0, dir1, dir2]:
        if not path.exists(d):
            os.makedirs(d)

    # use cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Creating training and testing datasets...')
    trans = transforms.Compose([transforms.Resize([args.dim_x, args.dim_y]),
                                transforms.ToTensor()
                                ])
    if args.dataset == 'mnist':
        ds = dataloaders.mnist_contrastive(500, 50, train=True, transform=trans)
        ds_test = dataloaders.mnist_contrastive(50, args.T, train=False, seed=7, transform=trans)
    elif args.dataset == 'cifar10':
        ds = dataloaders.cifar10_contrastive(1000, 50, train=True, transform=trans)
        ds_test = dataloaders.cifar10_contrastive(50, args.T, train=False, seed=7, transform=trans)
    elif args.dataset == 'celeba':
        ds = dataloaders.celeba_contrastive(500, 50, train=True, transform=trans)
        ds_test = dataloaders.celeba_contrastive(50, args.T, train=False, seed=7, transform=trans)
    elif args.dataset == 'clevr':
        pass
    else:
        raise Exception("invalid dataset name")

    print('Creating models...')
    if args.model == 'contrastive':
        encoder = networks.resnet34()
        model = networks.TwoPathNetwork(3*64*64, predefined_encoder=encoder).to(device)
    else:
        raise Exception("invalid model name")

    existing_dirs = [int(f) for f in os.listdir(dir2) if f.isdigit()]
    if args.test == 0:
        # create new directory for this training run
        new = '1' if not existing_dirs else str(max(existing_dirs) + 1)
        # root dir is the directory of this particular run of experiment
        # all data produced by training and testing will be saved in this root dir
        root_dir = path.join(dir2, new)
        if not path.exists(root_dir):
            os.makedirs(root_dir)
        # save args
        with open(path.join(root_dir, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        train(model, ds, root_dir)
    else:
        for existing in existing_dirs:
            root_dir = path.join(dir2, str(existing))
            model.load_state_dict(torch.load(path.join(root_dir, 'model')))
            test(model, ds_test, root_dir)