import os
import os.path as path
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from torchvision.utils import make_grid
import torchvision.transforms as transforms

import datasets
import networks
import utils

################################################################################
# things that make reconstructions worse (blurry, not able to see outlines...):
# too large or too small cs_dim
# too small number of training epochs
# larger beta values
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--N', type=int, default=1000)
parser.add_argument('--T', type=int, default=50)
parser.add_argument('--model', type=str)
parser.add_argument('--cs_dim', type=int)

parser.add_argument('--nepochs', type=int)
parser.add_argument('--batch_size', type=int, default=250)
parser.add_argument('--initial_lr', type=float, default=0.001)
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--test', type=int, default=0)
parser.add_argument('--test_method', default='graph-cut')

parser.add_argument('--log_file', default='log.txt')
parser.add_argument('--continue_saved', default=False)

parser.add_argument('--iterations', default=20)

parser.add_argument('--channels', default=3)
parser.add_argument('--dim_x', default=64)
parser.add_argument('--dim_y', default=64)

args = parser.parse_args()
img_shape = (args.channels, args.dim_x, args.dim_y)
################################################################################

def test(model, ds, dir, iterations):
    print("Running tests...")
    if args.test_method == "graph-cut":
        test_dir = "errors_graph_cut"
    else:
        test_dir = "errors_mean_distance"
    test_dir_path = path.join(dir, test_dir)
    if not path.exists(test_dir_path):
        os.makedirs(test_dir_path)

    # start testing
    eta_hats = []  # save predicted change points
    etas = []
    # iterate over test samples X_1, X_2, etc...
    all_i = range(ds.n) if not args.dataset == "clevr" else [args.T*6*(i-1)+j for i in range(1, 7) for j in range(5)]
    for i in all_i:
        etas.append(ds.cps[i])
        # load test sample X_i
        X = ds.get_time_series_sample(i).to(device)

        scores = {}  # save errors for all candidate etas
        min_eta = 2
        max_eta = ds.T - 2
        max_score = -float("inf")
        eta_hat, min_G1, min_G2 = -1, None, None
        for eta in range(min_eta, max_eta + 1):
            # get reconstructions and errors of 2 groups
            if args.test_method == "graph-cut":
                score = graph_cut(model, ds, X, eta, only_c=False)
            else:
                get_recon = get_recon_plain if iterations == 0 else get_recon_minimize
                G1, G1_error = get_recon(X[0:eta], torch.zeros(eta, 1), model)
                G2, G2_error = get_recon(X[eta:ds.T], torch.zeros(ds.T - eta, 1), model)
                score = - (G1_error.detach().item() + G2_error.detach().item())
            scores[eta] = score.detach().item()
            if score > max_score:
                max_score = score
                eta_hat = eta
                # min_G1 = G1
                # min_G2 = G2
        eta_hats.append(eta_hat)

        # # decode(s=0, c)
        # G1_onlyc, _ = get_recon_onlyc(X[0:eta_hat], torch.zeros(eta_hat, 1), model)
        # G2_onlyc, _ = get_recon_onlyc(X[eta_hat:ds.T], torch.zeros(ds.T-eta_hat, 1), model)
        #
        # # color change points
        # # blue strip for original images
        # X[etas[i] - 1][0, :, -5:-1] = X[etas[i] - 1][1, :, -5:-1] = 0
        # X[etas[i] - 1][2, :, -5:-1] = 255
        # # red strip for reconstructions
        # min_G1[-1][0, :, -5:-1] = 255
        # min_G1[-1][1, :, -5:-1] = min_G1[-1][2, :, -5:-1] = 0
        # # red strip for fixed c
        # G1_onlyc[-1][0, :, -5:-1] = 255
        # G1_onlyc[-1][1, :, -5:-1] = G1_onlyc[-1][2, :, -5:-1] = 0
        # grid = make_grid(torch.cat([X,   min_G1, min_G2,   G1_onlyc, G2_onlyc]), nrow=ds.T)
        # save_image(grid, path.join(test_dir_path, "X_{}.png".format(i)))

        # save errors
        plt.scatter(list(scores.keys()), list(scores.values()))
        plt.axvline(x=ds.cps[i])
        plt.axvline(x=eta_hat, color="r")
        plt.xlabel("etas (red: eta_hat, blue: true eta)")
        plt.ylabel("errors")
        plt.savefig(path.join(test_dir_path, "X_{}_errors.png".format(i)))
        plt.close()

    # compute mean of |eta-eta_hat| among all test samples
    diff = np.abs(np.asarray(etas) - np.asarray(eta_hats))
    score_mean = np.mean(diff)
    score_std = np.std(diff)
    # keep track of the errors associated with epochs
    with open(path.join(test_dir_path, "error.txt"), "w") as f:
        json.dump({"mean": score_mean, "std": score_std}, f, indent=2)
    # save etas and eta_hats
    with open(test_dir_path + "/cps.txt", "w") as cps_r:
        for tmp in eta_hats:
            cps_r.write("{} ".format(tmp))
        cps_r.write("\n")
        for tmp in etas:
            cps_r.write("{} ".format(tmp))

def graph_cut(model, ds, X, eta, only_c = True):
    s_mu, _, c_mu, _ = model.encode(X)

    E_r = 0
    E_rc = 0
    E_r_count = 0
    E_rc_count = 0
    for d1 in range(ds.T):
        for d2 in range(ds.T):
            if (d1 <= eta and d2 < eta) or (d1 > eta and d2 > eta):
                E_rc += torch.sum(torch.square(c_mu[d1] - c_mu[d2]))
                if not only_c:
                    E_rc += torch.sum(torch.square(s_mu[d1] - s_mu[d2]))
                E_rc_count += 1
            else:
                E_r += torch.sum(torch.square(c_mu[d1] - c_mu[d2]))
                if not only_c:
                    E_r += torch.sum(torch.square(s_mu[d1] - s_mu[d2]))
                E_r_count += 1
    score = E_r / E_r_count - E_rc / E_rc_count

    return score

def get_recon_plain(X, y, model):
    # style is individual, content is group
    s_mu, s_logvar, c_mu, c_logvar = model.encode(X)
    # put all content stuff into group in the grouping/evidence-accumulation stage
    group_mu, group_logvar, _, _ = utils.accumulate_group_evidence(
        c_mu.data, c_logvar.data, y
    )
    """
    sampling from group mu and logvar for each image in mini-batch differently makes
    the decoder consider content latent embeddings as random noise and ignore them
    """
    s_z = utils.reparameterize(mu=s_mu, logvar=s_logvar, training=False)
    c_z = utils.group_wise_reparameterize(
        mu=group_mu, logvar=group_logvar, labels_batch=y, cuda=True, training=False
    )
    recon = model.decode(s_z, c_z)
    recon_error = torch.sum((recon - X).pow(2))

    return recon, recon_error


def get_recon_minimize(X, y, model):
    s_mu, s_logvar, c_mu, c_logvar = model.encode(X)
    group_mu, group_logvar, _, _ = utils.accumulate_group_evidence(
        c_mu.data, c_logvar.data, y
    )
    # 2 variables to optimize wrt
    s_optimize = s_mu.clone().detach().requires_grad_(True)
    c_optimize = group_mu[0].clone().detach().requires_grad_(True)
    # c is c_optimized stacked vertically batch_size times
    c = c_optimize.expand(s_mu.size(0), -1)

    optimizer = torch.optim.Adam(
        [s_optimize, c_optimize]
    )

    for itr in range(int(args.iterations)):
        optimizer.zero_grad()

        # reconstruction loss
        recon = model.decode(s_optimize, c)
        recon_error = torch.sum((recon - X).pow(2))
        # total loss
        recon_error.backward()

        optimizer.step()

    return recon, recon_error


def get_recon_onlyc(X, y, model):
    _, _, c_mu, c_logvar = model.encode(X)
    group_mu, group_logvar, _, _ = utils.accumulate_group_evidence(
        c_mu.data, c_logvar.data, y
    )
    c_z = utils.group_wise_reparameterize(
        mu=group_mu, logvar=group_logvar, labels_batch=y, cuda=True, training=False
    )
    recon = model.decode(torch.zeros(c_z.size()).to(device=device), c_z)
    recon_error = torch.sum((recon - X).pow(2))

    return recon, recon_error


def train(model, ds, root_dir):
    # should not shuffle here
    train_loader = DataLoader(ds, args.batch_size, shuffle=False, drop_last=False)

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
            log.write('Epoch\tIteration\tLoss\tRecon_error\tStyle_KL\tContent_KL\n')
    # initialize summary writer
    writer = SummaryWriter()

    # start training
    for epoch in range(0, args.nepochs):
        print('Epoch {}'.format(epoch))

        # the total loss at each epoch after running all iterations of batches
        iteration = 0

        for batch_index, (X, y) in enumerate(train_loader):
            # set zero grad for the optimizer
            optimizer.zero_grad()

            # move data to cuda
            X = X.to(device=device)
            y = y.to(device=device)

            # style is individual, content is group
            style_mu, style_logvar, content_mu, content_logvar = model.encode(X)
            # put all content stuff into group in the grouping/evidence-accumulation stage
            group_mu, group_logvar, var_dict, mu_dict = utils.accumulate_group_evidence(
                content_mu.data, content_logvar.data, y
            )

            '''
            c1_c2_kl = 0.0
            start = batch_index * int(args.batch_size / args.T)
            end = (batch_index+1) * int(args.batch_size / args.T)
            for k in range(start, end):
                var1, mu1 = var_dict[2*k], mu_dict[2*k]
                var2, mu2 = var_dict[2*k+1], mu_dict[2*k+1]
                c1_c2_kl += 1/2*(torch.sum(torch.div(var1, var2)) + \
                                 torch.sum(torch.div(torch.square(mu2-mu1), var2)) + \
                    torch.sum(mu2-mu1))
            c1_c2_kl /= args.batch_size * np.prod(ds.data_dim)
            '''

            # KL-divergence errors
            style_kl = -0.5 * torch.sum(1 + style_logvar - style_mu.pow(2) - style_logvar.exp())
            content_kl = -0.5 * torch.sum(1 + group_logvar - group_mu.pow(2) - group_logvar.exp())
            style_kl /= args.batch_size * np.prod(ds.data_dim)
            content_kl /= args.batch_size * np.prod(ds.data_dim)

            """
            sampling from group mu and logvar for each image in mini-batch differently makes
            the decoder consider content latent embeddings as random noise and ignore them 
            """
            # reconstruction error
            style_z = utils.reparameterize(mu=style_mu, logvar=style_logvar, training=True)
            content_z = utils.group_wise_reparameterize(
                mu=group_mu, logvar=group_logvar, labels_batch=y, cuda=True, training=True
            )
            reconstruction = model.decode(style_z, content_z)
            reconstruction_error = utils.mse_loss(reconstruction, X)

            loss = reconstruction_error + float(args.beta) * (style_kl + content_kl)
            loss.backward()
            optimizer.step()

            # print losses
            if batch_index % 50 == 0 or batch_index == args.batch_size - 1:
                print('[%d/%d][%d/%d]\tLoss: %.4E | Recon error: %.4E | Style KL: %.4E | Content KL: %.4E'
                      % (epoch, args.nepochs, batch_index, len(train_loader),
                         loss.item(), reconstruction_error.item(), style_kl.item(), content_kl.item()))

            # write to log
            with open(path.join(root_dir, args.log_file), 'a') as log:
                log.write('{0}\t{1}\t{2}\t{3}\t{4}\n{5}\n'.format(
                    epoch,
                    batch_index,
                    loss.detach().item(),
                    reconstruction_error.detach().item(),
                    style_kl.detach().item(),
                    content_kl.detach().item()
                ))

            # write to tensorboard
            itr = epoch * (int(len(ds) / args.batch_size) + 1) + batch_index
            writer.add_scalar('Loss', loss.detach().item(), itr)
            writer.add_scalar('Recon error', reconstruction_error.detach().item(), itr)
            writer.add_scalar('Style KL', style_kl.detach().item(), itr)
            writer.add_scalar('Content KL', style_kl.detach().item(), itr)

        # save the model at every epoch
        torch.save(model.state_dict(), path.join(root_dir, 'model'))


def test(model, ds, dir, iterations):
    print("Running tests...")
    if args.test_method == 'graph-cut':
        test_dir = 'errors_graph_cut'
    else:
        test_dir = 'errors_mean_distance'
    test_dir_path = path.join(dir, test_dir)
    if not path.exists(test_dir_path):
        os.makedirs(test_dir_path)

    # start testing
    eta_hats = []  # save predicted change points
    etas = []
    # iterate over test samples X_1, X_2, etc...
    all_i = range(ds.n) if not args.dataset == 'clevr' else [args.T*6*(i-1)+j for i in range(1, 7) for j in range(5)]
    for i in all_i:
        etas.append(ds.cps[i])
        # load test sample X_i
        X = ds.get_time_series_sample(i).to(device)

        scores = {}  # save errors for all candidate etas
        min_eta = 2
        max_eta = ds.T - 2
        max_score = -float('inf')
        eta_hat, min_G1, min_G2 = -1, None, None
        for eta in range(min_eta, max_eta + 1):
            # get reconstructions and errors of 2 groups
            if args.test_method == 'graph-cut':
                score = graph_cut(model, ds, X, eta, only_c=False)
            else:
                get_recon = get_recon_plain if iterations == 0 else get_recon_minimize
                G1, G1_error = get_recon(X[0:eta], torch.zeros(eta, 1), model)
                G2, G2_error = get_recon(X[eta:ds.T], torch.zeros(ds.T - eta, 1), model)
                score = - (G1_error.detach().item() + G2_error.detach().item())
            scores[eta] = score.detach().item()
            if score > max_score:
                max_score = score
                eta_hat = eta
                # min_G1 = G1
                # min_G2 = G2
        eta_hats.append(eta_hat)

        # # decode(s=0, c)
        # G1_onlyc, _ = get_recon_onlyc(X[0:eta_hat], torch.zeros(eta_hat, 1), model)
        # G2_onlyc, _ = get_recon_onlyc(X[eta_hat:ds.T], torch.zeros(ds.T-eta_hat, 1), model)
        #
        # # color change points
        # # blue strip for original images
        # X[etas[i] - 1][0, :, -5:-1] = X[etas[i] - 1][1, :, -5:-1] = 0
        # X[etas[i] - 1][2, :, -5:-1] = 255
        # # red strip for reconstructions
        # min_G1[-1][0, :, -5:-1] = 255
        # min_G1[-1][1, :, -5:-1] = min_G1[-1][2, :, -5:-1] = 0
        # # red strip for fixed c
        # G1_onlyc[-1][0, :, -5:-1] = 255
        # G1_onlyc[-1][1, :, -5:-1] = G1_onlyc[-1][2, :, -5:-1] = 0
        # grid = make_grid(torch.cat([X,   min_G1, min_G2,   G1_onlyc, G2_onlyc]), nrow=ds.T)
        # save_image(grid, path.join(test_dir_path, 'X_{}.png'.format(i)))

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
    with open(path.join(test_dir_path, 'error.txt'), 'w') as f:
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
    dir2 = path.join(dir1, args.model + '_' + str(args.cs_dim))
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
        ds = datasets.mnist_loader(args.N, args.T, train=True, seed=7, transform=trans)
        ds_test = datasets.mnist_loader(300, args.T, train=False, seed=7, transform=trans)
    elif args.dataset == 'cifar10':
        ds = datasets.cifar10_loader(args.N, args.T, train=True, seed=7, transform=trans)
        ds_test = datasets.cifar10_loader(300, args.T, train=False, seed=7, transform=trans)
    elif args.dataset == 'celeba':
        ds = datasets.celeba_gender_change(args.N, args.T, train=True, seed=7, transform=trans)
        ds_test = datasets.celeba_gender_change(300, args.T, train=False, seed=7, transform=trans)
    elif args.dataset == 'clevr':
        ds = datasets.clevr_change('n=2100T=50', args.T, transform=trans)
        ds_test = datasets.clevr_change('n=2100T=50', args.T, transform=trans)
    else:
        raise Exception("invalid dataset name")

    print('Creating models...')
    if args.model == 'linearmlvae':
        model = networks.linearMLVAE(ds.data_dim, 500, args.cs_dim).to(device)
    elif args.model == 'convmlvae':
        model = networks.convMLVAE(args.cs_dim).to(device)
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
        print("iterations = ", args.iterations)
        for existing in existing_dirs:
            root_dir = path.join(dir2, str(existing))
            model.load_state_dict(torch.load(path.join(root_dir, 'model')))
            test(model, ds_test, root_dir, int(args.iterations))