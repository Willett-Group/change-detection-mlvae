import os
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from multiprocessing import Pool
from functools import partial

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.utils import make_grid

import data_loaders
import utils
import networks


def extract_reconstructions(X, style_mu, class_mu, class_logvar, n_iterations):
    grouped_mu, _ = utils.accumulate_group_evidence(
        class_mu.data, class_logvar.data, torch.zeros(style_mu.size(0), 1)
    )
    decoder_style_input = style_mu.clone().detach().requires_grad_(True)
    decoder_content_input = grouped_mu[0].clone().detach().requires_grad_(True)

    content = decoder_content_input.expand(style_mu.size(0), decoder_content_input.size(0))

    optimizer = torch.optim.Adam(
        [decoder_style_input, decoder_content_input]
    )

    for iterations in range(n_iterations):
        optimizer.zero_grad()

        # reconstruction loss
        reconstruction = model.decode(decoder_style_input, content)
        reconstruction_error = torch.sum((reconstruction - X).pow(2))

        # feature loss
        '''
        reconstruction_features = model.extract_features(reconstruction)
        input_features = model.extract_features(X)
        feature_loss = 0.0
        for (r, i) in zip(reconstruction_features, input_features):
            feature_loss += utils.mse_loss(r, i)
        '''

        # total loss
        loss = reconstruction_error
        loss.backward()

        optimizer.step()

    return reconstruction, reconstruction_error


def get_reconstructions(model, X, eta, T):
    g1 = X[0:eta]  # group 1 (before change point)
    g2 = X[eta:T]  # group 2 (after change point)
    style_mu_g1, _, class_mu_g1, class_logvar_g1 = model.encode(g1)
    style_mu_g2, _, class_mu_g2, class_logvar_g2 = model.encode(g2)

    g1_reconstructions, g1_reconstruction_error = extract_reconstructions(g1, style_mu_g1, class_mu_g1, class_logvar_g1,
                                                                          config['iterations'])
    g2_reconstructions, g2_reconstruction_error = extract_reconstructions(g2, style_mu_g2, class_mu_g2, class_logvar_g2,
                                                                          config['iterations'])
    total_error = g1_reconstruction_error.item() + g2_reconstruction_error.item()

    return g1_reconstructions, g2_reconstructions, total_error


# settings
torch.set_printoptions(precision=8)
print = partial(print, flush=True)

gpu_ids = []
for ii in range(6):
    try:
        torch.cuda.get_device_properties(ii)
        print(str(ii), flush=True)
        if not gpu_ids:
            gpu_ids = [ii]
        else:
            gpu_ids.append(ii)
    except AssertionError:
        print('Not ' + str(ii) + "!", flush=True)

print(os.getenv('CUDA_VISIBLE_DEVICES'), flush=True)
gpu_ids = [int(x) for x in gpu_ids]
use_dataparallel = len(gpu_ids) > 1
print("GPU IDs: " + str([int(x) for x in gpu_ids]), flush=True)

# make necessary directories
config = {
    'experiment_setting': 'cifar10',
    'model': 'linearvae',  # 'dfcvae', 'linearvae', 'convvae', 'resnetvae'

    'N': 30,
    'T': 50,
    'iterations': 20
}
dir1 = 'experiments/' + config['experiment_setting']
for dir in os.listdir(dir1):
    print('Running directory', dir)
    if dir.isdigit():
        root_dir = os.path.join(dir1, dir)
        recon = root_dir + '/reconstructions/'
        sqerrors = root_dir + '/sqerrors/'

        for dir in [recon, sqerrors]:
            if not os.path.exists(dir):
                os.makedirs(dir)

        # use gpu or cpu
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # model definition
        if config['model'] == 'dfcvae':
            model = networks.DFCVAE()
        elif config['model'] == 'convvae':
            model = networks.convVAE()
        elif config['model'] == 'linearvae':
            model = networks.linearVAE2(500, 500)
        # load saved parameters of model
        model.load_state_dict(torch.load(os.path.join(root_dir, 'model'), map_location=lambda storage, loc: storage))
        model = model.to(device=device)

        # load dataset
        print('Loading test data...')
        ds = data_loaders.cifar10_loader(config['N'], config['T'], 7, False)
        eta_hats = []  # save predicted change points

        # iterate over test samples X_1, X_2, etc...
        for i in range(ds.n):
            print('Running time series sample X_' + str(i))

            # load the test sample X_i
            X = ds.get_time_series_sample(i)
            X = X.to(device=device)

            errors = {}  # errors for all candidate etas
            min_eta = 2
            max_eta = ds.T - 2

            for eta in range(min_eta, max_eta + 1):
                g1_reconstructions, g2_reconstructions, total_error = get_reconstructions(model, X, eta, ds.T)
                errors[eta] = total_error

            # finished iterating through candidate etas, now can get eta_hat = argmin eta
            eta_hat = min(errors, key=errors.get)
            eta_hats.append(eta_hat)

            # save originals, reconstructions with smallest error, reconstructions with true eta
            g1_reconstructions_hat, g2_reconstructions_hat, _ = get_reconstructions(model, X, eta_hat, ds.T)
            g1_reconstructions_true, g2_reconstructions_true, _ = get_reconstructions(model, X, ds.cps[i], ds.T)
            # print(torch.mean(g2_reconstructions, dim = 0).size())

            g1_subtracted = g1_reconstructions_true - torch.mean(g1_reconstructions_true, dim=0)
            g2_subtracted = g2_reconstructions_true - torch.mean(g2_reconstructions_true, dim=0)

            g1_mul_ratio = torch.div(256, torch.amax(g1_subtracted, dim=(0)))
            g2_mul_ratio = torch.div(256, torch.amax(g2_subtracted, dim=(0)))
            print(g1_mul_ratio.size())

            g1_subtracted_normalized = torch.mul(g1_subtracted, g1_mul_ratio)
            g2_subtracted_normalized = torch.mul(g2_subtracted, g2_mul_ratio)

            grid = make_grid(torch.cat([X,
                                        g1_reconstructions_true, g2_reconstructions_true,
                                        g1_subtracted, g2_subtracted,
                                        g1_subtracted_normalized, g2_subtracted_normalized
                                        ]), nrow=ds.T)
            save_image(grid, recon + 'X_{}.png'.format(i))

            # save square errors
            plt.scatter(list(errors.keys()), list(errors.values()))
            plt.axvline(x=ds.cps[i])
            plt.axvline(x=eta_hat, color='r')
            plt.title(config['experiment_setting'])
            plt.xlabel('etas (red: eta_hat, blue: true eta)')
            plt.ylabel('squared errors')
            plt.savefig(sqerrors + 'X_{}.jpg'.format(i))
            plt.close()

        with open(root_dir + '/cps.txt', 'w') as cps_r:
            for tmp in eta_hats:
                cps_r.write('{} '.format(tmp))
            cps_r.write('\n')
            for tmp in ds.cps:
                cps_r.write('{} '.format(tmp))