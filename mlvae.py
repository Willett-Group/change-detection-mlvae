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

parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--end_epoch', type=int)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--initial_lr', type=float, default=0.001)
parser.add_argument('--beta', type=float, default=1)
parser.add_argument('--val_period', type=int, default=10)

parser.add_argument('--log_file', default='log.txt')
parser.add_argument('--continue_saved', default=False)

parser.add_argument('--iterations', default=20)


#################################################################################

def get_recon(X, y, model):
    # style is individual, content is group
    s_mu, s_logvar, c_mu, c_logvar = model.encode(X)
    # put all content stuff into group in the grouping/evidence-accumulation stage
    group_mu, group_logvar = utils.accumulate_group_evidence(
        c_mu.data, c_logvar.data, y
    )

    """
    sampling from group mu and logvar for each image in mini-batch differently makes
    the decoder consider content latent embeddings as random noise and ignore them
    """
    s_z = utils.reparameterize(mu=s_mu, logvar=s_logvar)
    c_z = utils.group_wise_reparameterize(
        mu=group_mu, logvar=group_logvar, labels_batch=y, cuda=True
    )
    recon = model.decode(s_z, c_z)
    recon_error = torch.sum((recon - X).pow(2))

    return recon, recon_error


def get_recon_minimize(X, y, model):
    s_mu, s_logvar, c_mu, c_logvar = model.encode(X)
    group_mu, group_logvar = utils.accumulate_group_evidence(
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

    for itr in range(args.iterations):
        optimizer.zero_grad()

        # reconstruction loss
        recon = model.decode(s_optimize, c)
        recon_error = torch.sum((recon - X).pow(2))
        # feature loss if using dfc(ml)vae
        '''
        recon_features = model.extract_features(recon)
        X_features = model.extract_features(X)
        feature_loss = 0.0
        for (r, i) in zip(recon_features, X_features):
            feature_loss += utils.mse_loss(r, i)
        '''
        # total loss
        recon_error.backward()

        optimizer.step()

    return recon, recon_error


def get_reconstructions_fixed_style(X, eta, T, i, j, model):
    g1 = X[0:eta]  # group 1 (before change point)
    g2 = X[eta:T]  # group 2 (after change point)
    s_mu_g1, _, c_mu_g1, c_logvar_g1 = model.encode(g1)
    s_mu_g2, _, c_mu_g2, c_logvar_g2 = model.encode(g2)

    s_mu_g1_grouped = torch.empty(s_mu_g1.size())
    for p in range(s_mu_g1_grouped.size(0)):
        s_mu_g1_grouped[p] = s_mu_g1[i]
    s_mu_g2_grouped = torch.empty(s_mu_g2.size())
    for q in range(s_mu_g2_grouped.size(0)):
        s_mu_g2_grouped[q] = s_mu_g2[j]

    s_mu_g1_grouped = s_mu_g1_grouped.to(device=device)
    s_mu_g2_grouped = s_mu_g2_grouped.to(device=device)

    g1_reconstructions, g1_reconstruction_error = extract_reconstructions(g1, s_mu_g1_grouped, c_mu_g1, c_logvar_g1,
                                                                          args.iterations)
    g2_reconstructions, g2_reconstruction_error = extract_reconstructions(g2, s_mu_g2_grouped, c_mu_g2, c_logvar_g2,
                                                                          args.iterations)
    total_error = g1_reconstruction_error.item() + g2_reconstruction_error.item()

    return g1_reconstructions, g2_reconstructions, total_error


def main():
    print('Initializing training and testing datasets...')
    if args.dataset == 'mnist':
        ds = dataloaders.mnist_loader(args.T, args.T, train=True, seed=7, transform=utils.trans_config)
        ds_test = dataloaders.mnist_loader(100, args.T, train=False, seed=7, transform=utils.trans_config)
    elif args.dataset == 'cifar10':
        ds = dataloaders.cifar10_loader(args.N, args.T, train=True, seed=7, transform=utils.trans_config)
        ds_test = dataloaders.cifar10_loader(100, args.T, train=False, seed=7, transform=utils.trans_config)
    elif args.dataset == 'celeba':
        ds = dataloaders.celeba_gender_change(args.N, args.T, train=True, seed=7, transform=utils.trans_config1)
        ds_test = dataloaders.celeba_gender_change(100, args.T, train=False, seed=7, transform=utils.trans_config1)
    elif args.dataset == 'clevr':
        ds = dataloaders.clevr_change('n=2100T=50', args.T, utils.trans_config1_special)
        ds_test = dataloaders.clevr_change('n=2100T=50', args.T, utils.trans_config1_special)
    else:
        raise Exception("invalid dataset name")

    print('Initializing models...')
    if args.model == 'linearmlvae':
        model = networks.linearMLVAE(ds.data_dim, 500, args.cs_dim)
        model_test = networks.linearMLVAE(ds.data_dim, 500, args.cs_dim)
    elif args.model == 'dfcmlvae':
        model = networks.dfcMLVAE()
        model_test = networks.dfcMLVAE()
    elif args.model == 'convmlvae':
        model = networks.convMLVAE()
        model_test = networks.convMLVAE()
    else:
        raise Exception("invalid model name")

    # create new directory for this training run
    numbered_dirs = [int(f) for f in os.listdir(dir2) if f.isdigit()]
    new_dir = '1' if not numbered_dirs else str(max(numbered_dirs) + 1)

    # root dir is the directory of this particular run of experiment
    # all data produced by training and testing will be saved in this root dir
    for i in (1,2,3,5,6):
        root_dir = path.join(dir2, str(i))
        if not path.exists(root_dir):
            os.makedirs(root_dir)

        # save args
        #with open(path.join(root_dir, 'args.txt'), 'w') as f:
        #   json.dump(args.__dict__, f, indent=2)

        #####################################################################################

        # should not shuffle here
        train_loader = DataLoader(ds, args.batch_size, shuffle=False, drop_last=False)

        # move model to gpu
        model.to(device=device)

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
                log.write('Epoch\tIteration\tReconstruction_loss\tStyle_KL\tContent_KL\n')
        # initialize summary writer
        writer = SummaryWriter()

        # save information for testing phase
        curr_best_error = float('inf')
        epoch_error = {}

        # start training
        for epoch in range(args.start_epoch, args.end_epoch):
            print('Epoch {}'.format(epoch))

            '''
            # the total loss at each epoch after running all iterations of batches
            total_loss = 0
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
                group_mu, group_logvar = utils.accumulate_group_evidence(
                    content_mu.data, content_logvar.data, y
                )
    
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
                style_z = utils.reparameterize(mu=style_mu, logvar=style_logvar)
                content_z = utils.group_wise_reparameterize(
                    mu=group_mu, logvar=group_logvar, labels_batch=y, cuda=True
                )
                reconstruction = model.decode(style_z, content_z)
                reconstruction_error = utils.mse_loss(reconstruction, X)
                # feature loss
                feature_loss = 0.0
                if args.model == 'dfcmlvae':
                    reconstruction_features = model.extract_features(reconstruction)
                    input_features = model.extract_features(X)
                    for (r, i) in zip(reconstruction_features, input_features):
                        feature_loss += utils.mse_loss(r, i)
    
                loss = reconstruction_error + feature_loss + \
                       float(args.beta) * (style_kl + content_kl)
                loss.backward()
                # update optimizer
                optimizer.step()
                # compute total loss for this epoch
                total_loss += loss.detach().item()
    
                # print losses
                if (iteration + 1) % 50 == 0:
                    print("Total loss:", total_loss)
                iteration += 1
    
                # write to log
                with open(path.join(root_dir, args.log_file), 'a') as log:
                    log.write('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(
                        epoch,
                        iteration,
                        reconstruction_error.detach().item(),
                        style_kl.detach().item(),
                        content_kl.detach().item()
                    ))
    
                # write to tensorboard
                itr = epoch * (int(len(ds) / args.batch_size) + 1) + iteration
                writer.add_scalar('Reconstruction loss', reconstruction_error.detach().item(), itr)
                writer.add_scalar('Style KL', style_kl.detach().item(), itr)
                writer.add_scalar('Content KL', style_kl.detach().item(), itr)
                if args.model == 'dfcvae':
                    writer.add_scalar('Feature loss', feature_loss.detach().item(), itr)
    
            # save the model at every epoch
            torch.save(model.state_dict(), path.join(root_dir, 'model_cur'))
            '''

            if (args.val_period < args.end_epoch and epoch % args.val_period == 0) \
                    or epoch == args.end_epoch-1:
                # run validations
                print('\nRunning tests at epoch{}'.format(epoch))
                recon_dir = path.join(root_dir, 'images_epoch{}'.format(epoch))
                if not path.exists(recon_dir):
                    os.makedirs(recon_dir)

                model_test.load_state_dict(torch.load(path.join(root_dir, 'model_cur')))
                model_test = model_test.to(device=device)

                # start testing
                eta_hats = []  # save predicted change points
                etas = []

                # iterate over test samples X_1, X_2, etc...
                if args.dataset == 'clevr':
                    all_i = [args.T*6*(i-1)+j for i in range(1,7) for j in range(5)]
                else:
                    all_i = range(ds_test.n)
                for i in all_i:
                    etas.append(ds_test.cps[i])
                    # load the test sample X_i
                    X = ds_test.get_time_series_sample(i)
                    X = X.to(device=device)

                    errors = {}  # save errors for all candidate etas
                    min_eta = 2
                    max_eta = ds_test.T - 2
                    min_total_error = float('inf')
                    eta_hat, min_recon1, min_recon2 = -1, None, None
                    for eta in range(min_eta, max_eta + 1):
                        recon1, recon_error1 = get_recon_minimize(X[0:eta], torch.zeros(eta, 1), model_test)
                        recon2, recon_error2 = get_recon_minimize(X[eta:ds_test.T], torch.zeros(ds_test.T - eta, 1), model_test)
                        total_error = recon_error1.detach().item() + recon_error2.detach().item()
                        errors[eta] = total_error
                        if total_error < min_total_error:
                            min_total_error = total_error
                            eta_hat = eta
                            min_recon1 = recon1
                            min_recon2 = recon2
                    eta_hats.append(eta_hat)

                    # reconstruction of g1 and g2 without minimizing P(x), i.e. iteration = 0
                    # recon1_plain, _ = get_recon(X[0:eta_hat], torch.zeros(eta_hat, 1), model_test)
                    # recon2_plain, _ = get_recon(X[eta_hat:ds_test.T], torch.zeros(ds_test.T - eta_hat, 1), model_test)

                    recon1_diff = torch.empty(size=recon1.size())
                    recon2_diff = torch.empty(size=recon2.size())
                    recon1_diff = recon1_diff.to(device=device)
                    recon2_diff = recon2_diff.to(device=device)
                    recon1_diff[0] = recon1[0]
                    recon2_diff[0] = recon2[0]
                    for k in range(1, recon1.size(0)):
                        recon1_diff[k] = recon1[k]-recon1[k-1]
                    for k in range(1, recon2.size(0)):
                        recon2_diff[k] = recon2[k]-recon2[k-1]

                    grid = make_grid(torch.cat([X,
                                                min_recon1, min_recon2,
                                                recon1_diff, recon2_diff
                                                ]), nrow=ds_test.T)
                    save_image(grid, path.join(recon_dir, 'X_{}.png'.format(i)))

                    # save square errors
                    plt.scatter(list(errors.keys()), list(errors.values()))
                    plt.axvline(x=ds_test.cps[i])
                    plt.axvline(x=eta_hat, color='r')
                    plt.xlabel('etas (red: eta_hat, blue: true eta)')
                    plt.ylabel('squared errors')
                    plt.savefig(path.join(recon_dir, 'X_{}_errors.png'.format(i)))
                    plt.close()

                # compute mean of |eta-eta_hat| among all test samples
                diff = np.abs(np.asarray(etas) - np.asarray(eta_hats))
                error = np.mean(diff)
                # keep track of the errors associated with epochs
                epoch_error[epoch] = error
                with open(path.join(root_dir, 'epoch_errors.txt'), 'w') as f:
                    json.dump(epoch_error, f, indent=2)
                if error < curr_best_error:
                    curr_best_error = error
                    # save current best model
                    torch.save(model_test.state_dict(), path.join(root_dir, 'model_best'))
                    # save eta_hats of all test samples at this current best model
                    with open(root_dir + '/cps.txt', 'w') as cps_r:
                        for tmp in eta_hats:
                            cps_r.write('{} '.format(tmp))
                        cps_r.write('\n')
                        for tmp in etas:
                            cps_r.write('{} '.format(tmp))


if __name__ == '__main__':
    args = parser.parse_args()
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

    # start training and/or testing
    main()
