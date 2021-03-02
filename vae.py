import os
import os.path as path
import numpy as np
import argparse
import json

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
from torchvision.utils import make_grid

import data_loaders
import networks
import utils


################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--N', type=int, default=1000)
parser.add_argument('--T', type=int, default=50)
parser.add_argument('--model', type=str)
parser.add_argument('--cs_dim', type=int)

parser.add_argument('--train', type=int, default=1)
parser.add_argument('--test', type=int, default=1)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--end_epoch', type=int)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--initial_lr', type=float, default=0.001)
parser.add_argument('--beta1', default=1)

parser.add_argument('--log_file', default='log.txt')
parser.add_argument('--continue_saved', default=False)



def main():
    # create necessary directories
    dir0 = 'experiments'
    dir1 = path.join(dir0, args.dataset)
    dir2 = path.join(dir1, args.model + '_' + str(args.cs_dim))
    for d in [dir0, dir1, dir2]:
        if not path.exists(d):
            os.makedirs(d)

    # use cpu or gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.train == 1:

        # create new directory for this run
        numbered_dirs = [int(f) for f in os.listdir(dir2) if f.isdigit()]
        new_dir = '1' if not numbered_dirs else str(max(numbered_dirs) + 1)

        # root dir is the directory of this particular run of experiment
        # all data produced by training and testing will be saved in this root dir
        root_dir = path.join(dir2, new_dir)
        if not path.exists(root_dir):
            os.makedirs(root_dir)

        # save args
        with open(path.join(root_dir, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        #####################################################################################

        # load data set and create data loader instance
        print('Loading training data...')
        if args.dataset == 'celeba':
            ds = data_loaders.celeba_gender_change(args.N, args.T, train=True, seed=7)
        elif args.dataset == 'clevr_change':
            ds = data_loaders.clevr_change(args.dataset, args.T, utils.transform_config2)
        elif args.dataset == 'cifar10':
            ds = data_loaders.cifar10_loader(args.N, args.T, train=True, seed=7, transform=utils.trans_config)
        elif args.dataset == 'mnist':
            ds = data_loaders.mnist_loader(args.T, args.T, train=True, seed=7)
        else:
            raise Exception("invalid dataset name")
        # should not shuffle here
        train_loader = DataLoader(ds, args.batch_size, shuffle=False, drop_last=False)

        # model definition
        if args.model == 'linearvae':
            model = networks.linearVAE(ds.data_dim, 500, args.cs_dim)
        elif args.model == 'dfcvae':
            model = networks.DFCVAE()
        elif args.model == 'convvae':
            model = networks.convVAE()
        else:
            raise Exception("invalid model name")
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
                log.write('Epoch\tIteration\tReconstruction_loss\tKL\n')
        # initialize summary writer
        writer = SummaryWriter()

        # start training
        for epoch in range(args.start_epoch, args.end_epoch):
            print('Epoch {}'.format(epoch))

            # the total loss at each epoch after running all iterations of batches
            total_loss = 0
            iteration = 0

            for batch_index, (X, y) in enumerate(train_loader):
                # set zero grad for the optimizer
                optimizer.zero_grad()

                # move data to cuda
                X = X.to(device=device)
                y = y.to(device=device)

                mu, logvar = model.encode(X)

                # style and content KL-divergence
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kl /= args.batch_size * np.prod(ds.data_dim)
                """
                sampling from group mu and logvar for each image in mini-batch differently makes
                the decoder consider content latent embeddings as random noise and ignore them 
                """
                # reconstruction error
                z = utils.reparameterize(training=True, mu=mu, logvar=logvar)
                reconstruction = model.decode(z)
                reconstruction_error = utils.mse_loss(reconstruction, X)
                # feature loss in dfcvae
                feature_loss = 0.0
                if args.model == 'dfcvae':
                    reconstruction_features = model.extract_features(reconstruction)
                    input_features = model.extract_features(X)
                    for (r, i) in zip(reconstruction_features, input_features):
                        feature_loss += utils.mse_loss(r, i)
                # total_loss and backpropagate
                loss = (reconstruction_error + feature_loss) + int(args.beta1) * kl
                loss.backward()
                # update optimizer
                optimizer.step()
                # compute total loss for this epoch
                total_loss += loss.detach()

                # print losses
                if (iteration + 1) % 50 == 0:
                    print("Total loss:", total_loss)
                iteration += 1

                # write to log
                with open(path.join(root_dir, args.log_file), 'a') as log:
                    log.write('{0}\t{1}\t{2}\t{3}\n'.format(
                        epoch,
                        iteration,
                        reconstruction_error.detach().item(),
                        kl.detach().item()
                    ))

                # write to tensorboard
                itr = epoch * (int(len(ds) / args.batch_size) + 1) + iteration
                writer.add_scalar('Reconstruction loss', reconstruction_error.detach().item(), itr)
                writer.add_scalar('KL', kl.detach().item(), itr)
                if args.model == 'dfcvae':
                    writer.add_scalar('Feature loss', feature_loss.detach().item(), itr)

            # save checkpoints after at every epoch
            torch.save(model.state_dict(), path.join(root_dir, 'model'))

    # testing
    if args.test == 1:
        for rdir in os.listdir(dir2):
            root_dir = path.join(dir2, rdir)
            recon_dir = path.join(root_dir, 'reconstructions')
            if path.exists(recon_dir):
                continue
            else:
                os.makedirs(recon_dir)

            print('Loading test data...')
            if args.dataset == 'celeba':
                ds = data_loaders.celeba_gender_change(30, args.T, train=False, seed=7)
            elif args.dataset == 'clevr_change':
                ds = data_loaders.clevr_change(args.dataset, args.T, utils.trans_config2)
            elif args.dataset == 'cifar10':
                ds = data_loaders.cifar10_loader(30, args.T, train=False, seed=7)
            elif args.dataset == 'mnist':
                ds = data_loaders.mnist_loader(30, args.T, train=False, seed=7)
            else:
                raise Exception("invalid dataset name")

            # model definition
            if args.model == 'linearvae':
                model = networks.linearVAE(ds.data_dim, 500, args.cs_dim)
            elif args.model == 'dfcvae':
                model = networks.DFCVAE()
            elif args.model == 'convvae':
                model = networks.convVAE()
            else:
                raise Exception("invalid model name")

            model.load_state_dict(torch.load(path.join(root_dir, 'model')))
            model = model.to(device=device)

            for i in range(ds.n):
                print('Running time series test sample X_{}'.format(i))
                X = ds.get_time_series_sample(i)
                X = X.to(device=device)
                mu, logvar = model.encode(X)
                z = utils.reparameterize(training=False, mu=mu, logvar=logvar)
                reconstruction = model.decode(z)

                grid = make_grid(torch.cat([X,
                                            reconstruction
                                            ]), nrow=ds.T)
                save_image(grid, path.join(recon_dir, 'X_{}.png'.format(i)))

if __name__ == '__main__':
    args = parser.parse_args()
    main()