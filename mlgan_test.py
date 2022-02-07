import os
import os.path as path
import numpy as np
import argparse
import json
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torchvision
from torchvision.utils import save_image
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from torchvision import datasets

import dataloaders
import networks
import utils

################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--N', type=int, default=4000)
parser.add_argument('--T', type=int, default=50)
parser.add_argument('--model', type=str)
parser.add_argument('--cs_dim', type=int)

parser.add_argument('--nepochs', type=int)
parser.add_argument('--batch_size', type=int, default=250)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--b1', type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument('--b2', type=float, default=0.999, help="decay of second order momentum of gradient")
parser.add_argument('--val_period', type=int, default=10)

parser.add_argument('--iterations', default=20)

parser.add_argument('--channels', default=3)
parser.add_argument('--img_size_x', default=64)
parser.add_argument('--img_size_y', default=64)

args = parser.parse_args()

manualSeed = 999
print("Random seed： ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

img_shape = (args.channels, args.img_size_x, args.img_size_y)
################################################################################

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = args.img_size_x // 4
        self.l1 = nn.Sequential(nn.Linear(2*args.cs_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, args.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        print(z.size())
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(args.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = args.img_size_x // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


class DCGenerator(nn.Module):
    def __init__(self):
        super(DCGenerator, self).__init__()
        self.ngpu = 1
        self.main = nn.Sequential(
            nn.ConvTranspose2d(2*args.cs_dim, 64*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64*8, 64*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64*4, 64*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64*2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, args.channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z)

class DCDiscriminator(nn.Module):
    def __init__(self):
        super(DCDiscriminator, self).__init__()
        self.ngpu = 1
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(args.channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, image):
        return self.main(image)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_recon_minimize(X, G):
    # 2 variables to optimize wrt
    l = X.size(0)
    s_optimize = torch.zeros((l, args.cs_dim, 1, 1), device=device, requires_grad=True)
    c_optimize = torch.zeros((args.cs_dim, 1, 1), device=device, requires_grad=True)
    # c is c_optimized stacked vertically batch_size times
    c = c_optimize.expand(l, -1, -1, -1)

    z = torch.cat((c, s_optimize), dim=1)

    optimizer = torch.optim.Adam(
        [s_optimize, c_optimize]
    )

    for itr in range(args.iterations):
        optimizer.zero_grad()
        # reconstruction loss
        recon = G(z)
        recon_error = torch.sum((recon - X).pow(2))
        # feature loss if using dfc(ml)vae
        recon_error.backward()

        optimizer.step()

    return recon, recon_error


dataset_dir = "../datasets"
# create parent directories, like 'experiments/cifar10/linearmlvae_50'
# and 'experiments/cifar10/dfcmlvae_128'
dir0 = 'experiments'
dir1 = path.join(dir0, args.dataset)
dir2 = path.join(dir1, args.model + '_' + str(args.cs_dim))
for d in [dir0, dir1, dir2]:
    if not path.exists(d):
        os.makedirs(d)

# data transformer
trans = transforms.Compose([transforms.Resize([args.img_size_x, args.img_size_y]),
                            transforms.ToTensor()
                            ])
print('Initializing training and testing datasets...')
if args.dataset == 'mnist':
    ds = dataloaders.mnist_loader(args.N, args.T, train=True, seed=7, transform=trans)
    ds_test = dataloaders.mnist_loader(10, args.T, train=False, seed=7, transform=trans)
elif args.dataset == 'cifar10':
    ds = dataloaders.cifar10_loader(args.N, args.T, train=True, seed=7, transform=trans)
    ds_test = dataloaders.cifar10_loader(10, args.T, train=False, seed=7, transform=trans)
elif args.dataset == 'celeba':
    ds = dataloaders.celeba_gender_change(args.N, args.T, train=True, seed=7, transform=trans)
    ds_test = dataloaders.celeba_gender_change(10, args.T, train=False, seed=7, transform=trans)
elif args.dataset == 'clevr':
    ds = dataloaders.clevr_change('n=2100T=50', args.T, transform=trans)
    ds_test = dataloaders.clevr_change('n=2100T=50', args.T, transform=trans)
else:
    raise Exception("invalid dataset name")

# create new directory for this training run
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

# use cpu or gpu
# ds = datasets.CelebA(root=dataset_dir, download=True, split='train', transform=trans)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader = DataLoader(ds, args.batch_size, shuffle=False, drop_last=False)


G = DCGenerator().to(device)
D = DCDiscriminator().to(device)
G_test = DCGenerator().to(device)
G.apply(weights_init)
D.apply(weights_init)
print("Generator arch: ")
print(G)
print("Discriminator arch: ")
print(D)

criterion = torch.nn.BCELoss()
# criterion = torch.nn.BCEWithLogitsLoss()
Gopt = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.b1, args.b2))
Dopt = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.b1, args.b2))

fixed_s = torch.randn(64, args.cs_dim, 1, 1, device=device)
fixed_c = torch.randn(64, args.cs_dim, 1, 1, device=device)
fixed_z = torch.cat((fixed_c, fixed_s), dim=1)

# save information for testing phase
epoch_error = {}


for epoch in range(args.nepochs):
    for batch_index, (X, y) in enumerate(train_loader):
        X = X.to(device)
        batch_size = X.size(0)

        real_labels = torch.ones(batch_size).to(device)
        fake_labels = torch.zeros(batch_size).to(device)

        # TRAIN DISCRIMINATOR
        D.zero_grad()
        output = D(X).view(-1) # probabilities of D(real samples)
        errD_real = criterion(output, real_labels)
        errD_real.backward()
        D_x = output.mean().item()

        c = torch.randn(batch_size, args.cs_dim, 1, 1, device=device)
        s = torch.randn(batch_size, args.cs_dim, 1, 1, device=device)
        # gan analog of grouping stage in mlvae
        values_by_labels = {}
        for i in range(len(y)):
            label = y[i].item()
            if label not in values_by_labels:
                values_by_labels[label] = c[i]
            else:
                c[i] = values_by_labels[label]
        z = torch.cat((c, s), dim=1)
        fake = G(z)
        output = D(fake.detach()).view(-1)
        errD_fake = criterion(output, fake_labels)
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        Dopt.step() # update once errD_real and errD_fake are both backwarded

        # TRAIN GENERATOR
        G.zero_grad()
        output = D(fake).view(-1)
        errG = criterion(output, real_labels)
        errG.backward()
        D_G_z2 = output.mean().item()
        Gopt.step()

        if batch_index % 50 == 0 or batch_index == batch_size-1:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, args.nepochs, batch_index, len(train_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
    # see how fake images look like now
    with torch.no_grad():
        fake = G(fixed_z).detach().cpu()
    grid = make_grid(fake, nrow=8, normalize=True)
    save_image(grid, path.join(root_dir, 'gen_{}.png'.format(epoch)))

    # save models
    torch.save(G.state_dict(), path.join(root_dir, 'G'))
    torch.save(D.state_dict(), path.join(root_dir, 'D'))

    if (args.val_period < args.nepochs and epoch % args.val_period == 0) \
            or epoch == args.nepochs - 1:
        # run validations
        print('\nStart testing at epoch{}'.format(epoch))
        recon_dir = path.join(root_dir, 'images_epoch{}'.format(epoch))
        if not path.exists(recon_dir):
            os.makedirs(recon_dir)

        G_test.load_state_dict(torch.load(path.join(root_dir, 'G')))
        G_test = G_test.to(device)

        # start testing
        eta_hats = []  # save predicted change points
        etas = []

        # iterate over test samples X_1, X_2, etc...
        if args.dataset == 'clevr':
            all_i = [args.T * 6 * (i - 1) + j for i in range(1, 7) for j in range(5)]
        else:
            all_i = range(ds_test.n)
        for i in all_i:
            etas.append(ds_test.cps[i])
            # load the test sample X_i
            X = ds_test.get_time_series_sample(i)
            X = X.to(device)

            errors = {}  # save errors for all candidate etas
            min_eta = 2
            max_eta = ds_test.T - 2
            min_total_error = float('inf')
            eta_hat, min_recon1, min_recon2 = -1, None, None
            for eta in range(min_eta, max_eta + 1):
                recon1, recon_error1 = get_recon_minimize(X[0:eta], G_test)
                recon2, recon_error2 = get_recon_minimize(X[eta:ds_test.T], G_test)
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

            # style transfer
            # style, _, _, _ = model_test.encode(X)
            # recon1_sfixed, _ = get_recon_sfixed(X[0:eta_hat], torch.zeros(eta_hat, 1), model_test, style[0])
            # recon2_sfixed, _ = get_recon_sfixed(X[eta_hat:ds_test.T],
            #                                     torch.zeros(ds_test.T - eta_hat, 1), model_test, style[0])

            X[etas[i] - 1][0, :, -3:-1] = 0
            X[etas[i] - 1][1, :, -3:-1] = 0
            X[etas[i] - 1][2, :, -3:-1] = 255
            min_recon1[-1][0, :, -3:-1] = 255
            min_recon1[-1][1, :, -3:-1] = 0
            min_recon1[-1][2, :, -3:-1] = 0
            # recon1_sfixed[-1][0, :, -3:-1] = 255
            # recon1_sfixed[-1][1, :, -3:-1] = 0
            # recon1_sfixed[-1][2, :, -3:-1] = 0
            grid = make_grid(torch.cat([X,
                                        min_recon1, min_recon2,
                                        # recon1_sfixed, recon2_sfixed
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
        # save eta_hats of all test samples at this current best model
        with open(root_dir + '/cps.txt', 'w') as cps_r:
            for tmp in eta_hats:
                cps_r.write('{} '.format(tmp))
            cps_r.write('\n')
            for tmp in etas:
                cps_r.write('{} '.format(tmp))