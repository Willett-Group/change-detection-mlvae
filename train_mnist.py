import os
import numpy as np
from functools import partial

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import data_loaders
import networks
import utils

################################################################################
# torch settings
torch.set_printoptions(precision=6)
print = partial(print, flush=True)


################################################################################
# configurations
config = {
    'experiment_name': '1',
    'experiment_type': 'nonrepetitive', # or 'nonrepetitive'
    'model': 'dfcvae', # or 'linearvae', 'naiveconvvae', 'resnetvae'
    'n': 1000,
    'T': 50,
    'z_dim': 20,

    'start_epoch': 0,
    'end_epoch': 5,
    'b_size': 256,
    'initial_lr': 0.001,
    'beta': 5000,

    'log_file': 'log.txt',
    'load_saved': False
}


################################################################################
# create necessary directories
if not path.exists('experiments/'):
    os.makedirs('experiments')
root_dir = 'experiments/' + config['experiment_name']
if not path.exists(root_dir):
    os.makedirs(root_dir)
with open(path.join(root_dir, 'config.txt'), 'w') as f:
    for key in config:
        f.write(key + ': ' + str(config[key]) + '\n')

# use cpu or gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load data set and create data loader instance
print('Loading training data...')
if config['experiment_type'] == 'repetitive':
    ds = data_loaders.mnist_loader_repetitive(config['n'], config['T'], cp_way = 3,
                                        train=True, seed=7, model=config['model'])
else:
    ds = data_loaders.mnist_loader(config['n'], config['T'], cp_way = 3,
                                        train=True, seed=7, model=config['model'])
train_loader = DataLoader(ds, batch_size=config['b_size'], shuffle=True, drop_last=False)

# model definition
if config['model'] == 'linearvae':
    model = networks.linearVAE(config['z_dim'], config['z_dim'])
elif config['model'] == 'dfcvae':
    model = networks.DFCVAE()
elif config['model'] == 'convvae':
    model = networks.convVAE()
model.to(device=device)

# load saved models if loading from previously saved model parameters
# initialize log file if not loading from previously saved model parameters
if config['load_saved']:
    model.load_state_dict(torch.load(path.join(root_dir, 'model')))
else:
    with open(path.join(root_dir, config['log_file']), 'w') as log:
        log.write('Epoch\tIteration\tReconstruction_loss\tStyle_KL\tContent_KL\n')
# initialize summary writer
writer = SummaryWriter()

# optimizer definition
optimizer = optim.Adam(
    model.parameters(),
    lr = config['initial_lr']
)

###############################################################################s
# start training
for epoch in range(config['start_epoch'], config['end_epoch']):
    print('\nEpoch #' + str(epoch) + '..............................................')

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

        # KL-divergence losses
        style_kl = -0.5*torch.sum(1+style_logvar-style_mu.pow(2)-style_logvar.exp())
        content_kl = -0.5*torch.sum(1+group_logvar-group_mu.pow(2)-group_logvar.exp())
        style_kl /= config['b_size'] * np.prod(ds.data_dim)
        content_kl /= config['b_size'] * np.prod(ds.data_dim)
        """
        sampling from group mu and logvar for each image in mini-batch differently makes
        the decoder consider content latent embeddings as random noise and ignore them 
        """
        # reconstruction error
        style_z = utils.reparameterize(training=True, mu=style_mu, logvar=style_logvar)
        content_z = utils.group_wise_reparameterize(
            training=True, mu=group_mu, logvar=group_logvar, labels_batch=y, cuda=True
        )
        reconstruction = model.decode(style_z, content_z)
        reconstruction_error = utils.mse_loss(reconstruction, X)

        # feature loss
        
        feature_loss = 0.0
        if config['model'] == 'dfcvae':
            reconstruction_features = model.extract_features(reconstruction)
            input_features = model.extract_features(X)
            for (r, i) in zip(reconstruction_features, input_features):
                feature_loss += utils.mse_loss(r, i)
        

        # total_loss
        # loss = (reconstruction_error) + config['beta']*(style_kl+content_kl)
        loss = (reconstruction_error + feature_loss) + config['beta']*(style_kl+content_kl)
        loss.backward()

        optimizer.step()

        total_loss += loss.detach()

        # print losses
        if (iteration+1) % 50 == 0:
            print('\tIteration #' + str(iteration))
            print('Reconstruction loss: ' + str(reconstruction_error.data.storage().tolist()[0]))
            print('Style KL loss: ' + str(style_kl.data.storage().tolist()[0]))
            print('Content KL loss: ' + str(content_kl.data.storage().tolist()[0]))
            #if config['model'] == 'dfcvae':
            #    print('Feature loss: ' + str(feature_loss.data.storage().tolist()[0]))
        iteration += 1

        # write to log
        with open(path.join(root_dir, config['log_file']), 'a') as log:
            log.write('{0}\t{1}\t{2}\t{3}\t{4}\n'.format(
                epoch,
                iteration,
                reconstruction_error.data.storage().tolist()[0],
                style_kl.data.storage().tolist()[0],
                content_kl.data.storage().tolist()[0]
            ))

        # write to tensorboard
        writer.add_scalar('Reconstruction loss', reconstruction_error.data.storage().tolist()[0],
                        epoch * (int(len(ds) / config['b_size']) + 1) + iteration)
        writer.add_scalar('Style KL', style_kl.data.storage().tolist()[0],
                        epoch * (int(len(ds) / config['b_size']) + 1) + iteration)
        writer.add_scalar('Content KL', content_kl.data.storage().tolist()[0],
                        epoch * (int(len(ds) / config['b_size']) + 1) + iteration)
        if config['model'] == 'dfcvae':
            writer.add_scalar('Feature loss', content_kl.data.storage().tolist()[0],
                        epoch * (int(len(ds) / config['b_size']) + 1) + iteration)

    writer.add_scalar('Total loss', total_loss.item(), epoch)
    print('\nTotal loss: ' + str(total_loss.item()))

    # save checkpoints after at every epoch
    torch.save(model.state_dict(), path.join(root_dir, 'model'))