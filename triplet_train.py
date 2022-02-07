import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import pytorch_metric_learning
from pytorch_metric_learning import losses, miners, samplers, trainers, testers
from pytorch_metric_learning.utils import common_functions
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

import numpy as np
import matplotlib.pyplot as plt
import umap
from cycler import cycler
import record_keeper
from PIL import Image

import logging
import argparse
import os
import sys
import glob
import time

import utils

logging.getLogger().setLevel(logging.INFO)
logging.info("VERSION %s"%pytorch_metric_learning.__version__)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--method', type=str)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--save', type=str, default='TRAIN')
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=96)
parser.add_argument('--num_workers', type=int, default=4)

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

class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set trunk model and replace the softmax layer with an identity function
trunk = torchvision.models.resnet18(pretrained=True)
trunk_output_size = trunk.fc.in_features
trunk.fc = common_functions.Identity()
trunk = torch.nn.DataParallel(trunk.to(device))

# Set embedder model. This takes in the output of the trunk and outputs 64 dimensional embeddings
embedder = torch.nn.DataParallel(MLP([trunk_output_size, 64]).to(device))

# Set optimizers
trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=0.00001, weight_decay=0.0001)
embedder_optimizer = torch.optim.Adam(embedder.parameters(), lr=0.0001, weight_decay=0.0001)

# Download the original datasets
original_train = datasets.CIFAR100(root="../data", train=True, transform=None, download=True)
original_val = datasets.CIFAR100(root="../data", train=False, transform=None, download=True)


# This will be used to create train and val sets that are class-disjoint
class ClassDisjointCIFAR100(torch.utils.data.Dataset):
    def __init__(self, original_train, original_val, train, transform):
        rule = (lambda x: x < 50) if train else (lambda x: x >= 50)
        train_filtered_idx = [i for i, x in enumerate(original_train.targets) if rule(x)]
        val_filtered_idx = [i for i, x in enumerate(original_val.targets) if rule(x)]
        self.data = np.concatenate([original_train.data[train_filtered_idx], original_val.data[val_filtered_idx]],
                                   axis=0)
        self.targets = np.concatenate(
            [np.array(original_train.targets)[train_filtered_idx], np.array(original_val.targets)[val_filtered_idx]],
            axis=0)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


# Class disjoint training and validation set
train_dataset = ClassDisjointCIFAR100(original_train, original_val, True, utils.transforms['cifar10_train'])
val_dataset = ClassDisjointCIFAR100(original_train, original_val, False, utils.transforms['cifar10_test'])
assert set(train_dataset.targets).isdisjoint(set(val_dataset.targets))



# Set the loss function
loss = losses.TripletMarginLoss(margin=0.1)

# Set the mining function
miner = miners.MultiSimilarityMiner(epsilon=0.1)

# Set the dataloader sampler
sampler = samplers.MPerClassSampler(train_dataset.targets, m=4, length_before_new_iter=len(train_dataset))

# Package the above stuff into dictionaries.
models = {"trunk": trunk, "embedder": embedder}
optimizers = {"trunk_optimizer": trunk_optimizer, "embedder_optimizer": embedder_optimizer}
loss_funcs = {"metric_loss": loss}
mining_funcs = {"tuple_miner": miner}


record_keeper, _, _ = logging_presets.get_record_keeper(os.path.join(args.save, "example_logs"),
                                                        os.path.join(args.save, "example_tensorboard"))
hooks = logging_presets.get_hook_container(record_keeper)
dataset_dict = {"val": val_dataset}
model_folder = os.path.join(args.save, "example_saved_models")

def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, *arguments):
    logging.info("UMAP plot for the {} split and label set {}".format(split_name, keyname))
    label_set = np.unique(labels)
    num_classes = len(label_set)
    fig = plt.figure(figsize=(20,15))
    plt.gca().set_prop_cycle(cycler("color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]))
    for i in range(num_classes):
        idx = labels == label_set[i]
        plt.plot(umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=1)
    plt.savefig(os.path.join(args.save, keyname+'.png'))

# Create the tester
tester = testers.GlobalEmbeddingSpaceTester(end_of_testing_hook = hooks.end_of_testing_hook,
                                            visualizer = umap.UMAP(),
                                            visualizer_hook = visualizer_hook,
                                            dataloader_num_workers = args.num_workers,
                                            accuracy_calculator=AccuracyCalculator(k="max_bin_count"))

end_of_epoch_hook = hooks.end_of_epoch_hook(tester,
                                            dataset_dict,
                                            model_folder,
                                            test_interval = 1)

trainer = trainers.MetricLossOnly(models,
                                optimizers,
                                args.batch_size,
                                loss_funcs,
                                mining_funcs,
                                train_dataset,
                                sampler=sampler,
                                dataloader_num_workers = args.num_workers,
                                end_of_iteration_hook = hooks.end_of_iteration_hook,
                                end_of_epoch_hook = end_of_epoch_hook)
trainer.train(num_epochs=args.epochs)