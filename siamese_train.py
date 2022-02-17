import os
import os.path as path
import sys
import time
import glob
import logging
import argparse
import random
from math import comb
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from torchvision.utils import make_grid
import torchvision.transforms as transforms

import dataloaders
import networks
import utils

################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--set', type=str, default='mnist', help='which dataset')
parser.add_argument('--n_max', type=int, default=1000)
parser.add_argument('--t_max', type=int, default=50)
parser.add_argument('--p_max', type=int, default=50)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--save', type=str, default='TRAIN', help='experiment name')
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--resume_dir', type=str)
parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--lr', type=float, default=0.1, help='adam/sgd learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--test_method', default='1')
args = parser.parse_args()


args.save = Path('runs', f'{args.save}-{time.strftime("%Y%m%d-%H%M%S")}')
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

# logging
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
# tensorboard writer
writer = SummaryWriter(args.save)


# use the value for each pair to compute the score at an eta value
def graph_cut(scores, eta):
    E_r = 0
    E_rc = 0
    E_r_count = 0
    E_rc_count = 0
    for d1 in range(ds.T):
        for d2 in range(ds.T):
            if (d1 < eta and d2 < eta) or (d1 >= eta and d2 >= eta):
                E_rc += scores[d1][d2]
                E_rc_count += 1
            else:
                E_r += scores[d1][d2]
                E_r_count += 1
    score = E_r / E_r_count - E_rc / E_rc_count

    return score


def compute_pair_values(model, ds, X):
    ps = np.zeros((ds.T, ds.T))
    gs = np.zeros((ds.T, ds.T))
    for d1 in range(ds.T):
        for d2 in range(ds.T):
            # grid = make_grid(torch.cat((X[d1].unsqueeze_(0), X[d2].unsqueeze_(0))), nrow=1)
            # save_image(grid, path.join(root_dir, 'X_{}_{}.png'.format(d1, d2)))
            p = model.forward(X[d1].unsqueeze_(0), X[d2].unsqueeze_(0))
            p = torch.sigmoid(p).item()
            if p == 1:
                p -= 0.0001
            ps[d1][d2] = p
            g = p / (1-p)
            gs[d1][d2] = g
    return ps, gs



def option2a(R, eta):
    r1, r2 = R[0:eta], R[eta:ds.T]
    diff1 = r1 - torch.mean(r1, 0)
    diff2 = r1 - torch.mean(r2, 0)

    return torch.sum(diff1.pow(2)).detach().item() + \
            torch.sum(diff2.pow(2)).detach().item()


def option2b(R, eta, alpha):
    first_term = 0
    second_term = 0
    third_term = 0

    T = R.size(0)
    for t in range(T):
        for tprime in range(T):
            if t <= eta and tprime >= eta+1:
                first_term += torch.sum(torch.pow(torch.abs(R[t]-R[tprime]), alpha))
            elif t <= eta and tprime <= eta:
                second_term += torch.sum(torch.pow(torch.sum(torch.abs(R[t] - R[tprime])), alpha))
            else:
                third_term += torch.sum(torch.pow(torch.sum(torch.abs(R[t] - R[tprime])), alpha))

    first_term *= 2/(eta*(T-eta))
    second_term /= comb(eta, 2)
    third_term /= comb(T-eta, 2)

    return first_term - second_term - third_term
    # return eta*(t-eta)/T * (first_term - second_term - third_term)


def validation_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for batch_index, (X, y) in enumerate(loader):
            X1, X2 = X[0], X[1]
            X1 = X1.to(device=device)
            X2 = X2.to(device=device)
            y = y.to(device=device)

            p = model.forward(X1, X2)
            p = p.view(p.size(0))
            p = torch.sigmoid(p)

            predictions = p >= 0.5
            num_correct += (predictions == y).sum().item()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


def ts_sample_precision(ps, eta):
    positive_count = 0
    negative_count = 0
    for d1 in range(ds.T):
        for d2 in range(ds.T):
            if (d1 < eta and d2 < eta) or (d1 >= eta and d2 >= eta):
                if ps[d1][d2] >= 0.5:
                    positive_count += 1
            else:
                if ps[d1][d2] < 0.5:
                    negative_count += 1

    negative_n = 2*2*eta*(ds.T-eta)
    positive_n = 2*ds.T*ds.T - negative_n
    return positive_count / positive_n, negative_count / negative_n



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def visualize():
    train_data_con = dataloaders.CON(args.set, 'train', n_max=args.n_max, t_max=args.t_max, p_max=args.p_max,
                                     classes=range(10), transform=utils.transforms[args.set])
    image = torch.cat([train_data_con[n][0] for n in range(0 * train_data_con.P, 1 * train_data_con.P)], dim=0)
    save_image(make_grid(image, nrow=train_data_con.P), path.join(args.save, 'CON.png'))

    image = torch.cat([train_data_con.get_x_n(n) for n in range(0, 1)], dim=0)
    save_image(make_grid(image, nrow=train_data_con.T), path.join(args.save, 'TS.png'))


def train(train_queue, model, criterion, optimizer, epoch):
    model.train()
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(train_queue):
        n = input.size(0)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        target = target.float()

        optimizer.zero_grad()
        output = model(input)
        output = torch.flatten(output)
        p = torch.sigmoid(output)
        logits = torch.stack((1-p, p), dim=1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        prec1, = utils.accuracy(logits, target, topk=(1,))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)

        if step % args.report_freq == 0:
            logging.info('train %03d loss %e top1 %f', step, objs.avg, top1.avg)
            writer.add_scalar('LossBatch/train', objs.avg, epoch * len(train_queue) + step)
            writer.add_scalar('AccuBatch/train', top1.avg, epoch * len(train_queue) + step)

        writer.add_scalar('LossEpoch/train', objs.avg, epoch)
        writer.add_scalar('AccuEpoch/train', top1.avg, epoch)
        if step % args.report_freq == 0 and args.debug:
            break

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion, epoch):
    model.eval()
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(valid_queue):
        n = input.size(0)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        target = target.float()

        output = model(input)
        output = torch.flatten(output)
        p = torch.sigmoid(output)
        logits = torch.stack((1 - p, p), dim=1)
        loss = criterion(output, target)

        prec1, = utils.accuracy(logits, target, topk=(1,))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)

        if step % args.report_freq == 0:
            logging.info('valid %03d loss %e top1 %f', step, objs.avg, top1.avg)
            writer.add_scalar('LossBatch/valid', objs.avg, epoch * len(valid_queue) + step)
            writer.add_scalar('AccuBatch/valid', top1.avg, epoch * len(valid_queue) + step)

        writer.add_scalar('LossEpoch/valid', objs.avg, epoch)
        writer.add_scalar('AccuEpoch/valid', top1.avg, epoch)
        if step % args.report_freq == 0 and args.debug:
            break

    return top1.avg, objs.avg


def main():
    ngpu = torch.cuda.device_count()
    logging.info('ngpu = %d', ngpu)

    logging.info("args = %s", args)

    set_seed(args.seed)
    cudnn.benchmark = True
    cudnn.enabled = True

    visualize()
    classes = {
        'mnist': range(10),
        'cifar10': range(10),
        'cifar100': range(100),
        'celeba': [20]
    }
    train_data_con = dataloaders.CON(args.set, 'train', n_max=args.n_max, t_max=args.t_max, p_max=args.p_max,
                                     classes=classes[args.set], transform=utils.transforms[args.set])
    valid_data_con = dataloaders.CON(args.set, 'valid', n_max=200, t_max=args.t_max, p_max=args.p_max,
                                     classes=classes[args.set], transform=utils.transforms[args.set])
    train_queue_con = DataLoader(train_data_con, shuffle=True, num_workers=args.num_workers, batch_size=args.batch_size)
    valid_queue_con = DataLoader(valid_data_con, shuffle=False, num_workers=args.num_workers, batch_size=args.batch_size)

    model = networks.SiameseNet(arch='cnn').cuda()
    criterion = nn.BCEWithLogitsLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    best_acc = 0.0
    is_best = False
    start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()

        logging.info('epoch %d', epoch)
        train_acc, train_obj = train(train_queue_con, model, criterion, optimizer, epoch)
        logging.info('train_acc %f train_loss %e \n', train_acc, train_obj)
        with torch.no_grad():
            valid_acc, valid_obj = infer(valid_queue_con, model, criterion, epoch)
        if valid_acc > best_acc:
            best_acc = valid_acc
            is_best = True
        else:
            is_best = False
        # scheduler.step()
        logging.info('valid_acc %f best_acc %f valid_loss %e\n', valid_acc, best_acc, valid_obj)

        utils.save_checkpoint({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()}, is_best, args.save)

        end_time = time.time()
        duration = end_time - start_time
        logging.info('Epoch time %ds', duration)


if __name__ == '__main__':
    main()


# 0 5_o_Clock_Shadow
    # 1 Arched_Eyebrows
    # 2 Attractive
    # 3 Bags_Under_Eyes
    # 4 - Bald
    # 5 Bangs
    # 6 Big_Lips
    # 7 Big_Nose
    # 8 Black_Hair
    # 9 - Blond_Hair
    # 10 Blurry
    # 11 Brown_Hair
    # 12 Bushy_Eyebrows
    # 13 Chubby
    # 14 Double_Chin
    # 15 - Eyeglasses
    # 16 Goatee
    # 17 Gray_Hair
    # 18 Heavy_Makeup
    # 19 High_Cheekbones
    # 20 - Male
    # 21 Mouth_Slightly_Open
    # 22 Mustache
    # 23 Narrow_Eyes
    # 24 No_Beard
    # 25 Oval_Face
    # 26 Pale_Skin
    # 27 Pointy_Nose
    # 28 Receding_Hairline
    # 29 Rosy_Cheeks
    # 30 Sideburns
    # 31 - Smiling
    # 32 Straight_Hair
    # 33 Wavy_Hair
    # 34 Wearing_Earrings
    # 35 Wearing_Hat
    # 36 Wearing_Lipstick
    # 37 Wearing_Necklace
    # 38 Wearing_Necktie
    # 39 Young
